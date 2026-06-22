"""LangGraph RAG pipeline: contextualize -> retrieve -> generate.

State flows through three nodes and is persisted per session by a checkpointer,
which is what gives us multi-turn memory. The graph is compiled once and reused
across requests; conversations are isolated by `thread_id`.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.prompts import CONTEXTUALIZE_PROMPT, SYSTEM_GROUNDED, format_context
from src.vectorstore import get_vector_store


class GroundedAnswer(BaseModel):
    """Structured output the chat model must return."""

    found: bool = Field(description="True only if the answer is in the context")
    answer: str = Field(description="The answer, or a message that it was not found")
    citations: list[int] = Field(
        default_factory=list, description="Page numbers of the excerpts actually used"
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]  # persisted conversation (multi-turn)
    question: str  # latest raw user question
    search_query: str  # standalone query after contextualization
    context: list[Document]  # retrieved chunks
    answer: str
    found: bool
    sources: list[dict]  # [{page, section}] for the UI, validated against context


def build_graph(settings: Settings | None = None):
    settings = settings or get_settings()

    vector_store = get_vector_store(settings)
    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        api_key=settings.openai_api_key,
    )
    structured_llm = llm.with_structured_output(GroundedAnswer)
    k = settings.retrieval_k

    def contextualize(state: State) -> dict:
        """Turn a follow-up into a standalone query using prior turns."""
        history = state["messages"][:-1]  # everything before the current question
        if not history:
            return {"search_query": state["question"]}
        msgs = (
            [SystemMessage(CONTEXTUALIZE_PROMPT)]
            + history
            + [HumanMessage(state["question"])]
        )
        rewritten = llm.invoke(msgs).content.strip()
        return {"search_query": rewritten or state["question"]}

    def retrieve(state: State) -> dict:
        docs = vector_store.similarity_search(state["search_query"], k=k)
        return {"context": docs}

    def generate(state: State) -> dict:
        docs = state["context"]
        system = SYSTEM_GROUNDED.format(context=format_context(docs))
        # Include prior turns so the answer is conversational, plus the question.
        msgs = (
            [SystemMessage(system)]
            + state["messages"][:-1]
            + [HumanMessage(state["question"])]
        )
        result: GroundedAnswer = structured_llm.invoke(msgs)

        # Hallucination guard: keep only citations that point at retrieved pages.
        retrieved_pages = {d.metadata.get("page") for d in docs}
        valid_pages = [p for p in result.citations if p in retrieved_pages]

        sources: list[dict] = []
        if result.found:
            seen = set()
            for d in docs:
                page = d.metadata.get("page")
                if page in valid_pages and page not in seen:
                    seen.add(page)
                    sources.append(
                        {"page": page, "section": d.metadata.get("section")}
                    )

        return {
            "answer": result.answer,
            "found": result.found,
            "sources": sources,
            "messages": [AIMessage(result.answer)],
        }

    builder = StateGraph(State)
    builder.add_node("contextualize", contextualize)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=MemorySaver())
