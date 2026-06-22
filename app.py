"""Chainlit chat UI for the iPhone User Guide RAG chatbot.

Run locally:   chainlit run app.py -w
In Docker:     chainlit run app.py --host 0.0.0.0 --port $APP_PORT

The LangGraph pipeline is built once at startup so the container is ready to
answer immediately. Each browser session gets its own thread_id, which keys the
graph's checkpointer for isolated, multi-turn conversations.
"""

import uuid

import chainlit as cl
from langchain_core.messages import HumanMessage

from src.config import get_settings
from src.graph import build_graph

settings = get_settings()
graph = build_graph(settings)

WELCOME = (
    "👋 **iPhone User Guide assistant**\n\n"
    "Ask me anything about the **iPhone User Guide (iOS 7.1)**. "
    "I answer strictly from the document and cite the page(s) I used. "
    "If something isn't in the guide, I'll tell you instead of guessing.\n\n"
    "_Try: \"How do I set up Touch ID?\" or \"How do I take a panorama photo?\"_"
)


def _format_sources(sources: list[dict]) -> str:
    """Render validated page/section citations as an inline footer."""
    if not sources:
        return ""
    lines = ["\n\n---", "**📄 Sources**"]
    for s in sources:
        section = s.get("section")
        suffix = f" — {section}" if section else ""
        lines.append(f"- Page {s['page']}{suffix}")
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    # graph.invoke is sync; run it off the event loop so the UI stays responsive.
    state = await cl.make_async(graph.invoke)(
        {"messages": [HumanMessage(message.content)], "question": message.content},
        config,
    )

    content = state["answer"]
    if state.get("found"):
        content += _format_sources(state.get("sources", []))

    await cl.Message(content=content).send()
