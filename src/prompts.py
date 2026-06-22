"""Prompt templates for the RAG graph.

Kept separate so the grounding behaviour (the most important part of this
assessment) can be reviewed and tuned in one place.
"""

# Rewrites a follow-up question into a standalone query using prior turns, so
# retrieval works for messages like "how do I turn it off?".
CONTEXTUALIZE_PROMPT = """Given the conversation so far and the user's latest \
message, rewrite the latest message as a standalone question that can be \
understood without the conversation history.

Only rewrite it to resolve references (pronouns, "it", "that feature", etc.). \
Do NOT answer the question. If the message is already standalone, return it \
unchanged."""

# The grounding contract. The model must answer ONLY from the retrieved
# context and must flag when the answer is not present.
SYSTEM_GROUNDED = """You are a helpful assistant that answers questions about \
the Apple iPhone User Guide (iOS 7.1). You must follow these rules without \
exception:

1. Answer ONLY using the information in the provided context excerpts below. \
Never use general knowledge or anything outside the context.
2. If the answer is not contained in the context, set "found" to false and \
reply that you could not find the answer in the document. Do not guess.
3. When you do answer, set "found" to true and list in "citations" the page \
numbers of the excerpts you actually used.
4. Be concise and accurate. Quote steps faithfully from the guide.

Context excerpts (each is tagged with its page number and section):
{context}"""


def format_context(docs) -> str:
    """Render retrieved chunks as page/section-tagged excerpts for the prompt."""
    blocks = []
    for d in docs:
        page = d.metadata.get("page", "?")
        section = d.metadata.get("section") or "Unknown section"
        blocks.append(f"[page {page} — {section}]\n{d.page_content}")
    return "\n\n".join(blocks)
