"""
Prompt templates kept separate so the 'behavior' of the assistant is easy
to read and tune. The core instruction — answer ONLY from context and cite
sources — is what prevents hallucination and builds trust.
"""

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the "
    "provided context. If the answer is not in the context, say clearly: "
    "'I couldn't find that in the provided documents.' Do not use outside "
    "knowledge. Cite the source(s) you used by their [source] label."
)


def build_user_prompt(question: str, retrieved: list[dict]) -> str:
    """Assemble the context block + question into a single user message."""
    context_blocks = []
    for r in retrieved:
        label = f"[{r['source']} #chunk{r['chunk_id']}]"
        context_blocks.append(f"{label}\n{r['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above, and cite the [source] labels you used."
    )