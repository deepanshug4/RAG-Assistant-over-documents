"""
RAG Document Assistant — Streamlit UI.

Upload PDF/TXT documents, build a searchable index, and chat with your
documents. Answers are grounded in your files and cite their sources.
A low-confidence banner warns when the documents likely don't contain
a strong match — a guardrail against hallucinated answers.

Run locally:
    streamlit run app.py
"""

import streamlit as st

from src.ingest import build_chunks
from src.rag import build_index, answer_question, has_api_key

st.set_page_config(page_title="RAG Document Assistant", layout="wide")
st.title("📄 RAG Document Assistant")
st.caption("Upload documents and chat with them. Answers are grounded in your "
           "files and cite their sources.")

# --- API key notice ---
if not has_api_key():
    st.warning(
        "No OPENAI_API_KEY found. Document indexing and retrieval require an "
        "API key to generate embeddings and answers. Add one via a local .env "
        "file or Streamlit secrets."
    )

with st.expander("ℹ️ How it works"):
    st.markdown(
        "1. **Upload** PDF or TXT files and build the index.\n"
        "2. Documents are **chunked** and **embedded** into a vector index.\n"
        "3. Your question is embedded and matched against chunks "
        "(cosine similarity).\n"
        "4. The top matches are passed to the LLM, which answers **only** from "
        "that context and **cites sources** — reducing hallucination.\n\n"
        "*If no strong match is found, a low-confidence warning is shown.*"
    )

# --- Session state ---
# `store` holds the built vector index; `messages` holds chat history.
if "store" not in st.session_state:
    st.session_state.store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: document management ---
with st.sidebar:
    st.header("📚 Documents")
    uploaded = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("Build index", use_container_width=True):
        with st.spinner("Reading, chunking, and embedding documents..."):
            records = build_chunks(uploaded)
            if not records:
                st.error("No text could be extracted. If your PDF is a scanned "
                         "image, text extraction won't work without OCR.")
            else:
                st.session_state.store = build_index(records)
                st.success(
                    f"Indexed {len(records)} chunks from {len(uploaded)} file(s)."
                )

    # Status indicator
    if st.session_state.store is not None:
        st.info("✅ Index ready — ask a question in the chat.")
    else:
        st.caption("No index built yet.")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main: chat interface ---
st.subheader("💬 Chat with your documents")

# Render prior turns so the conversation persists across interactions.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about your documents")

if question:
    # Show and store the user's message.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if st.session_state.store is None:
            st.warning("Please upload documents and build the index first "
                       "(see the sidebar).")
        else:
            with st.spinner("Searching your documents..."):
                result = answer_question(st.session_state.store, question)

            # Guardrail: warn the user if retrieval was weak.
            if result.get("low_confidence"):
                st.warning(
                    "⚠️ Low confidence — your documents may not contain a strong "
                    "match for this question. The answer may be incomplete or "
                    "not supported by the source material."
                )

            st.write(result["answer"])

            # Persist the assistant's answer in history.
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"]}
            )

            # Transparency: show exactly which chunks informed the answer.
            sources = result.get("sources", [])
            if sources:
                with st.expander("🔎 Sources used"):
                    for r in sources:
                        st.markdown(
                            f"**{r['source']} — chunk {r['chunk_id']}** "
                            f"(similarity: {r['score']:.3f})"
                        )
                        preview = r["text"][:400]
                        st.caption(preview + ("..." if len(r["text"]) > 400 else ""))