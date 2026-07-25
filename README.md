# 📄 RAG Document Assistant

Ask natural-language questions about your own documents and get answers that
are **grounded in the source material and cite where they came from** — not
hallucinated.

🔗 **Live demo:** _add your link here_
📸 _Add a screenshot of a Q&A with sources here._

---

## The problem
Teams sit on piles of documents — policies, contracts, manuals, reports — and
finding a specific answer means manually searching PDFs. Generic chatbots
hallucinate and can't reference internal material.

## The approach (Retrieval-Augmented Generation)
1. **Ingest & chunk** documents into overlapping passages.
2. **Embed** each chunk into a vector and store it.
3. On a question, **embed the query** and retrieve the most similar chunks
   via cosine similarity.
4. **Generate** an answer with an LLM constrained to *only* that context, and
   **cite the sources** used.

## Why it's trustworthy
- The model is instructed to answer **only from retrieved context** and to say
  so when the answer isn't present — this is the key guardrail against
  hallucination.
- Every answer shows the **source chunks** it used, so users can verify.

## Design note
Uses a lightweight in-memory vector store (NumPy cosine similarity) — zero
extra infrastructure and fully transparent. For production scale, the same
retrieval interface can be backed by a dedicated vector DB (FAISS, Chroma,
Pinecone).

## Tech stack
Python · Streamlit · OpenAI (embeddings + chat) · NumPy · pypdf

## Run locally
```bash
git clone https://github.com/deepanshug4/rag-document-assistant.git
cd rag-document-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # add your OPENAI_API_KEY
streamlit run app.py