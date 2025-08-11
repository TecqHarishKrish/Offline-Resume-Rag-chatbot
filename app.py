"""
Offline RAG Resume Search Chatbot (Streamlit)

- Uses PyPDF (pypdf) to read PDFs
- Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings (auto-downloaded)
- Uses FAISS (faiss-cpu) for vector search
- Uses HuggingFace small transformer (auto-downloaded) for generation (local, no API keys)
- No external API keys required
"""

import os
import tempfile
import pickle
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline, set_seed
from typing import List, Dict

# ========== CONFIG ==========
st.set_page_config(page_title="Offline Resume RAG Chatbot", layout="wide")
st.title("📁 Offline Resume RAG Chatbot (No API keys)")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"          # small, fast sentence-transformer (auto downloads)
GEN_MODEL_NAME = "facebook/opt-125m"           # small generation model (auto downloads) — swap if you like
EMBED_DIM = 384                                # embedding dimension for all-MiniLM-L6-v2
TOP_K = 4                                     # number of chunks to retrieve

# persistent storage filenames (optional)
INDEX_PATH = "faiss_index.bin"
DOC_STORE_PATH = "doc_store.pkl"

# ========== UTILITIES ==========

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name=EMBED_MODEL_NAME):
    st.info(f"Loading embedding model `{model_name}` (will download on first run) — please wait...")
    model = SentenceTransformer(model_name)
    return model

@st.cache_resource(show_spinner=False)
def load_generation_pipeline(model_name=GEN_MODEL_NAME):
    st.info(f"Loading generation model `{model_name}` (will download on first run) — please wait...")
    txt_gen = pipeline("text-generation", model=model_name, device=-1)  # device=-1 forces CPU
    # optional deterministic seed
    set_seed(0)
    return txt_gen

def pdf_to_text_chunks(file_path: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict]:
    """
    Extract text from PDF and split into overlapping chunks.
    Returns list of { "content": ..., "metadata": {...} }
    """
    reader = PdfReader(file_path)
    full_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception:
            text = ""
        if text:
            # simple cleaning
            text = text.replace("\n", " ").strip()
            if len(text) > 0:
                full_text.append(text)
    text = " ".join(full_text)
    # naive splitting by characters / words
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        start = end - chunk_overlap if end - chunk_overlap > start else end
    # return chunk dicts
    return [{"content": c} for c in chunks]

def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS index (IndexFlatL2) given embeddings (N x D numpy array)
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index

def save_index_and_docs(index, doc_store, index_path=INDEX_PATH, doc_path=DOC_STORE_PATH):
    faiss.write_index(index, index_path)
    with open(doc_path, "wb") as f:
        pickle.dump(doc_store, f)

def load_index_and_docs(index_path=INDEX_PATH, doc_path=DOC_STORE_PATH):
    if os.path.exists(index_path) and os.path.exists(doc_path):
        index = faiss.read_index(index_path)
        with open(doc_path, "rb") as f:
            doc_store = pickle.load(f)
        return index, doc_store
    return None, None

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    # returns numpy array (N x D)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    return embs

def retrieve_top_k(index, query_emb: np.ndarray, k=TOP_K):
    D, I = index.search(query_emb.astype(np.float32), k)
    # I is indices
    return I[0], D[0]

def generate_answer(gen_pipeline, context_text: str, question: str, max_new_tokens=256) -> str:
    prompt = (
        "You are an assistant that answers user questions using only the provided context. "
        "If the answer is not contained in the context, say 'I don't know from the provided resumes.'\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )
    # the pipeline returns list of dicts
    out = gen_pipeline(prompt, max_length= len(prompt.split()) + max_new_tokens, do_sample=True, num_return_sequences=1)
    answer = out[0]["generated_text"]
    # The generated_text includes the prompt — strip prompt portion
    if prompt in answer:
        answer = answer.split(prompt, 1)[-1].strip()
    return answer

# ========== MAIN UI ==========
st.markdown("**Upload multiple resume PDFs** — the first run may take a bit while models download automatically.")

# Load models (cached, auto-download on first run)
embed_model = load_embedding_model()
gen_pipeline = load_generation_pipeline()

# Allow user to optionally load existing index (persistence)
if os.path.exists(INDEX_PATH) and os.path.exists(DOC_STORE_PATH):
    if st.button("Load existing index & resumes from disk"):
        index, doc_store = load_index_and_docs()
        st.success("Loaded stored FAISS index and document store.")
    else:
        index, doc_store = None, None
else:
    index, doc_store = None, None

uploaded_files = st.file_uploader("Upload resume PDFs (multiple)", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    # process uploaded files: extract chunks, collect metadata
    st.info("Processing uploaded PDFs — extracting and chunking text...")
    all_chunks = []
    for uploaded in uploaded_files:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        chunks = pdf_to_text_chunks(tmp_path)
        # add metadata: original filename
        for c in chunks:
            c["metadata"] = {"filename": uploaded.name}
        all_chunks.extend(chunks)

    if not all_chunks:
        st.warning("No text extracted from the uploaded PDFs.")
    else:
        st.success(f"Extracted {len(all_chunks)} chunks from uploaded PDFs.")

        # embed chunks
        texts = [c["content"] for c in all_chunks]
        with st.spinner("Computing embeddings for chunks (this runs locally)..."):
            embs = embed_texts(embed_model, texts)  # N x D

        # build index
        index = build_faiss_index(embs)
        # store doc metadata and content in doc_store
        doc_store = {
            "texts": texts,
            "metadatas": [c.get("metadata", {}) for c in all_chunks],
            "embeddings_shape": embs.shape
        }

        # optional: save index & docs for reuse
        save_index_and_docs(index, doc_store)
        st.success("FAISS index built and saved locally. Ready to answer queries.")

# Chat / Query UI
st.markdown("---")
st.header("Query resumes")
query = st.text_input("Ask about a candidate or skills (e.g. 'Show details for Harishwar R' or 'Candidates with Python and AWS')")

if st.button("Search") and query:
    if index is None or doc_store is None:
        st.error("No index available. Upload resumes first (or load existing index).")
    else:
        # embed query
        with st.spinner("Embedding query..."):
            q_emb = embed_texts(embed_model, [query])

        # retrieve
        ids, dists = retrieve_top_k(index, q_emb, k=TOP_K)
        retrieved_texts = []
        sources = []
        for idx in ids:
            if idx < len(doc_store["texts"]):
                retrieved_texts.append(doc_store["texts"][idx])
                sources.append(doc_store["metadatas"][idx].get("filename", "Unknown file"))
        # build context
        context = "\n\n---\n\n".join([f"[{sources[i]}]\n{retrieved_texts[i]}" for i in range(len(retrieved_texts))])

        st.subheader("Retrieved source chunks")
        for i, txt in enumerate(retrieved_texts):
            st.markdown(f"**Source:** {sources[i]}")
            st.write(txt[:1000] + ("..." if len(txt) > 1000 else ""))
            st.markdown("---")

        # generate answer
        with st.spinner("Generating answer (local model)..."):
            answer = generate_answer(gen_pipeline, context, query)

        st.subheader("Answer")
        st.write(answer)

        # show raw context if requested
        if st.checkbox("Show full assembled context used for generation"):
            st.text(context)
