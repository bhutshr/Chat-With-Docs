import os

import re

import json

from datetime import datetime

 

# Handle SSL certificate issues (corporate proxy with self-signed certs)

os.environ["CURL_CA_BUNDLE"] = ""

os.environ["REQUESTS_CA_BUNDLE"] = ""

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

 

import streamlit as st

from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama

 

# ════════════════════════════════════════════════

# CONFIGURATION

# ════════════════════════════════════════════════

 

FAISS_INDEX_DIR = "faiss_index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "gemma3n:e2b"

MEMORY_FILE = "conversation_memory.json"

TOP_K_FAISS = 5

TOP_K_BM25 = 5

TOP_K_FINAL = 5

 

# ════════════════════════════════════════════════

# 1. CONVERSATION MEMORY (persistent JSON file)

# ════════════════════════════════════════════════

 

def load_memory() -> list[dict]:

    """Load conversation history from disk."""

    if os.path.exists(MEMORY_FILE):

        with open(MEMORY_FILE, "r") as f:

            return json.load(f)

    return []

 

def save_memory(history: list[dict]):

    """Persist conversation history to disk."""

    with open(MEMORY_FILE, "w") as f:

        json.dump(history, f, indent=2, default=str)

 

def append_to_memory(query: str, answer: str, follow_ups: list[str], sources: list[str]):

    """Append a single Q&A turn to persistent memory."""

    history = load_memory()

    history.append({

        "timestamp": datetime.now().isoformat(),

        "query": query,

        "answer": answer,

        "follow_up_questions": follow_ups,

        "sources": sources,

    })

    # Keep last 50 turns to avoid unbounded growth

    if len(history) > 50:

        history = history[-50:]

    save_memory(history)

 

def get_memory_context(last_n: int = 5) -> str:

    """Build a text summary of recent conversation for follow-up context."""

    history = load_memory()

    if not history:

        return ""

    recent = history[-last_n:]

    lines = []

    for turn in recent:

        lines.append(f"User: {turn['query']}")

        lines.append(f"Assistant: {turn['answer'][:500]}")

    return "\n".join(lines)

 

# ════════════════════════════════════════════════

# 2. HYBRID RETRIEVAL (FAISS cosine + BM25)

# ════════════════════════════════════════════════

 

@st.cache_resource

def load_faiss_index():

    """Load the persisted FAISS vector store."""

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    return vectorstore

 

@st.cache_resource

def build_bm25_index():

    """Build a BM25 index from all documents in FAISS store."""

    vectorstore = load_faiss_index()

    all_docs = list(vectorstore.docstore._dict.values())

    tokenized_corpus = [_tokenize(doc.page_content) for doc in all_docs]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, all_docs

 

def _tokenize(text: str) -> list[str]:

    """Simple whitespace + lowering tokenizer for BM25."""

    text = re.sub(r"[^\w\s]", " ", text.lower())

    return text.split()

 

def hybrid_retrieve(query: str, k: int = TOP_K_FINAL) -> list[Document]:

    """

    Hybrid retrieval combining:

      - FAISS cosine similarity (dense vector search)

      - BM25 (sparse keyword search)

    Results are fused using Reciprocal Rank Fusion (RRF).

    """

    vectorstore = load_faiss_index()

    bm25, all_docs = build_bm25_index()

 

    # --- Dense retrieval (FAISS cosine similarity) ---

    faiss_results = vectorstore.similarity_search_with_score(query, k=TOP_K_FAISS)

    faiss_ranked = [(doc, rank) for rank, (doc, _score) in enumerate(faiss_results)]

 

    # --- Sparse retrieval (BM25) ---

    query_tokens = _tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_BM25]

    bm25_ranked = [(all_docs[i], rank) for rank, i in enumerate(top_bm25_indices)]

 

    # --- Reciprocal Rank Fusion ---

    RRF_K = 60

    doc_scores: dict[str, float] = {}

    doc_map: dict[str, Document] = {}

 

    for doc, rank in faiss_ranked:

        key = doc.page_content[:200]

        doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

        doc_map[key] = doc

 

    for doc, rank in bm25_ranked:

        key = doc.page_content[:200]

        doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)

        doc_map[key] = doc

 

    # Sort by fused score and return top-k

    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)[:k]

    return [doc_map[key] for key in sorted_keys]

 

# ════════════════════════════════════════════════

# 3. LLM SETUP

# ════════════════════════════════════════════════

 

@st.cache_resource

def get_llm():

    """Initialize the Ollama LLM."""

    return ChatOllama(model=LLM_MODEL, temperature=0)

 

# ════════════════════════════════════════════════

# 4. AGENT 1: RAG Answer Agent (grounded retrieval)

# ════════════════════════════════════════════════
