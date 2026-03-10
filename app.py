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

AGENT1_SYSTEM_PROMPT = """You are a Policy Expert Assistant. Your job is to answer the user's question

accurately using ONLY the provided policy document excerpts as context.

 

STRICT RULES:

1. Base your answer ONLY on the provided context. Do not make up information.

2. If the context does not contain enough information, clearly state that.

3. Cite the relevant section/policy name when answering (use the header_path from metadata).

4. Be concise but thorough. Use bullet points for multi-part answers.

5. If the question is a follow-up, use the conversation history to understand what the user is referring to.

 

CONVERSATION HISTORY (for follow-up context):

{memory_context}

 

RETRIEVED POLICY CONTEXT:

{retrieval_context}

"""

 

 

def run_agent1(query: str) -> tuple[str, list[Document]]:

    """

    Agent 1: Retrieves relevant context via hybrid search, then generates

    a grounded answer using the LLM.

    Returns (answer_text, source_documents).

    """

    llm = get_llm()

 

    # Get conversation history for follow-up resolution

    memory_context = get_memory_context(last_n=5)

 

    # If this looks like a follow-up, enrich the query with context

    enriched_query = query

    if memory_context and _is_likely_followup(query):

        history = load_memory()

        if history:

            last_turn = history[-1]

            enriched_query = f"(Previous topic: {last_turn['query']}) {query}"

 

    # Hybrid retrieval

    docs = hybrid_retrieve(enriched_query)

 

    # Build context string with source attribution

    context_parts = []

    for i, doc in enumerate(docs):

        header = doc.metadata.get("header_path", "Unknown Section")

        context_parts.append(f"[Source {i+1}: {header}]\n{doc.page_content}")

    retrieval_context = "\n\n---\n\n".join(context_parts)

 

    # Build messages

    system_msg = AGENT1_SYSTEM_PROMPT.format(

        memory_context=memory_context if memory_context else "(No prior conversation)",

        retrieval_context=retrieval_context,

    )

    messages = [

        ("system", system_msg),

        ("human", query),

    ]

 

    response = llm.invoke(messages)

    return response.content, docs

 

 

def _is_likely_followup(query: str) -> bool:

    """Heuristic to detect follow-up questions."""

    followup_indicators = [

        "what about", "how about", "and ", "also", "more", "elaborate",

        "explain", "tell me more", "can you", "what else", "regarding that",

        "related to", "in addition", "furthermore", "the same",

        "it", "this", "that", "those", "these", "they",

    ]

    q_lower = query.lower().strip()

    return any(q_lower.startswith(ind) or ind in q_lower for ind in followup_indicators)

 

 

# ════════════════════════════════════════════════

# 5. AGENT 2: Recommendation Agent (follow-up Qs)

# ════════════════════════════════════════════════

 

AGENT2_SYSTEM_PROMPT = """You are a Follow-Up Question Generator for a policy document assistant.

 

Given the user's original question and the assistant's answer, generate exactly 3 relevant

follow-up questions that the user might want to ask next.

 

RULES:

1. Questions should be directly related to the topic discussed.

2. Questions should help the user explore the policy topic deeper.

3. Each question should be self-contained and clear.

4. Return ONLY the 3 questions, one per line, prefixed with 1., 2., 3.

5. Do NOT include any other text or explanation.

 

USER QUESTION: {query}

 

ASSISTANT ANSWER: {answer}

"""

 

 

def run_agent2(query: str, answer: str) -> list[str]:

    """

    Agent 2: Takes the output from Agent 1 and generates

    3 follow-up recommendation questions.

    """

    llm = get_llm()

 

    system_msg = AGENT2_SYSTEM_PROMPT.format(query=query, answer=answer)

    messages = [

        ("system", system_msg),

        ("human", "Generate 3 follow-up questions."),

    ]

 

    response = llm.invoke(messages)

    # Parse numbered questions

    lines = response.content.strip().split("\n")

    questions = []

    for line in lines:

        line = line.strip()

        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()

        if cleaned and len(cleaned) > 5:

            questions.append(cleaned)

    return questions[:3]

 

 

# ════════════════════════════════════════════════

# 6. STREAMLIT UI

# ════════════════════════════════════════════════

 

def main():

    st.set_page_config(

        page_title="Policy Assistant",

        page_icon="📋",

        layout="wide",

    )

 

    st.title("📋 Policy Document Assistant")

    st.caption("Hybrid RAG with FAISS + BM25 | Powered by Gemma 3n via Ollama")

 

    # ── Sidebar ──

    with st.sidebar:

        st.header("⚙️ Settings")

        st.markdown(f"**LLM Model:** `{LLM_MODEL}`")

        st.markdown(f"**Embedding:** `{EMBEDDING_MODEL}`")

        st.markdown(f"**Retrieval:** Hybrid (FAISS + BM25)")

        st.markdown("---")

 

        if st.button("🗑️ Clear Conversation History"):

            if os.path.exists(MEMORY_FILE):

                os.remove(MEMORY_FILE)

            st.session_state.messages = []

            st.rerun()

 

        st.markdown("---")

        st.markdown("### 📄 Indexed Document")

        st.markdown("Sample Policies and Procedures Manual v2.1")

 

    # ── Initialize session state ──

    if "messages" not in st.session_state:

        st.session_state.messages = []

    if "follow_ups" not in st.session_state:

        st.session_state.follow_ups = []

 

    # ── Display chat history ──

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "sources" in msg:

                with st.expander("📚 Sources"):

                    for src in msg["sources"]:

                        st.markdown(f"- **{src}**")

 

    # ── Follow-up question buttons ──

    if st.session_state.follow_ups:

        st.markdown("---")

        st.markdown("**💡 Suggested follow-up questions:**")

        cols = st.columns(len(st.session_state.follow_ups))

        for i, q in enumerate(st.session_state.follow_ups):

            if cols[i].button(q, key=f"followup_{i}_{q[:20]}"):

                st.session_state.follow_ups = []

                _process_query(q)

                st.rerun()

 

    # ── Chat input ──

    if user_input := st.chat_input("Ask about policies..."):

        st.session_state.follow_ups = []

        _process_query(user_input)

        st.rerun()

 

 

def _process_query(query: str):

    """Process a user query through Agent 1 → Agent 2 pipeline and update state."""

    # Add user message

    st.session_state.messages.append({"role": "user", "content": query})

 

    # ── Agent 1: Generate grounded answer ──

    with st.spinner("🔍 Searching policies & generating answer..."):

        answer, source_docs = run_agent1(query)

 

    source_headers = list({doc.metadata.get("header_path", "Unknown") for doc in source_docs})

 

    # ── Agent 2: Generate follow-up recommendations ──

    with st.spinner("💡 Generating follow-up suggestions..."):

        follow_ups = run_agent2(query, answer)

 

    # Store assistant message

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": source_headers,

    })

    st.session_state.follow_ups = follow_ups

 

    # Persist to memory file

    append_to_memory(query, answer, follow_ups, source_headers)

 

 

if __name__ == "__main__":

    main()
