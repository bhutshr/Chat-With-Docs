# 📋 Policy Document Assistant

 

A conversational AI assistant that lets you ask questions about policy documents and get grounded, accurate answers — powered by hybrid search and a local LLM.

 

---

 

## What Does This Do?

 

You drop in a PDF (like a company policies manual), the system chews through it, and spins up a chat interface where you can ask natural-language questions like:

 

> *"What's the travel reimbursement policy?"*

> *"Can you explain the conflict of interest rules?"*

 

It finds the most relevant sections from the document, feeds them to a local LLM, and gives you a properly sourced answer. It also suggests follow-up questions so you can keep exploring.

 

---

 

## How It Works (The Pipeline)

 

```

PDF Document

     │

     ▼

┌─────────────────────────┐

│  1. Text Extraction     │  (PyPDF2)

└────────┬────────────────┘

         ▼

┌─────────────────────────┐

│  2. Hierarchical        │  Detects headers (ALL CAPS = L1, Title Case = L2)

│     Header Chunking     │  and splits into structured sections

└────────┬────────────────┘

         ▼

┌─────────────────────────┐

│  3. Semantic            │  Splits large sections by paragraphs/sentences

│     Sub-Chunking        │  with overlap for context continuity

└────────┬────────────────┘

         ▼

┌─────────────────────────┐

│  4. FAISS Vector Store  │  Embeds chunks with all-MiniLM-L6-v2

│                         │  and stores in a local FAISS index

└─────────────────────────┘

```

 

When you ask a question:

 

```

User Query

     │

     ▼

┌───────────────────────────────────────┐

│  Hybrid Retrieval                     │

│  ┌──────────┐    ┌──────────┐        │

│  │  FAISS   │    │  BM25    │        │

│  │ (cosine) │    │(keyword) │        │

│  └────┬─────┘    └────┬─────┘        │

│       └───────┬───────┘              │

│         Reciprocal Rank Fusion       │

└───────────────┬───────────────────────┘

                ▼

┌───────────────────────────┐

│  Agent 1: Answer Agent    │  Generates a grounded answer

│  (Gemma 3n via Ollama)    │  citing specific policy sections

└───────────────┬───────────┘

                ▼

┌───────────────────────────┐

│  Agent 2: Follow-Up Agent │  Suggests 3 related questions

│  (Gemma 3n via Ollama)    │  to explore the topic further

└───────────────────────────┘

```

 

---

 

## Key Features

 

- **Hierarchical chunking** — respects document structure (headers, sub-headers) instead of blindly splitting every N characters

- **Hybrid retrieval** — combines dense vector search (FAISS + cosine similarity) with sparse keyword search (BM25), fused via Reciprocal Rank Fusion

- **Two-agent architecture** — Agent 1 answers with grounding; Agent 2 generates follow-up recommendations

- **Conversation memory** — stores chat history in a JSON file so follow-up questions have context from previous turns

- **Fully local** — runs entirely on your machine using Ollama. No data leaves your system.

 

---

 

## Tech Stack

 

| Component | Tool |

|---|---|

| LLM | Gemma 3n (`gemma3n:e2b`) via Ollama |

| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |

| Vector Store | FAISS (local, on-disk) |

| Keyword Search | BM25 (rank-bm25) |

| PDF Parsing | PyPDF2 |

| Framework | LangChain |

| UI | Streamlit |

| Language | Python 3.11 |

 

---

 

## Project Structure

 

```

Zycus/

├── injestion.py              # Ingestion pipeline (PDF → chunks → FAISS)

├── app.py                    # Streamlit chat app (retrieval + agents + UI)

├── Sample Policies.pdf       # The policy document

├── faiss_index/              # Persisted FAISS index (generated after ingestion)

├── conversation_memory.json  # Chat history (generated at runtime)

└── .venv/                    # Python virtual environment

```

 

---

 

## Setup & Run

 

### Prerequisites

 

- **Python 3.11+**

- **Ollama** installed and running with the `gemma3n:e2b` model

 

If you don't have Ollama set up yet:

 

```bash

# Install Ollama (macOS)

brew install ollama

 

# Pull the model

ollama pull gemma3n:e2b

 

# Make sure Ollama is running

ollama serve

```

 

### 1. Create a virtual environment and install dependencies

 

```bash

python3 -m venv .venv

source .venv/bin/activate

 

pip install langchain langchain-core langchain-community langchain-huggingface langchain-ollama \

            faiss-cpu sentence-transformers rank-bm25 PyPDF2 streamlit

```

 

### 2. Run the ingestion pipeline

 

This reads the PDF, chunks it, embeds it, and saves the FAISS index to disk.

 

```bash

python injestion.py

```

 

### 3. Launch the chat app

 

```bash

streamlit run app.py

```

 

This opens a browser tab at `http://localhost:8501` where you can start asking questions.

 
