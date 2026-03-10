import re

import os

from typing import Optional

from PyPDF2 import PdfReader

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

 

 

# ──────────────────────────────────────────────

# 1. PDF Text Extraction

# ──────────────────────────────────────────────

 

def extract_text_from_pdf(pdf_path: str) -> str:

    """Extract full text from a PDF file, preserving page breaks."""

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:

            pages.append(text.strip())

    return "\n\n".join(pages)

 

 

# ──────────────────────────────────────────────

# 2. Header Detection & Hierarchical Chunking

# ──────────────────────────────────────────────

 

# Patterns for different header levels observed in the document:

# Level 1 (ALL CAPS, standalone):  PURPOSE, PERSONNEL POLICIES, PROCUREMENT POLICIES, etc.

# Level 2 (Title Case, often multi-word): Policy on Affirmative Action, Employment, Vacancies, etc.

# Level 3 (Sub-sub sections): Statement, Compliant Procedure, Procedures, etc.

 

LEVEL1_PATTERN = re.compile(

    r"^([A-Z][A-Z /&\-]{4,})$", re.MULTILINE

)

 

LEVEL2_PATTERN = re.compile(

    r"^([A-Z][a-zA-Z]+(?:\s+(?:of|on|and|the|in|for|at|or|to|an|a|&|-|/)*\s*[A-Za-z]+){1,10})\s*$",

    re.MULTILINE,

)

 

# Known top-level section headers from the document (used for precise matching)

KNOWN_L1_HEADERS = {

    "PURPOSE",

    "PERSONNEL POLICIES",

    "PROCUREMENT POLICIES",

    "PROPERTY/EQUIPMENT STANDARDS",

    "RECORDS MANAGEMENT POLICY",

    "COMPUTER AND INTERNET SECURITY",

    "ACKNOWLEDGEMENT FORM",

}

 

# Known Level-2 sub-section headers

KNOWN_L2_HEADERS = {

    "Policy on Affirmative Action/Equal Employment Opportunity",

    "Compliant Procedure",

    "Employment",

    "Employment at Will",

    "Employment Status",

    "Attendance",

    "Work Schedule and Pay Periods",

    "Pay Practices",

    "Outside Employment",

    "Conflict of Interest",

    "Policy Prohibiting Unlawful Harassment, Including Sexual Harassment",

    "Employee Performance Evaluations",

    "Disciplinary Procedures",

    "Corrective Action Process",

    "Termination Policy",

    "Vacations",

    "Holidays",

    "Personal Leave",

    "Sick Leave",

    "Bereavement Leave",

    "Jury Duty",

    "Leave of Absence",

    "Worker's Compensation",

    "Health Insurance",

    "Pension/Retirement Plan",

    "Travel Policy",

    "Introduction and Scope",

    "Code of Conduct",

    "Requirements and Protocol",

    "Disciplinary Actions",

    "Acquisition Planning",

    "Solicitations for Goods and Services",

    "Procurement Instruments",

    "Contract Cost and Price and Other Selection Criteria",

    "Contractor Monitoring",

    "Procurement Records",

    "Contract Provisions and Bonding Requirements Award and Administration",

    "Records Management Policy",

    "Drug-Free Work Place Policy",

    "The Internet and e-mail",

    "Computer viruses",

    "Access and passwords",

    "Physical security",

    "Employee responsibilities",

    "Copyrights and license agreements",

    "Budget Principles/Procedures",

    "Audit Procedure",

    "Whistleblower Policy",

}

 

 

def classify_header(line: str) -> Optional[int]:

    """Return header level (1, 2) or None if the line is not a header."""

    stripped = line.strip()

    if not stripped or len(stripped) < 3:

        return None

 

    # Skip page footers / headers like "Sample Policies and Procedures Manual v 2.1"

    if "Sample Policies and Procedures Manual" in stripped:

        return None

    if re.match(r"^Page \d+ of \d+", stripped):

        return None

 

    # Check against known L1 headers

    if stripped.upper() in {h.upper() for h in KNOWN_L1_HEADERS}:

        return 1

 

    # Check ALL-CAPS pattern for L1

    if LEVEL1_PATTERN.match(stripped) and len(stripped.split()) <= 8:

        return 1

 

    # Check against known L2 headers

    for h in KNOWN_L2_HEADERS:

        if stripped.lower().startswith(h.lower()):

            return 2

 

    return None

 

 

def hierarchical_chunk(full_text: str) -> list[dict]:

    """

    Split text into hierarchical chunks based on detected headers.

 

    Returns a list of dicts:

      {

        "level1_header": str,   # top-level section

        "level2_header": str,   # sub-section (may be same as level1 if none)

        "content": str,         # the actual text content

        "header_path": str,     # e.g. "PERSONNEL POLICIES > Employment > Hiring"

      }

    """

    lines = full_text.split("\n")

    chunks = []

 

    current_l1 = "GENERAL"

    current_l2 = ""

    current_content_lines: list[str] = []

 

    def flush_chunk():

        content = "\n".join(current_content_lines).strip()

        # Remove table-of-contents lines (lines with many dots)

        content = re.sub(r"^.*\.{5,}.*$", "", content, flags=re.MULTILINE)

        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        if content and len(content) > 30:

            header_path = current_l1

            if current_l2 and current_l2 != current_l1:

                header_path += f" > {current_l2}"

            chunks.append({

                "level1_header": current_l1,

                "level2_header": current_l2 or current_l1,

                "content": content,

                "header_path": header_path,

            })

 

    for line in lines:

        level = classify_header(line)

 

        if level == 1:

            flush_chunk()

            current_l1 = line.strip()

            current_l2 = ""

            current_content_lines = []

        elif level == 2:

            flush_chunk()

            current_l2 = line.strip()

            current_content_lines = []

        else:

            current_content_lines.append(line)

 

    # Flush the last chunk

    flush_chunk()

    return chunks

 

 

# ──────────────────────────────────────────────

# 3. Semantic Sub-Chunking

# ──────────────────────────────────────────────

 

def semantic_subchunk(text: str, max_chunk_size: int = 1000, overlap: int = 150) -> list[str]:

    """

    Split a text block into semantically meaningful sub-chunks.

 

    Strategy:

    - Split on paragraph boundaries (double newlines) first.

    - If a paragraph is still too large, split on sentence boundaries.

    - Maintain overlap between chunks for context continuity.

    """

    paragraphs = re.split(r"\n\s*\n", text)

    sub_chunks: list[str] = []

    current_chunk = ""

 

    for para in paragraphs:

        para = para.strip()

        if not para:

            continue

 

        # If adding this paragraph stays within limit, accumulate

        if len(current_chunk) + len(para) + 2 <= max_chunk_size:

            current_chunk = f"{current_chunk}\n\n{para}".strip()

        else:

            # Flush current chunk

            if current_chunk:

                sub_chunks.append(current_chunk)

 

            # If the paragraph itself is too large, split by sentences

            if len(para) > max_chunk_size:

                sentences = re.split(r"(?<=[.!?])\s+", para)

                current_chunk = ""

                for sent in sentences:

                    if len(current_chunk) + len(sent) + 1 <= max_chunk_size:

                        current_chunk = f"{current_chunk} {sent}".strip()

                    else:

                        if current_chunk:

                            sub_chunks.append(current_chunk)

                        current_chunk = sent

            else:

                current_chunk = para

 

    if current_chunk:

        sub_chunks.append(current_chunk)

 

    # Apply overlap: prepend tail of previous chunk to next chunk

    if overlap > 0 and len(sub_chunks) > 1:

        overlapped = [sub_chunks[0]]

        for i in range(1, len(sub_chunks)):

            prev_tail = sub_chunks[i - 1][-overlap:]

            # Find a clean word boundary for overlap

            space_idx = prev_tail.find(" ")

            if space_idx != -1:

                prev_tail = prev_tail[space_idx + 1:]

            overlapped.append(f"...{prev_tail}\n\n{sub_chunks[i]}")

        sub_chunks = overlapped

 

    return sub_chunks

 

 

# ──────────────────────────────────────────────

# 4. Full Pipeline: PDF → Hierarchical Chunks → FAISS

# ──────────────────────────────────────────────

 

def build_documents(pdf_path: str, max_chunk_size: int = 1000, overlap: int = 150) -> list[Document]:

    """

    End-to-end: extract PDF → hierarchical header chunking → semantic sub-chunking → LangChain Documents.

    """

    print(f"[1/4] Extracting text from: {pdf_path}")

    full_text = extract_text_from_pdf(pdf_path)

    print(f"       Extracted {len(full_text):,} characters")

 

    print("[2/4] Performing hierarchical header-based chunking...")

    header_chunks = hierarchical_chunk(full_text)

    print(f"       Found {len(header_chunks)} header-based sections")

 

    print("[3/4] Applying semantic sub-chunking within each section...")

    documents: list[Document] = []

    for chunk in header_chunks:

        sub_chunks = semantic_subchunk(chunk["content"], max_chunk_size, overlap)

        for j, sc in enumerate(sub_chunks):

            doc = Document(

                page_content=sc,

                metadata={

                    "level1_header": chunk["level1_header"],

                    "level2_header": chunk["level2_header"],

                    "header_path": chunk["header_path"],

                    "sub_chunk_index": j,

                    "total_sub_chunks": len(sub_chunks),

                    "source": os.path.basename(pdf_path),

                },

            )

            documents.append(doc)

 

    print(f"       Total document chunks: {len(documents)}")

    return documents

 

 

def ingest_to_faiss(

    pdf_path: str,

    persist_dir: str = "faiss_index",

    embedding_model: str = "all-MiniLM-L6-v2",

    max_chunk_size: int = 1000,

    overlap: int = 150,

) -> FAISS:

    """

    Full ingestion pipeline:

      PDF → Header-based hierarchical chunking → Semantic sub-chunking → FAISS vector store.

    """

    documents = build_documents(pdf_path, max_chunk_size, overlap)

 

    print(f"[4/4] Embedding {len(documents)} chunks and building FAISS index...")

    print(f"       Using embedding model: {embedding_model}")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = FAISS.from_documents(documents, embeddings)

 

    # Persist to disk

    vectorstore.save_local(persist_dir)

    print(f"       FAISS index saved to: {persist_dir}/")

 

    return vectorstore

 

 

# ──────────────────────────────────────────────

# 5. Query utility (for verification)

# ──────────────────────────────────────────────

 

def query_faiss(query: str, persist_dir: str = "faiss_index", k: int = 5) -> list[Document]:

    """Load a persisted FAISS index and perform a similarity search."""

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

    results = vectorstore.similarity_search(query, k=k)

    return results

 

 

# ──────────────────────────────────────────────

# Main

# ──────────────────────────────────────────────

 

if __name__ == "__main__":

    PDF_PATH = "Sample Policies.pdf"

    PERSIST_DIR = "faiss_index"

 

    # ── Run ingestion ──

    vectorstore = ingest_to_faiss(PDF_PATH, PERSIST_DIR)

 

    # ── Verify with a sample query ──

    print("\n" + "=" * 60)

    print("VERIFICATION: Sample queries against the FAISS index")

    print("=" * 60)

 

    test_queries = [

        "What is the sexual harassment policy?",

        "What are the procurement methods?",

        "What is the travel reimbursement policy?",

        "What are the rules around conflict of interest?",

    ]

 

    for q in test_queries:

        print(f"\n📌 Query: {q}")

        results = query_faiss(q, PERSIST_DIR, k=2)

        for i, doc in enumerate(results):

            print(f"  [{i+1}] Section: {doc.metadata['header_path']}")

            print(f"      Preview: {doc.page_content[:150]}...")

            print()
