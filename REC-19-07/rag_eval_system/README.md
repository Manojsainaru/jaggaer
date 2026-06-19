# Multi-Document RAG System

This repository contains a robust Multi-Document Retrieval-Augmented Generation (RAG) and evaluation pipeline. It combines state-of-the-art open-source embeddings, hybrid retrieval, and cross-encoder reranking to accurately answer complex queries across extensive PDF documents (like SEC 10-Q filings).

## 🚀 Architecture Overview

The pipeline executes the following steps:

1. **Ingestion & Chunking (`ingest.py`)**: 
   - Uses Langchain's `PyPDFLoader` to read PDF documents from the `data/` directory.
   - Splits text using `RecursiveCharacterTextSplitter` while strictly preserving the source document's filename in the metadata.
2. **Intelligent Caching (`index_manager.py`)**:
   - Computes a SHA-256 hash of your document contents.
   - If the documents haven't changed, it automatically loads the cached FAISS index from disk, saving you massive amounts of time on re-embedding!
3. **Embeddings & Vector Store**: 
   - Embedded using **Qwen** (`Qwen/Qwen3-Embedding-0.6B`) running locally via PyTorch.
   - Stored in a local **FAISS** index for dense retrieval.
4. **Hybrid Retrieval (`retriver.py`)**: 
   - An **Ensemble Retriever** combines the semantic accuracy of Dense retrieval (FAISS) with the exact keyword matching of Sparse retrieval (BM25).
5. **Cross-Encoder Reranking (`retriver.py`)**: 
   - The combined chunks are passed through the **`BAAI/bge-reranker-base`** model to intelligently re-order the contexts, ensuring only the most highly relevant chunks make it to the LLM.
6. **Generation (`main.py` & `evaluate.py`)**: 
   - Passes the reranked context to **`gemini-2.5-flash`**. 
   - The LLM is strictly instructed to cite its sources at the end of the answer using the exact filenames (`SOURCE(S): doc1.pdf`).

---

## 📁 Detailed File Breakdown

Here is a deep dive into the specific responsibilities of each file in this repository:

### `main.py` (The Orchestrator)
This is the primary entry point for querying the RAG system interactively.
* **Responsibilities:**
  * Parses command-line arguments (e.g., `--data-dir`, `--query`).
  * Calls `ingest.py` to load and chunk the raw PDFs.
  * Calls `index_manager.py` to either load the cached FAISS index or build a new one using the Qwen embeddings.
  * Calls `retriver.py` to assemble the Hybrid Retriever (BM25 + FAISS) and wrap it with the BAAI Cross-Encoder reranker.
  * Constructs the final Langchain LCEL (LangChain Expression Language) pipeline bridging the retriever, the prompt template, and the Gemini LLM.
  * Executes the query and outputs the final cited answer to the console.

### `ingest.py` (Data Loader)
Handles the ingestion of raw documents into memory.
* **Key Function:** `load_and_chunk_pdfs(input_dir)`
* **Responsibilities:**
  * Uses `glob` to recursively find all `.pdf` files in the target directory.
  * Parses each PDF page-by-page using `PyPDFLoader`.
  * Extracts the raw filename (e.g., `2023 Q1 AAPL.pdf`) and forces it into the `source` metadata field of every single document chunk (critical for the LLM to know where the text came from).
  * Uses `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200) to break the pages down into manageable semantic chunks.

### `index_manager.py` (Vector Cache Manager)
Prevents the system from having to run the heavy embedding models every time you start the script.
* **Key Functions:** `try_load_index()`, `save_index()`, `compute_content_hash()`
* **Responsibilities:**
  * Scans all the loaded chunks and computes a deterministic `SHA-256` hash based on their text content.
  * Saves this hash in a `manifest.json` file alongside the FAISS database.
  * On subsequent runs, it compares the current hash to the saved hash. If they match, it skips embedding and instantly loads the FAISS index from disk. If they don't match (meaning you added/removed/edited a PDF), it triggers a complete rebuild.

### `retriver.py` (Search & Ranking Engine)
Assembles the multi-stage retrieval architecture.
* **Key Functions:** `build_hybrid_retriever()`, `build_rerank_retriever()`
* **Responsibilities:**
  * **Hybrid Retriever:** Combines `FAISS` (Dense Vector Search - finds chunks with similar meaning) and `BM25` (Sparse Keyword Search - finds chunks with exact word matches like specific ID numbers). Weights are controlled via `config.py`.
  * **Reranker:** Wraps the Ensemble Retriever in a `ContextualCompressionRetriever`. It passes the top chunks to the `BAAI/bge-reranker-base` cross-encoder, which scores how relevant each chunk is to the specific query and re-sorts them, dropping irrelevant context before it hits the LLM.

### `config.py` (Central Configuration)
A clean, centralized place to tweak the RAG hyperparameters without digging through code.
* **Responsibilities:**
  * Defines dataclasses for retrieval configuration.
  * Adjusts how many chunks FAISS and BM25 return initially (`faiss_k`, `bm25_k`).
  * Adjusts the weighting between dense and sparse retrieval (`faiss_weight`, `bm25_weight`).
  * Adjusts the final number of chunks the Cross-Encoder sends to the LLM (`rerank_top_k`).

### `evaluate.py` (Evaluation Suite)
A specialized script for benchmarking the RAG system against a ground-truth dataset.
* **Responsibilities:**
  * Loads an evaluation dataset (e.g., `qna_data.csv`).
  * Instantiates the exact same retrieval and generation pipeline used in `main.py`.
  * Iterates through a sample of questions, generates answers, and extracts the `SOURCE(S):` block from the LLM's output.
  * Automatically calculates **Citation Recall** (what percentage of the correct ground-truth documents were actually cited by the LLM).
  * Outputs a detailed `evaluation_results.csv` and prints summary statistics to the terminal.

---

## ⚙️ Installation & Setup

1. **Python Environment**: Ensure you are using the provided Python 3.10 virtual environment to avoid PyTorch C++ DLL errors.
2. **Activate Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set your API Key**:
   Create a `.env` file and add your Gemini API Key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

## 🧠 Usage

### Querying the System
You can execute a single query end-to-end using `main.py`. 

```bash
python main.py --data-dir ../data/sec10 --query "What is the total revenue?"
```

### Evaluating the System
Measure how perfectly the system retrieves and cites the correct documents based on a known Q&A CSV.

```bash
python evaluate.py --eval-csv qna_data.csv --num-samples 10
```
