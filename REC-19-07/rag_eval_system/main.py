import argparse
import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from src.config import RetrieverConfig, IndexCacheConfig, RerankerConfig
from src.ingest import load_and_chunk_pdfs
from src.index_manager import try_load_index, save_index
from src.retriever import build_hybrid_retriever, build_rerank_retriever

def format_docs(docs):
    formatted = []
    for i, d in enumerate(docs):
        source = d.metadata.get("source", "Unknown")
        formatted.append(f"--- Document Source: {source} (Rank {i+1}) ---\n{d.page_content}")
    return "\n\n".join(formatted)

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Multi-Document RAG System")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to PDF directory")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    args = parser.parse_args()

    # 1. Load Configurations
    idx_config = IndexCacheConfig()
    ret_config = RetrieverConfig()
    rerank_config = RerankerConfig()

    # 2. Ingest Documents
    documents = load_and_chunk_pdfs(args.data_dir)
    if not documents:
        return

    # 3. Setup Embeddings & Vectorstore
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    vectorstore = try_load_index(documents, embeddings, idx_config.cache_dir)
    if vectorstore is None:
        print("Building new FAISS index...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        save_index(vectorstore, documents, idx_config.cache_dir)

    # 4. Build Pipeline (Ensemble -> Reranker)
    ensemble = build_hybrid_retriever(vectorstore, documents, ret_config)
    final_retriever = build_rerank_retriever(ensemble, rerank_config)

    # 5. Build LLM Chain
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert financial and aviation incident analyst. 
        Answer the question based ONLY on the following context. 
        If the context does not contain the answer, say "I cannot answer this based on the provided documents."

        Context:
        {context}

        Question: {question}

        At the end of your answer, provide a list of sources you used in the format: "SOURCE(S): doc1.pdf, doc2.pdf"
        """
    )
    
    chain = (
        {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Execute
    print(f"\nQuerying: {args.query}")
    print("Retrieving and generating answer...\n")
    
    response = chain.invoke(args.query)
    
    print("=== ANSWER ===")
    print(response)
    print("==============")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") # Suppress HF tokenizer warnings for clean CLI output
    main()