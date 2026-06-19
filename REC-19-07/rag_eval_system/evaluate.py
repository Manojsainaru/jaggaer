import argparse
import pandas as pd
import re
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
import os
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

def format_docs(docs):
    formatted = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        formatted.append(f"--- Document Source: {source} ---\n{d.page_content}")
    return "\n\n".join(formatted)

def extract_sources(text):
    """Extracts the list of sources from the end of the text."""
    match = re.search(r"SOURCE\(S\):\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        sources_str = match.group(1).replace('"', '')
        # Split by comma and clean up
        sources = [s.strip() for s in sources_str.split(",")]
        return set(sources)
    return set()

def evaluate_system(vector_dir: str, eval_csv: str, num_samples: int = 5):
    print(f"Loading vector store from {vector_dir}...")
    embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"trust_remote_code": True})
    vectorstore = FAISS.load_local(vector_dir, embeddings, allow_dangerous_deserialization=True)
    # Get all documents to initialize BM25
    all_docs = list(vectorstore.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 15

    # Base FAISS Retriever
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    # Ensemble Retriever (Sparse + Dense)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )

    # Cross Encoder Reranker
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=8)
    
    # Final Contextual Compression Retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """You are an expert financial and aviation incident analyst. 
Answer the question based ONLY on the following context, which includes markdown-formatted tables and text. 
If the context does not contain the answer, say "I cannot answer this based on the provided documents."

Context:
{context}

Question: {question}

At the end of your answer, provide a list of sources you used in the format: "SOURCE(S): doc1.pdf, doc2.pdf"
"""
    )
    chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    df = pd.read_csv(eval_csv)
    
    # We want to sample across different Question Types if possible
    if num_samples < len(df):
        # Sample stratifying by Question Type if it exists
        try:
            sample_df = df.groupby('Question Type', group_keys=False).apply(lambda x: x.sample(min(len(x), max(1, num_samples // len(df['Question Type'].unique())))))
        except:
            sample_df = df.sample(num_samples)
    else:
        sample_df = df

    results = []
    correct_citations = 0

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Evaluating"):
        question = row['Question']
        gt_answer = row['Answer']
        q_type = row.get('Question Type', 'Unknown')
        
        # Ground truth sources extracted from the provided answer
        gt_sources = extract_sources(gt_answer)
        
        # Predict
        predicted_answer = chain.invoke(question)
        pred_sources = extract_sources(predicted_answer)
        
        # Evaluate citations
        # We consider it a "hit" if the predicted sources contain ALL ground truth sources.
        # Alternatively, we can calculate recall. Let's do recall: |gt_sources intersect pred_sources| / |gt_sources|
        if len(gt_sources) > 0:
            recall = len(gt_sources.intersection(pred_sources)) / len(gt_sources)
        else:
            recall = 1.0 if len(pred_sources) == 0 else 0.0
            
        is_perfect = recall == 1.0
        if is_perfect:
            correct_citations += 1
            
        results.append({
            "Question": question,
            "Type": q_type,
            "Ground Truth Sources": ", ".join(gt_sources),
            "Predicted Sources": ", ".join(pred_sources),
            "Citation Recall": recall
        })

    print("\n=== EVALUATION RESULTS ===")
    print(f"Total Evaluated: {len(sample_df)}")
    print(f"Perfect Citation Recall: {correct_citations}/{len(sample_df)} ({correct_citations/len(sample_df)*100:.1f}%)")
    
    res_df = pd.DataFrame(results)
    res_df.to_csv("evaluation_results.csv", index=False)
    print("Detailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument("--vector-dir", type=str, default="vectorstore", help="Path to FAISS index")
    parser.add_argument("--eval-csv", type=str, required=True, help="Path to evaluation CSV")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of questions to evaluate")
    args = parser.parse_args()
    
    evaluate_system(args.vector_dir, args.eval_csv, args.num_samples)
