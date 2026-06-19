from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.config import RetrieverConfig, RerankerConfig

def build_hybrid_retriever(vectorstore, documents, config: RetrieverConfig):
    print("Building Ensemble Retriever (FAISS + BM25)...")
    
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": config.faiss_k})
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = config.bm25_k
    
    ensemble = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[config.faiss_weight, config.bm25_weight],
    )
    return ensemble

def build_rerank_retriever(ensemble_retriever, config: RerankerConfig):
    print(f"Loading HuggingFace CrossEncoder ({config.model_name})...")
    
    model = HuggingFaceCrossEncoder(model_name=config.model_name)
    compressor = CrossEncoderReranker(model=model, top_n=config.rerank_top_k)
    
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=compressor
    )
    return compression_retriever