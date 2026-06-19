from dataclasses import dataclass

@dataclass
class RetrieverConfig:
    faiss_k: int = 15
    bm25_k: int = 15
    faiss_weight: float = 0.5
    bm25_weight: float = 0.5

@dataclass
class IndexCacheConfig:
    cache_dir: str = "vectorstore"
    auto_load: bool = True

@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-base"
    rerank_top_k: int = 4