import hashlib
import json
import logging
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

_MANIFEST_FILE = "manifest.json"
_FAISS_INDEX_NAME = "index"

def compute_content_hash(documents: list[Document]) -> str:
    hasher = hashlib.sha256()
    for text in sorted(doc.page_content for doc in documents):
        hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()

def save_index(vectorstore: FAISS, documents: list[Document], cache_dir: str) -> None:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    vectorstore.save_local(str(path), index_name=_FAISS_INDEX_NAME)
    content_hash = compute_content_hash(documents)
    
    with open(path / _MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump({"content_hash": content_hash, "num_documents": len(documents)}, fh, indent=2)
    print(f"Index saved locally at {cache_dir}.")

def try_load_index(documents: list[Document], embeddings, cache_dir: str):
    path = Path(cache_dir)
    manifest_path = path / _MANIFEST_FILE
    
    if not manifest_path.exists():
        return None

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
        
    stored_hash = manifest.get("content_hash")
    current_hash = compute_content_hash(documents)
    
    if stored_hash != current_hash:
        print("Document content changed. Rebuilding index...")
        return None

    try:
        print("Loading cached FAISS index...")
        return FAISS.load_local(str(path), embeddings, index_name=_FAISS_INDEX_NAME, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None