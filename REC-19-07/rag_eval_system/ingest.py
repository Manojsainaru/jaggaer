import os
import glob
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdfs(input_dir: str = "../data/sec10"):
    print(f"Ingesting PDFs from {input_dir}")
    pdf_files = glob.glob(os.path.join(input_dir, "**/*.pdf"), recursive=True)
    
    if not pdf_files:
        print("No PDF files found.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = []

    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            file_name = os.path.basename(pdf_path)
            for doc in docs:
                doc.metadata["source"] = file_name
                
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    print(f"Generated {len(all_splits)} chunks.")
    return all_splits