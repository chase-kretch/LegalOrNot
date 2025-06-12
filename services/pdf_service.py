from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from fastapi import UploadFile
from data.db_context import save_pdf_metadata, load_pdf_metadata

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pdf_store = {}  # {pdf_id: {"text": ..., "vectors": ...}}

def save_and_process_pdf(file, pdf_id):
    pdfs_dir = Path("pdfs")
    pdfs_dir.mkdir(exist_ok=True)
    pdf_path = pdfs_dir / pdf_id
    with open(pdf_path, "wb") as f:
        f.write(file)
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    sentences = [doc.page_content for doc in docs if doc.page_content.strip()]
    vectors = embedder.embed_documents(sentences)
    pdf_store[pdf_id] = {"text": sentences, "vectors": vectors}
    # Store in SQLite using db_context
    save_pdf_metadata(pdf_id, pdf_id, sentences, vectors)
    return {"pdf_id": pdf_id, "num_sentences": len(sentences)}

def upload_pdf_service(file: UploadFile):
    pdf_id = file.filename
    pdfs_dir = Path("pdfs")
    pdfs_dir.mkdir(exist_ok=True)
    pdf_path = pdfs_dir / pdf_id
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    sentences = [doc.page_content for doc in docs if doc.page_content.strip()]
    vectors = embedder.embed_documents(sentences)
    pdf_store[pdf_id] = {"text": sentences, "vectors": vectors}
    # Store in SQLite using db_context
    save_pdf_metadata(pdf_id, pdf_id, sentences, vectors)
    return {"pdf_id": pdf_id, "num_sentences": len(sentences)}

def load_pdf_if_needed(pdf_id):
    if pdf_id not in pdf_store:
        # Try to load from SQLite using db_context
        sentences, vectors = load_pdf_metadata(pdf_id)
        if sentences is not None and vectors is not None:
            pdf_store[pdf_id] = {"text": sentences, "vectors": vectors}
            return True
        pdf_path = Path("pdfs") / pdf_id
        if not pdf_path.exists():
            return False
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        sentences = [doc.page_content for doc in docs if doc.page_content.strip()]
        vectors = embedder.embed_documents(sentences)
        pdf_store[pdf_id] = {"text": sentences, "vectors": vectors}
    return True

def get_pdf_store():
    return pdf_store

def get_embedder():
    return embedder

def embed_texts(texts):
    return embedder.embed_documents(texts)

def list_pdfs_service():
    import sqlite3
    conn = sqlite3.connect('pdfs_metadata.db')
    c = conn.cursor()
    c.execute("SELECT filename FROM pdfs")
    pdfs = [row[0] for row in c.fetchall()]
    conn.close()
    return pdfs
