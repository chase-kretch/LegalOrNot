from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv
import requests
from services.ask_service import ask_question_service
from models.ask import AskRequest
from services.pdf_service import upload_pdf_service, load_pdf_if_needed, list_pdfs_service
from data.db_context import init_db
import sqlite3

app = FastAPI()
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pdf_store = {}  # {pdf_id: {"text": ..., "vectors": ...}}

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Load all PDFs from the pdfs directory into memory on startup
pdfs_dir = "pdfs"
if os.path.exists(pdfs_dir):
    for fname in os.listdir(pdfs_dir):
        if fname.lower().endswith(".pdf"):
            load_pdf_if_needed(fname)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    return upload_pdf_service(file)

@app.get("/list_pdfs/")
async def list_pdfs():
    pdfs = list_pdfs_service()
    return {"pdfs": pdfs}

@app.post("/ask/")
async def ask_question(request: AskRequest):
    result = ask_question_service(request.pdf_id, request.question, request.top_k)
    return result
