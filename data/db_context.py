import sqlite3
import json
from pathlib import Path

DB_PATH = 'pdfs_metadata.db'

# Ensure the database and table exist

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pdfs (
        pdf_id TEXT PRIMARY KEY,
        filename TEXT,
        sentences TEXT,
        vectors TEXT
    )''')
    conn.commit()
    conn.close()

# CRUD operations

def save_pdf_metadata(pdf_id, filename, sentences, vectors):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("REPLACE INTO pdfs (pdf_id, filename, sentences, vectors) VALUES (?, ?, ?, ?)",
              (pdf_id, filename, json.dumps(sentences), json.dumps(vectors)))
    conn.commit()
    conn.close()

def load_pdf_metadata(pdf_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT sentences, vectors FROM pdfs WHERE pdf_id=?", (pdf_id,))
    row = c.fetchone()
    conn.close()
    if row:
        sentences = json.loads(row[0])
        vectors = json.loads(row[1])
        return sentences, vectors
    return None, None

init_db()

