import os
import requests
from services.pdf_service import get_pdf_store, get_embedder, load_pdf_if_needed

# Helper to select best PDF if pdf_id is 'auto' or not provided
def select_best_pdf(question):
    pdf_store = get_pdf_store()
    embedder = get_embedder()
    best_pdf_id = None
    best_score = -float('inf')
    question_vec = embedder.embed_query(question)
    for pdf_id, pdf_data in pdf_store.items():
        import numpy as np
        vectors = np.array(pdf_data["vectors"])
        sims = np.dot(vectors, question_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(question_vec) + 1e-8)
        # Take the average of the top 3 similarities for a more robust match
        top_n = 3
        if len(sims) > 0:
            top_similarities = np.sort(sims)[-top_n:]
            avg_sim = float(np.mean(top_similarities))
        else:
            avg_sim = -float('inf')
        if avg_sim > best_score:
            best_score = avg_sim
            best_pdf_id = pdf_id
    return best_pdf_id

def ask_question_service(pdf_id, question, top_k=3):
    from pathlib import Path
    import numpy as np
    import os
    embedder = get_embedder()
    pdf_store = get_pdf_store()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    # DEBUG: Print the environment variable directly
    print("[DEBUG] os.environ.get('GEMINI_API_KEY'):", os.environ.get('GEMINI_API_KEY'))
    print("[DEBUG] GEMINI_API_KEY variable:", GEMINI_API_KEY)
    # Auto-select PDF if needed
    if not pdf_id or pdf_id == 'auto':
        pdf_id = select_best_pdf(question)
        if not pdf_id:
            return {"error": "No PDFs available for search."}
    # Load PDF if not in memory
    if not load_pdf_if_needed(pdf_id):
        return {"error": "PDF not found"}
    pdf_data = pdf_store[pdf_id]
    question_vec = embedder.embed_query(question)
    vectors = np.array(pdf_data["vectors"])
    sims = np.dot(vectors, question_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(question_vec) + 1e-8)
    top_indices = np.argsort(sims)[-top_k:][::-1]
    top_sentences = [pdf_data["text"][i] for i in top_indices]
    # Limit total context to 2048 characters
    max_context_chars = 2048
    context = ""
    total_chars = 0
    for sent in top_sentences:
        if total_chars + len(sent) > max_context_chars:
            break
        context += sent + "\n"
        total_chars += len(sent)
    prompt = f"You are a legal assistant. Use the following context from New Zealand law to answer the question. Cite the most relevant sections and please do not hesitate to quote or exactly use fine amounts. Also, please do not explicitly say you were given these documents and act like you know them. \n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    # Call Gemini API
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    params = {"key": GEMINI_API_KEY}
    print("[DEBUG] GEMINI_API_KEY:", GEMINI_API_KEY)
    print("[DEBUG] GEMINI_API_URL:", GEMINI_API_URL)
    response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
    print("[DEBUG] Gemini API response status:", response.status_code)
    print("[DEBUG] Gemini API response text:", response.text)
    if response.status_code == 200:
        gemini_answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        gemini_answer = f"Error from Gemini: {response.text}"
    return {
        "answer": gemini_answer,
        "citations": top_sentences,
        "pdf_id": pdf_id
    }
