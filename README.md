# Legal or Not App NZ

A fullstack application for answering New Zealand law questions with citations from uploaded legal PDFs. Built with FastAPI (Python backend) and React (frontend).

## Features

- Upload legal PDFs (e.g from NZ legislation)
- Vectorize PDFs using LangChain and stores content for semantic search
- Ask legal questions and get answers with citations from most relevant PDF
- Automatic PDF selection based on context via LangChain
- Gemini LLM API for question answering
- React frontend with Vite for fast development

### Setup

#### Backend (FastAPI):

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add Gemini API key to `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
3. Run the FastAPI server:
   ```bash
   fastapi dev main.py
   ```
Access Swagger docs at `http://localhost:8000/docs` to test endpoints.

#### Frontend (React):

1. CD into my-react-app:
   ```bash
   cd my-react-app
   ```
2. Install Node.js dependencies:
   ```bash
    npm install
    ```
3. Start the React development server:
4. ```bash
   npm run dev
   ```
   
Access the app at `http://localhost:5173`.

### API Endpoints

- `POST /upload`: Upload a PDF file
- `POST /ask`: Ask a legal question
- `GET /list_pdfs`: List all uploaded documents

