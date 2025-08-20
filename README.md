#  RAG Document   Ready-to-Use Project

This repository implements a **Retrieval-Augmented Generation (RAG)** system to query a document corpus (PDF/DOCX/TXT).

## 📁 Structure
```
rag-docs/
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── rag_engine.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── data/
│   └── raw_documents/    # place your documents here
└── README.md
```

##  Quickstart

### 1 Backend (Flask + FAISS)

#### Windows (PowerShell or CMD)
```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # duplicate env file and add your keys if needed
set FLASK_DEBUG=1
python main.py
```

Endpoints:
- `GET /healthcheck`
- `POST /rebuild_index`
- `POST /ask`  body JSON: `{ "question": "...", "top_k": 5, "temperature": 0.0 }`

> Without an OpenAI API key, the system will return a **fallback extractive** answer based on retrieved snippets.

### 2 Frontend (Streamlit)

#### Windows (PowerShell or CMD)
```powershell
cd ../frontend
pip install -r requirements.txt
set BACKEND_URL=http://localhost:8000
streamlit run app.py
```

##  Indexing
At the first startup, the backend will build a FAISS index from the files in `data/raw_documents`.  
You can rebuild the index anytime with `POST /rebuild_index` or the "Rebuild Index" button in the Streamlit frontend.

##  Technical Specs Covered
- Parsing of PDF/DOCX/TXT, cleaning, **chunking ~300–700 tokens**
- Embeddings with `sentence-transformers` (local) → vector store **FAISS**
- **Flask API** with **Pydantic** schemas
- **Streamlit frontend** (question, answer, sources)
- Bonus: rebuild button, environment variables, fallback without LLM

##  Notes
- Images inside PDFs are not OCR processed. For OCR, integrate Tesseract if needed.
- FAISS is fine for a corpus up to 1–2 GB. For production, consider Pinecone or Weaviate.
