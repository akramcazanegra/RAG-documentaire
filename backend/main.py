# backend/main.py
import os
from flask import Flask, request, jsonify
from pydantic import ValidationError
from models import AskRequest, AskResponse, HealthResponse
from rag_engine import RAGEngine

app = Flask(__name__)

DOCS_DIR = os.environ.get("RAG_DOCS_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "raw_documents"))
INDEX_DIR = os.environ.get("RAG_INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "index"))
os.makedirs(INDEX_DIR, exist_ok=True)

engine = RAGEngine(docs_dir=DOCS_DIR, index_dir=INDEX_DIR)

@app.get("/healthcheck")
def healthcheck():
    return jsonify(HealthResponse().model_dump())

@app.post("/ask")
def ask():
    try:
        payload = request.get_json(force=True)
        req = AskRequest(**payload)
    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    answer, sources = engine.answer(req.question, top_k=req.top_k)
    resp = AskResponse(answer=answer, sources=sources)
    return jsonify(resp.model_dump()), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)