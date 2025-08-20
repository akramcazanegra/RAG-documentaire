# backend/rag_engine.py
import os, re, glob, json
from typing import List, Tuple
import numpy as np

# Embeddings & Index
from sentence_transformers import SentenceTransformer
import faiss

# Document parsing
import docx2txt
from pypdf import PdfReader

# Optional LLM (OpenAI)
from typing import Optional
import os

from models import SourceChunk

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(path: str) -> str:
    return docx2txt.process(path) or ""

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return text

def load_documents(docs_dir: str) -> List[Tuple[str, str]]:
    paths = []
    for ext in ("*.pdf", "*.docx", "*.txt"):
        paths.extend(glob.glob(os.path.join(docs_dir, "**", ext), recursive=True))
    texts = []
    for fp in paths:
        try:
            if fp.lower().endswith(".txt"):
                raw = read_txt(fp)
            elif fp.lower().endswith(".docx"):
                raw = read_docx(fp)
            elif fp.lower().endswith(".pdf"):
                raw = read_pdf(fp)
            else:
                continue
            texts.append((os.path.basename(fp), normalize_space(raw)))
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    return texts

def simple_sentence_split(text: str) -> List[str]:
    return re.split(r'(?<=[\.!?])\s+', text)

def chunk_text(text: str, target_words: int = 350, max_words: int = 500) -> List[str]:
    sentences = simple_sentence_split(text)
    chunks, current, count = [], [], 0
    for s in sentences:
        words = s.split()
        if not words:
            continue
        if count + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(s)
        count += len(words)
        if count >= target_words:
            chunks.append(" ".join(current))
            current, count = [], 0
    if current:
        chunks.append(" ".join(current))
    # filter too-short chunks
    return [c for c in chunks if len(c.split()) > 30]

class RAGEngine:
    def __init__(self, docs_dir: str, index_dir: str):
        self.docs_dir = os.path.abspath(docs_dir)
        self.index_dir = os.path.abspath(index_dir)
        os.makedirs(self.index_dir, exist_ok=True)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index_path = os.path.join(self.index_dir, "faiss.index")
        self.meta_path = os.path.join(self.index_dir, "meta.json")
        self.index = None
        self.id2meta = []
        self._load_or_build()

    def _build(self):
        docs = load_documents(self.docs_dir)
        id2meta = []
        vectors = []
        for fname, text in docs:
            for ch in chunk_text(text):
                id2meta.append({"file": fname, "chunk": ch})
                vectors.append(self.model.encode(ch, show_progress_bar=False, normalize_embeddings=True))
        if not vectors:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            mat = np.vstack(vectors).astype("float32")
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(mat)
        self.id2meta = id2meta
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.id2meta, f, ensure_ascii=False)

    def _load_or_build(self):
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.id2meta = json.load(f)
            else:
                self._build()
        except Exception as e:
            print("[WARN] Rebuilding index due to error:", e)
            self._build()

    def _ensure_index(self):
        # If no docs were present at build time but user adds later
        if self.index is None or (self.index.ntotal == 0 and any(Path(self.docs_dir).glob('**/*'))):
            self._build()

    def _gen_with_openai(self, question: str, contexts: List[str]) -> Optional[str]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                "You are a helpful assistant. Answer the user's question using only the CONTEXT.\n"
                "If the answer is not in the context, say you don't know.\n\n"
                f"QUESTION: {question}\n\nCONTEXT:\n" + "\n---\n".join(contexts[:5])
            )
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("[WARN] OpenAI generation failed:", e)
            return None

    def retrieve(self, query: str, top_k: int = 4):
        self._ensure_index()
        if self.index is None or self.index.ntotal == 0:
            return []
        vec = self.model.encode(query, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(vec.reshape(1, -1), top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx == -1:
                continue
            meta = self.id2meta[idx]
            results.append((idx, float(score)))
        return results

    def answer(self, question: str, top_k: int = 4):
        hits = self.retrieve(question, top_k=top_k)
        contexts, sources = [], []
        for idx, score in hits:
            meta = self.id2meta[idx]
            sources.append(SourceChunk(file=meta["file"], chunk=meta["chunk"][:1200], score=score))
            contexts.append(f"({meta['file']}) {meta['chunk']}")
        # Try LLM
        gen = self._gen_with_openai(question, contexts)
        if gen is None:
            if not contexts:
                return ("No documents indexed yet. Add files to data/raw_documents/ then restart the backend.", [])
            joined = "\n\n".join(contexts)[:1500]
            gen = f"(No LLM configured) Relevant excerpts:\n\n{joined}"
        return gen, sources