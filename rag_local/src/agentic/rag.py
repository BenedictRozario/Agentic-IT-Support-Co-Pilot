# src/agentic/rag.py
# Embedding + FAISS wrapper used by the demo

import os
from typing import List, Dict, Any, Tuple, Optional

# Try sentence-transformers first; fallback to openai if env present
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Embedder
# -------------------------
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.dim = 384
        # prefer sentence-transformers
        if _HAS_ST:
            try:
                self.model = SentenceTransformer(model_name)
                self.dim = self.model.get_sentence_embedding_dimension()
                print(f"[rag] Using SentenceTransformer '{model_name}' dim={self.dim}")
            except Exception as e:
                print("[rag] sentence-transformers init failed:", e)
                self.model = None

        # fallback to OpenAI if available
        if self.model is None and (_HAS_OPENAI and OPENAI_API_KEY):
            openai.api_key = OPENAI_API_KEY
            self.model = "openai"
            print("[rag] Falling back to OpenAI embeddings (remote)")

        if self.model is None:
            raise RuntimeError(
                "No embedding model available. Install sentence-transformers or set OPENAI_API_KEY."
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        if isinstance(self.model, SentenceTransformer):
            embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embs.tolist()
        elif self.model == "openai":
            out = []
            for t in texts:
                r = openai.Embedding.create(input=t, model=OPENAI_EMBED_MODEL)
                out.append(r["data"][0]["embedding"])
            return out
        else:
            raise RuntimeError("Embedder not properly initialized")

# -------------------------
# FAISS store wrapper
# -------------------------
class FaissStore:
    def __init__(self, dim: int):
        if not _HAS_FAISS:
            raise RuntimeError("faiss not available. Install faiss-cpu.")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        import numpy as np
        arr = np.array(vectors).astype("float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        arr = arr / norms
        self.index.add(arr)
        self.metadata.extend(metadatas)

    def search(self, vector: List[float], k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        import numpy as np
        v = np.array(vector).astype("float32")
        v = v / (np.linalg.norm(v) + 1e-9)
        D, I = self.index.search(v.reshape(1, -1), k)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.metadata):
                continue
            results.append((float(score), self.metadata[idx]))
        return results

# -------------------------
# Utilities: normalize + chunk
# -------------------------
def normalize(text: str) -> str:
    return " ".join(text.strip().split())

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# -------------------------
# Build vector store
# -------------------------
def build_vector_store(docs: List[Dict[str, Any]], embedder: Embedder) -> Tuple[Optional[FaissStore], List[Dict[str, Any]]]:
    chunks = []
    metadatas = []
    for d in docs:
        text = normalize(d.get("text", ""))
        cks = chunk_text(text, chunk_size=120, overlap=30)
        for i, c in enumerate(cks):
            chunks.append(c)
            metadatas.append({
                "doc_id": d.get("id"),
                "title": d.get("title", ""),
                "chunk_id": f"{d.get('id')}_c{i}",
                "chunk": c,
            })
    if not chunks:
        return None, []
    vectors = embedder.embed(chunks)
    dim = len(vectors[0])
    if not _HAS_FAISS:
        raise RuntimeError("FAISS not available. Install faiss-cpu.")
    store = FaissStore(dim)
    store.add(vectors, metadatas)
    return store, metadatas
