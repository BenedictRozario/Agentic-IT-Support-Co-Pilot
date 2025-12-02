#!/usr/bin/env python3
"""
rag_faiss.py
Simple local RAG using FAISS + SentenceTransformers.

Works with:
- PDF text extraction
- Optional audio transcription (Whisper + ffmpeg)
- Chunking
- Local embeddings
- FAISS vector index
- Local retrieval (no LLM, no API keys)

Usage examples at bottom.
"""

import os
import requests
import json
import argparse
from typing import List, Dict
from pypdf import PdfReader
import numpy as np
from tqdm import tqdm

# Optional Whisper
try:
    import whisper
except Exception:
    whisper = None

# FAISS is required
try:
    import faiss
except Exception as e:
    raise RuntimeError("FAISS not installed. Please install: pip install faiss-cpu\nError: " + str(e))

from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clean_text(s: str) -> str:
    import re
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[\r\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_pdf_text(paths: List[str]) -> str:
    all_text = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"[WARN] PDF not found: {p}")
            continue

        print(f"[pdf] Reading: {p}")
        try:
            reader = PdfReader(p)
        except Exception as e:
            print(f"[ERR] Failed to read PDF {p}: {e}")
            continue

        pages = []
        for i, pg in enumerate(reader.pages):
            try:
                pages.append(pg.extract_text() or "")
            except Exception as e:
                print(f"[WARN] Failed to extract page {i} of {p}: {e}")
        all_text.append("\n\n".join(pages))

    return clean_text("\n\n".join(all_text))


def transcribe_audio_dir(audio_dir: str, model_name="base", device=None) -> str:
    if not audio_dir or not os.path.isdir(audio_dir):
        return ""

    if whisper is None:
        print("[WARN] Whisper not installed. Skipping audio transcription.")
        return ""

    print(f"[whisper] Loading Whisper model '{model_name}' (device={device})")
    model = whisper.load_model(model_name, device=device)

    transcripts = []
    for fname in os.listdir(audio_dir):
        if fname.startswith("."):
            continue
        path = os.path.join(audio_dir, fname)
        if not os.path.isfile(path):
            continue

        print(f"[whisper] Transcribing: {path}")
        try:
            result = model.transcribe(path, fp16=False)
            transcripts.append(result.get("text", ""))
        except Exception as e:
            print(f"[WARN] Whisper failed for {path}: {e}")

    return clean_text("\n\n".join(transcripts))


def chunk_text(text: str, chunk_size=300, overlap=60) -> List[str]:
    if not text:
        return []

    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ------------------------------------------------------------
# FAISS helpers
# ------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray, index_path: str):
    N, D = embeddings.shape

    # normalize embeddings for cosine similarity
    xb = embeddings.astype("float32")
    norms = np.linalg.norm(xb, axis=1, keepdims=True) + 1e-8
    xb = xb / norms

    index = faiss.IndexFlatIP(D)
    index.add(xb)

    faiss.write_index(index, index_path)
    return index


def save_metadata(metas: List[Dict], meta_path: str):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)


def load_metadata(meta_path: str) -> List[Dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- Ollama helpers (minimal) ----------------

# ---------------- Improved Ollama + extractive helpers ----------------
def _make_ollama_prompt(question: str, contexts: List[str], max_context_chars: int = 800) -> str:
    """
    Build a compact prompt listing contexts as [1], [2], ...
    Truncate each context to keep the beginning and end while removing long middles.
    """
    prompt = (
        "You are an assistant that answers questions using ONLY the provided sources.\n"
        "If the answer is not present in the sources, reply with: \"I don't know\".\n"
        "Be concise and cite sources inline using [n].\n\nSources:\n"
    )
    for i, c in enumerate(contexts, start=1):
        snippet = c or ""
        if len(snippet) > max_context_chars:
            half = max_context_chars // 2
            snippet = snippet[:half].rstrip() + "\n... (truncated) ...\n" + snippet[-half:].lstrip()
        # sanitize newlines a bit
        snippet = snippet.replace("\r", "")
        prompt += f"[{i}] {snippet}\n\n"
    prompt += f"Question: {question}\nAnswer (concise; cite sources like [1], [2]):"
    return prompt

def _gen_ollama_answer(question: str, contexts: List[str], host: str = "http://localhost:11434", model: str = "llama3.1", timeout: int = 300) -> str:
    """
    Call local Ollama with a truncated prompt and generous timeout.
    Returns model text or an error string (starting with [Ollama error: ...]).
    """
    prompt = _make_ollama_prompt(question, contexts, max_context_chars=800)
    url = host.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if "response" in data:
                return data["response"].strip()
            if "output" in data:
                return str(data["output"]).strip()
            # fallback: return stringified JSON (useful for debugging)
            return json.dumps(data)
        return str(data)
    except Exception as e:
        # include server response if available for diagnostics
        msg = f"[Ollama error: {e}]"
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                msg += " Server: " + resp.text[:1000]
            except Exception:
                pass
        return msg


def _split_sentences(text: str):
    import re
    if not text:
        return []
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    # filter out short/bad sentences and those with mainly numbers/headers
    cleaned = []
    for s in sents:
        s = s.strip()
        if len(s) < 25:
            continue
        if len(s.split()) < 5:
            continue
        # heuristic: skip if it's mostly numbers/symbols
        alpha_ratio = sum(c.isalpha() for c in s) / max(1, len(s))
        if alpha_ratio < 0.4:
            continue
        cleaned.append(s)
    return cleaned

def _extractive_synth(question: str, contexts: List[str], embed_model: str, top_n_sentences: int = 3):
    """
    Semantic extractive synthesis:
    - Split contexts into sentences, filter noisy ones
    - Embed sentences and question, score by cosine similarity
    - Pick diverse top sentences and return them with [chunk_id] citations
    """
    embedder = SentenceTransformer(embed_model)
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

    all_sents = []
    for ci, ctx in enumerate(contexts):
        sents = _split_sentences(ctx)
        for s in sents:
            all_sents.append({'chunk_idx': ci, 'text': s})

    if not all_sents:
        return "No useful sentences found in retrieved chunks."

    texts = [x['text'] for x in all_sents]
    sent_embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    scores = (sent_embs @ q_emb).tolist()

    for i, sc in enumerate(scores):
        all_sents[i]['score'] = float(sc)

    all_sents.sort(key=lambda x: x['score'], reverse=True)

    # diversity: max one sentence per chunk first; then allow more
    selected = []
    used = set()
    for s in all_sents:
        if len(selected) >= top_n_sentences:
            break
        if s['chunk_idx'] in used:
            continue
        selected.append(s)
        used.add(s['chunk_idx'])

    if len(selected) < top_n_sentences:
        for s in all_sents:
            if len(selected) >= top_n_sentences:
                break
            if s in selected:
                continue
            selected.append(s)

    answer_parts = [f"{s['text'].rstrip()} [{s['chunk_idx']}]" for s in selected]
    return " ".join(answer_parts)

# ------------------------------------------------------------
# Ingest pipeline
# ------------------------------------------------------------
def ingest(pdfs,
           audio_dir,
           persist_dir,
           chunk_size=300,
           overlap=60,
           embed_model="sentence-transformers/all-MiniLM-L6-v2",
           whisper_model="base",
           whisper_device=None):

    os.makedirs(persist_dir, exist_ok=True)

    # STEP 1 — Load text
    pdf_text = load_pdf_text(pdfs)
    audio_text = transcribe_audio_dir(audio_dir, model_name=whisper_model, device=whisper_device)

    if not pdf_text and not audio_text:
        print("[ERR] No text extracted from PDF or audio.")
        return

    full_text = "\n\n".join([t for t in [pdf_text, audio_text] if t])
    print(f"[info] Total characters: {len(full_text)}")

    # STEP 2 — Chunk
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    print(f"[info] {len(chunks)} chunks created.")

    if not chunks:
        print("[ERR] No chunks created.")
        return

    # STEP 3 — Embeddings
    print(f"[embed] Loading model: {embed_model}")
    model = SentenceTransformer(embed_model)
    embeddings = model.encode(chunks, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # STEP 4 — Build FAISS
    index_path = os.path.join(persist_dir, "faiss.index")
    print(f"[faiss] Building index at {index_path}")
    build_faiss_index(embeddings, index_path)

    # STEP 5 — Save metadata
    metas = [{"id": i, "text": txt} for i, txt in enumerate(chunks)]
    meta_path = os.path.join(persist_dir, "metadata.json")
    save_metadata(metas, meta_path)

    print("[OK] Ingestion complete.")


# ------------------------------------------------------------
# Query pipeline
# ------------------------------------------------------------

def query(persist_dir, question, k=5, embed_model="sentence-transformers/all-MiniLM-L6-v2",
          generator: str = "extractive", ollama_model: str = "llama3.1", ollama_host: str = "http://localhost:11434", synth_sentences: int = 3):
    """
    generator: 'ollama' | 'extractive' | 'chunks'
    """
    index_path = os.path.join(persist_dir, "faiss.index")
    meta_path = os.path.join(persist_dir, "metadata.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print("[ERR] No FAISS index found. Run ingest first.")
        return

    index = faiss.read_index(index_path)
    metas = load_metadata(meta_path)

    model = SentenceTransformer(embed_model)
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)

    D, I = index.search(q_emb, k)
    scores = D[0].tolist()
    indices = I[0].tolist()

    # collect contexts (top-k chunks text)
    contexts = []
    for idx in indices:
        if idx < 0 or idx >= len(metas):
            contexts.append("")
        else:
            contexts.append(metas[idx]['text'])

    # raw chunks mode (old behavior)
    if generator == "chunks":
        print("\n=== TOP RESULTS ===\n")
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0 or idx >= len(metas):
                continue
            text = metas[idx]["text"]
            snippet = text if len(text) <= 800 else text[:800] + " ..."
            print(f"[{rank}] idx={idx} score={score:.4f}\n{snippet}\n")
        return

    # extractive synthesis fallback (no LLM) - keep your previous behavior or improve later
    if generator == "extractive":
        # very small extractive summariser: pick first sentence from each context (simple)
        import re
        sents = []
        for i, c in enumerate(contexts):
            m = re.split(r'(?<=[.!?])\s+', c.strip())
            if m and m[0]:
                sents.append(f"{m[0].strip()} [{indices[i]}]")
        answer = " ".join(sents[:synth_sentences]) if sents else "No extractive answer produced."
        print("\n=== EXTRACTIVE ANSWER ===\n")
        print(answer)
        print("\nSources (chunk ids):", ", ".join([str(i) for i in indices]))
        return

    # ollama mode: build prompt, call local Ollama, fallback to extractive
    if generator == "ollama":
        print("[info] Generating with Ollama (local)...")
        resp = _gen_ollama_answer(question, contexts, host=ollama_host, model=ollama_model)
        if resp.startswith("[Ollama error:"):
            print("[WARN] Ollama failed, falling back to extractive.")
            # reuse extractive fallback above
            import re
            sents = []
            for i, c in enumerate(contexts):
                m = re.split(r'(?<=[.!?])\s+', c.strip())
                if m and m[0]:
                    sents.append(f"{m[0].strip()} [{indices[i]}]")
            # answer = " ".join(sents[:synth_sentences]) if sents else "No extractive answer produced."
            answer = _extractive_synth(question, contexts, embed_model, top_n_sentences=synth_sentences)
            print("\n=== EXTRACTIVE ANSWER ===\n")
            print(answer)
            print("\nSources (chunk ids):", ", ".join([str(i) for i in indices]))
            return
        # success - print Ollama output and chunk ids for traceability
        print("\n=== OLLAMA ANSWER ===\n")
        print(resp)
        print("\nSources referenced (top-k chunk ids):", ", ".join([str(i) for i in indices]))
        return

    print(f"[ERR] Unknown generator '{generator}'. Valid: ollama|extractive|chunks")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Local RAG with FAISS (no API keys)")
    p.add_argument("--pdfs", type=str, default="", help="Comma-separated PDF files")
    p.add_argument("--audio_dir", type=str, default="", help="Folder with audio files")
    p.add_argument("--persist_dir", type=str, required=True, help="Folder to save index + metadata")
    p.add_argument("--ingest", action="store_true", help="Run ingestion pipeline")
    p.add_argument("--query", type=str, default="", help="Ask a question")
    p.add_argument("--k", type=int, default=5, help="Top-K results")
    p.add_argument("--chunk_size", type=int, default=300)
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--whisper_model", type=str, default="base")
    p.add_argument("--whisper_device", type=str, default=None)
    p.add_argument("--generator", type=str, default="extractive", choices=["ollama", "extractive", "chunks"], help="Answer generator: ollama|extractive|chunks")
    p.add_argument("--ollama_model", type=str, default="llama3.1", help="Model name for Ollama (local)")
    p.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="Ollama host URL")
    p.add_argument("--synth_sentences", type=int, default=3, help="Sentences for extractive fallback")

    return p


def main():
    p = build_parser()
    args = p.parse_args()

    pdf_list = [x.strip() for x in args.pdfs.split(",") if x.strip()]

    if args.ingest:
        ingest(pdf_list,
               args.audio_dir,
               args.persist_dir,
               chunk_size=args.chunk_size,
               overlap=args.overlap,
               embed_model=args.embed_model,
               whisper_model=args.whisper_model,
               whisper_device=args.whisper_device)

    if args.query:
        query(args.persist_dir, args.query, k=args.k, embed_model=args.embed_model, generator=args.generator, ollama_model=args.ollama_model, ollama_host=args.ollama_host, synth_sentences=args.synth_sentences)


if __name__ == "__main__":
    main()
