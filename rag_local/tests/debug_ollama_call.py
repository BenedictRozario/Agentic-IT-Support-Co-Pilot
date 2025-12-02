# debug_ollama_call.py
import json, requests, sys, os

PERSIST_DIR = "./vectordb"
MODEL = "llama3.1"
HOST = "http://localhost:11434"
K = 5   # change to 3 if you want smaller prompt

def load_metadata(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(question, contexts, max_context_chars=1000):
    prompt = ("You are an assistant that answers questions using only the provided sources.\n"
              "If the answer cannot be found in the sources, say 'I don't know'. Be concise and cite sources using [n].\n\n"
              "Sources:\n")
    for i, c in enumerate(contexts, start=1):
        snippet = c
        if len(snippet) > max_context_chars:
            half = max_context_chars // 2
            snippet = snippet[:half] + "\n... (truncated) ...\n" + snippet[-half:]
        prompt += f"[{i}] {snippet}\n\n"
    prompt += f"Question: {QUESTION}\nAnswer (concise, cite sources like [1], [2]):"
    return prompt

if __name__ == "__main__":
    meta_path = os.path.join(PERSIST_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        print("No metadata.json at", meta_path); sys.exit(1)

    metas = load_metadata(meta_path)
    QUESTION = "What are the production Do's for RAG?"
    # get first K chunk texts (these are the ones your script fetches by index)
    contexts = [m["text"] for m in metas[:K]]

    prompt = build_prompt(QUESTION, contexts, max_context_chars=1000)
    payload = {"model": MODEL, "prompt": prompt, "stream": False}

    url = HOST.rstrip("/") + "/api/generate"
    print("=== PROMPT LENGTH (chars):", len(prompt))
    try:
        r = requests.post(url, json=payload, timeout=300)
        print("HTTP status:", r.status_code)
        try:
            j = r.json()
            print(json.dumps(j, indent=2, ensure_ascii=False))
        except Exception as e:
            print("Failed to decode JSON response:", e)
            print("Raw text:", r.text[:2000])
    except Exception as e:
        print("Request exception:", repr(e))
        # If server returned a response object on the exception, print it
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                print("Server response:", resp.text[:2000])
            except:
                pass
        sys.exit(1)
