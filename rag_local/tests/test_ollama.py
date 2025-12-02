# test_ollama.py — run inside your .venv (no PowerShell needed)
import requests, json, sys

URL = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.1",
    "prompt": "Say hello",
    "stream": False
}

try:
    r = requests.post(URL, json=payload, timeout=60)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("ERROR:", e)
    if hasattr(e, "response") and e.response is not None:
        try:
            print("Server returned:", e.response.text)
        except Exception:
            pass
    sys.exit(1)
