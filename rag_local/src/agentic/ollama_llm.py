# src/agentic/ollama_llm.py
import os
import requests
from typing import Optional


class OllamaLLM:
    """
    Minimal wrapper around a local Ollama model.

    Default model: 'llama3'.
    We only use this to rewrite the draft answer produced by the agent.
    """

    def __init__(self, model: str = "llama3", base_url: Optional[str] = None):
        self.model = model
        # Default Ollama endpoint
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def rewrite_answer(self, user_query: str, draft_answer: str) -> str:
        """
        Ask Ollama to improve clarity/wording of the draft answer.
        If anything fails, we return draft_answer unchanged.
        """
        try:
            prompt = (
                "You are a helpful Level 1 support engineer.\n"
                "You will be given:\n"
                "1) The user's issue description.\n"
                "2) A draft answer created by a rules-based system.\n\n"
                "Your task:\n"
                "- Keep ALL factual details and ticket numbers.\n"
                "- Keep the structure similar (sections, bullet points).\n"
                "- Improve clarity, tone, and grammar.\n"
                "- Do NOT invent new steps or systems.\n\n"
                f"User issue:\n{user_query}\n\n"
                f"Draft answer:\n{draft_answer}\n\n"
                "Rewrite the draft answer in a clear, concise, professional support style:\n"
            )

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response") or draft_answer
            return text.strip() or draft_answer
        except Exception:
            return draft_answer
