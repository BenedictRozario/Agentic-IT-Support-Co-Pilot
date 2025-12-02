# src/agentic/secrets.py
import os
from dotenv import load_dotenv
from pathlib import Path


# --------------------------------------------------------------------
# Load .env from project root (rag_local/.env)
# --------------------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(f"[secrets] WARNING: .env not found at {ENV_PATH}")


def get_servicenow_creds():
    """
    Returns (instance, username, password) for ServiceNow.

    Uses:
    - SN_INSTANCE
    - SN_USER
    - SN_PASS
    """
    env = os.environ

    # Instance / URL (default to empty string so .startswith is safe)
    instance = env.get("SN_INSTANCE", "")

    # If full URL, strip https:// and trailing slash
    if instance.startswith("https://"):
        instance = instance[len("https://"):]
    if instance.endswith("/"):
        instance = instance[:-1]

    # Username
    user = env.get("SN_USER", "")

    # Password
    pw = env.get("SN_PASS", "")

    if not instance or not user or not pw:
        raise RuntimeError(
            "ServiceNow credentials not set in .env "
            "(expected SN_INSTANCE/SN_USER/SN_PASS)."
        )

    return instance, user, pw


def get_n8n_webhook_url() -> str:
    """
    Read N8N_WEBHOOK_URL from .env.
    Returns empty string if not set.
    """
    return os.getenv("N8N_WEBHOOK_URL", "").strip()
