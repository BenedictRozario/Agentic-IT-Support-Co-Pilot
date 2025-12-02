# src/agentic/audit.py
"""
Simple audit logging and redaction utilities for the agent.
Write only non-sensitive metadata to the audit log (JSON lines).
"""

from typing import Any, Dict
import json
import datetime
import os

AUDIT_PATH = os.getenv("AGENT_AUDIT_PATH", "agent_audit.log")

def redact(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact likely sensitive keys and large raw payloads.
    Returns a shallow-copied dict with redactions applied.
    """
    e = dict(entry)
    for k in list(e.keys()):
        lk = k.lower()
        if any(s in lk for s in ("pass", "secret", "token", "auth", "credential")):
            e[k] = "<REDACTED>"
        # mask values that look like entire request/response bodies
        if isinstance(e[k], str) and len(e[k]) > 1000:
            e[k] = "<TRUNCATED>"
    return e

def audit_log(entry: Dict[str, Any], path: str = None) -> None:
    """
    Append a single audit record (JSON line) to the audit file.
    Make sure not to pass secrets into this function.
    """
    if path is None:
        path = AUDIT_PATH
    safe = redact(entry)
    safe["ts"] = datetime.datetime.utcnow().isoformat() + "Z"
    # write JSON line (append)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")
