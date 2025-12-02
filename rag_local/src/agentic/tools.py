# src/agentic/tools.py
from typing import Optional, Dict, Any, List, Tuple
import requests

from .log_tools import LogTools
from .secrets import get_n8n_webhook_url
# If you want type hints for ServiceNowTool, you can import it:
# from .servicenow_tool import ServiceNowTool


class Tools:
    """
    Thin wrapper exposing the tools the agent expects:
    - search_local(query, k) -> {'results': [(score, metadata), ...]}
    - parse_log_tool(text) -> dict
    - create_ticket(short, desc, group) -> dict (ServiceNow response or error)
    - compute(expr) -> dict (safe eval)
    - perform_fix(command) -> dict (stub or real)
    - notify_n8n(event_type, incident) -> dict
    """

    def __init__(self, store, embedder, docs: Optional[List[Dict[str, Any]]] = None, servicenow=None):
        self.store = store
        self.embedder = embedder
        self.docs = docs or []
        self.sn = servicenow or None  # ServiceNowTool or None
        self.logtools = LogTools()

        # n8n webhook URL from .env (via secrets.py)
        self.n8n_webhook_url = get_n8n_webhook_url()
        if not self.n8n_webhook_url:
            print("[n8n] N8N_WEBHOOK_URL not set – n8n notifications disabled")

    # ---------------------------
    # Retrieval (RAG)
    # ---------------------------
    def search_local(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Embeds the query and performs a vector search against the FAISS store.
        Returns a dict: {"results": [(score, metadata), ...]}
        If store is None, performs a naive keyword search over docs as a fallback.
        """
        if not query:
            return {"results": []}

        # embedding + vector search when available
        try:
            if self.store is not None and self.embedder is not None:
                q_emb = self.embedder.embed([query])[0]
                hits = self.store.search(q_emb, k=k)
                return {"results": hits}
        except Exception:
            # fall back silently to keyword search
            pass

        # naive keyword search
        lowq = query.lower()
        results: List[Tuple[float, Dict[str, Any]]] = []
        for d in self.docs:
            text = d.get("text", "") or ""
            if lowq in text.lower() or any(tok in text.lower() for tok in lowq.split()):
                tokens = [t for t in lowq.split() if t]
                found = sum(1 for t in tokens if t in text.lower())
                score = found / max(1, len(tokens))
                results.append(
                    (
                        float(score),
                        {
                            "doc_id": d.get("id"),
                            "title": d.get("title"),
                            "chunk": text[:600],
                        },
                    )
                )
        results.sort(key=lambda x: x[0], reverse=True)
        return {"results": results[:k]}

    # ---------------------------
    # Log parsing
    # ---------------------------
    def parse_log_tool(self, text: str) -> Dict[str, Any]:
        return self.logtools.parse_log(text)

    # ---------------------------
    # ServiceNow ticket creation
    # ---------------------------
    def create_ticket(self, short_description: str, description: str, group: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an incident in ServiceNow if ServiceNowTool is configured.
        If not configured, return a simulated response object so the agent can continue.
        """
        if not self.sn:
            simulated = {
                "simulated": True,
                "short_description": short_description,
                "description": description[:1000],
                "assignment_group": group,
                "sys_id": "SIMULATED_SYS_ID",
                "number": "SIM-0001",
                "note": "ServiceNow not configured in environment",
            }
            return simulated

        try:
            group_id = None
            if group:
                group_id = self.sn.find_group_sysid(group)
            res = self.sn.create_incident(short_description, description, assignment_group_sysid=group_id)
            return res
        except Exception as e:
            # log real error so you can see it in the console
            print("[ServiceNow] Error creating incident:", repr(e))
            return {"error": "servicenow_api_error", "message": str(e)}

    # ---------------------------
    # Compute tool (safe evaluator)
    # ---------------------------
    def compute(self, expression: str) -> Dict[str, Any]:
        """
        Very limited safe evaluator for arithmetic. Only allow digits and +-*/(). and spaces.
        """
        allowed = set("0123456789+-*/(). ")
        if any(ch not in allowed for ch in expression):
            return {"error": "disallowed_characters"}
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result}
        except Exception as e:
            return {"error": "eval_error", "message": str(e)}

    # ---------------------------
    # Perform fix (stub or real)
    # ---------------------------
    def perform_fix(self, fix_command: str, run_for_real: bool = False) -> Dict[str, Any]:
        """
        Try to perform a fix. Default: use shell_stub to avoid executing real commands.
        If run_for_real=True, it will attempt to run the command via subprocess (YOU are responsible for safety).
        Returns a dict with {ok: bool, stdout: str, exit_code: int}
        """
        if not run_for_real:
            return {
                "ok": True,
                "stdout": f"(stub) would run: {fix_command}",
                "exit_code": 0,
                "simulated": True,
            }

        import subprocess, shlex, traceback

        try:
            proc = subprocess.run(shlex.split(fix_command), capture_output=True, text=True, timeout=60)
            return {
                "ok": proc.returncode == 0,
                "stdout": proc.stdout + proc.stderr,
                "exit_code": proc.returncode,
            }
        except Exception as e:
            return {
                "ok": False,
                "stdout": "",
                "exit_code": -1,
                "error": str(e),
                "trace": traceback.format_exc(),
            }

    # ---------------------------
    # n8n notification helper
    # ---------------------------
    def notify_n8n(self, event_type: str, incident: dict) -> dict:
        """
        Send a small JSON payload to n8n when something happens
        (e.g. 'incident_not_fixed').
        """
        if not self.n8n_webhook_url:
            return {"skipped": True, "reason": "N8N_WEBHOOK_URL not set"}

        payload = {
            "event": event_type,
            "incident_number": incident.get("number"),
            "sys_id": incident.get("sys_id"),
            "short_description": incident.get("short_description") or incident.get("short", ""),
            "state": incident.get("state") or incident.get("incident_state"),
        }

        try:
            print(f"[n8n] POST -> {self.n8n_webhook_url} | {payload}")
            r = requests.post(self.n8n_webhook_url, json=payload, timeout=5)
            return {"ok": True, "status": r.status_code}
        except Exception as e:
            print(f"[n8n] ERROR posting to webhook: {e}")
            return {"ok": False, "error": str(e)}
