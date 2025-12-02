# src/agentic/core.py
from .rag import Embedder, build_vector_store
from .tools import Tools
from .ollama_llm import OllamaLLM
from .core_agent import SimpleAgent
from .servicenow_tool import ServiceNowTool
from .audit import audit_log
from pathlib import Path
import os

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

# -------------------------------------------------------------
# Interactive CLI demo
# -------------------------------------------------------------
def run_support_demo_interactive():
    """
    Interactive CLI loop: Ask user for a ticket/log text or a path to a file.
    The agent will:
        - parse logs
        - retrieve KB
        - create ticket
        - attempt auto-fix (stub)
        - update or escalate ticket
    """
    from .rag import Embedder, build_vector_store
    from .servicenow_tool import ServiceNowTool
    from .tools import Tools
    from .core_agent import SimpleAgent
    import pathlib
    import os

    # Load docs from project docs folder
    docs_dir = pathlib.Path(__file__).resolve().parents[2] / "docs"

    docs = []
    for p in docs_dir.glob("*.md"):
        docs.append({"id": p.stem, "title": p.stem, "text": p.read_text(encoding='utf-8')})

    # Build vectorstore
    embedder = Embedder()
    store, _ = build_vector_store(docs, embedder)

    # Initialize ServiceNow (or fallback to simulated)
    try:
        sn = ServiceNowTool()
    except Exception:
        sn = None
        print("⚠ ServiceNow not configured — using simulated tickets.")

    tools = Tools(store, embedder, docs, servicenow=sn)              
    llm = OllamaLLM(model="llama3")  # or whatever model name you use in Ollama
    agent = SimpleAgent(tools, embedder, llm=llm)



    print("\n=== Interactive Support Agent ===")
    print("Enter:")
    print("- raw text describing an issue")
    print("- a path to a log file")
    print("- 'quit' to exit")
    print("--------------------------------\n")

    while True:
        user_input = input("Provide issue/log path > ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        # If it's a file path, try reading it
        if os.path.exists(user_input):
            try:
                with open(user_input, "r", encoding="utf-8", errors="ignore") as f:
                    ticket_text = f.read()
            except Exception as e:
                print("Could not read file:", e)
                continue
        else:
            ticket_text = user_input

        print("\n--- Running agent ---\n")
        out = agent.reason_and_act(ticket_text)

        print("=== FINAL ANSWER ===")
        print(out.get("answer", "No answer returned"))

        # If agent created a ticket, show summary
        hist = out.get("history", [])
        if hist:
            ref = hist[-1].get("reflection", {})
            print("\nReflection:", ref)

        print("\n--------------------------------\n")
