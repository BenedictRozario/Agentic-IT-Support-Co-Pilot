# Agentic-IT-Support-Co-Pilot
Automated Incident Diagnostics + Tiered Remediation + ServiceNow + n8n Integration

**Overview**
This project implements a Level-1 AI Support Agent powered by:

Capability						                      Technology
Retrieval-Augmented Generation				      Vector search (FAISS) + MiniLM embeddings
Autonomous reasoning \& planning			      Custom Python agent
Log inspection \& error extraction		      Regex-based diagnostic tool
Ticketing automation					              ServiceNow API integration
L2 Automated Fix / Incident Resolution			n8n Cloud Webhook workflows
Confidence \& Self-reflection				        Retrieval score interpretation

The system can:
✔ Understand user problem statements or log files
✔ Retrieve relevant troubleshooting steps from a local KB
✔ Decide whether a fix is possible
✔ Automatically create, update, and resolve SN incidents
✔ Trigger n8n automation for deeper remediation
✔ Self-reflect → ask for more details if uncertain

**Architecture**

User Issue / Log
       │
       ▼
Python L1 Support Agent

 ├─ RAG Search (FAISS)
 ├─ Log Parser (error codes + paths)
 ├─ Reasoning + Reflection
 │   ├─ Confidence calculation
 │   ├─ Clarifying questions if needed
 │   └─ Plan tool actions
 ├─ Tools Layer
 │   ├─ ServiceNow (create/update/close)
 │   ├─ Scripted Fix Stub (perform\_fix)
 │   └─ Notify n8n Webhook
 │
 ▼
ServiceNow Incident  ◄─────►  n8n Cloud (L2 Auto-Fix)

**Project Structure**

rag\_local/
│
├── src/
│   ├── agentic/
│   │   ├── core.py       ← main agent logic (reasoning)
│   │   ├── core\_agent.py ← planning, execution, reflection
│   │   ├── rag.py        ← embeddings + vector DB
│   │   ├── tools.py      ← SN + log parser + n8n integration
│   │   ├── servicenow\_tool.py
│   │   ├── log\_tools.py
│   │   └── audit.py      ← structured debug output
│   │
│   ├── demo.py          ← interactive agent (CLI)
│   ├── evaluate\_agent.py ← accuracy + confidence metrics
│
├── data/                ← knowledge base docs for retrieval
├── vectordb/            ← FAISS index storage
├── .env                 ← credentials + instance URLs
└── README.md

**Technology Stack**

Layer				          Tool  
Runtime				        Python 3.10+
LLM / Embeddings	    Sentence-BERT MiniLM
Vector Search			    FAISS
ITSM				          ServiceNow REST API  
Workflow Automation		n8n Cloud
Config				        dotenv
Logging				        JSON-structured audit pipeline

**Agentic Reasoning Features**

Capability			          Example
Multi-step tool planning	Parse log → find KB → create ticket → attempt fix
Self-reflection			      “I’m not confident — provide more log lines”
Follow-ups			          “not fixed INC001000” → escalate \& update
Auto-resolution			      “fixed INC001000” → close SN ticket
Low hallucination		      Fixes only run when command found in KB

**Knowledge Base (RAG Sources)**

Each KB document stores:
* Issue Symptoms
* Log Error Codes (e.g., ERR\_CONFIG\_MISSING)
* Troubleshooting Steps
* Safe Auto-Fix Commands

**ServiceNow Integration**

Functions provided by Tools Layer:

Action				          Implementation
Create Incident			    create\_incident()
Add Comments			      add\_comment()
Update Fields			      update\_incident() (state, close code, notes)
Resolve Automatically		When agent detects confirmation

**n8n Automation Workflows (L2)**

Workflows triggered on:

Event				          Action
incident\_escalated		Auto-remediation + SNOW update
auto\_resolve			    Final closure steps

**How to Run**

1. Install dependencies - pip install -r requirements.txt 
2. Add .env:
SERVICENOW\_INSTANCE=https://xxxxx.service-now.com
SERVICENOW\_USER=APIUser
SERVICENOW\_PASSWORD=**********
N8N\_WEBHOOK\_URL=https://... (from n8n cloud)
3. Run interactive agent - python src/demo.py



