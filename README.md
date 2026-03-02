# ClearCheck

**Deepfake & Misinformation Shield for Older Adults**

ClearCheck is a verification tool that helps older adults (55+) check whether online content is trustworthy. Paste any claim into the N8N chat, and get a plain-language verdict backed by multiple independent sources.

## How It Works

```
N8N Chat UI → Chat Trigger → HTTP Request → FastAPI Server
                                                  │
                                     ┌────────────┼────────────┐
                                     ▼            ▼            ▼
                                 Pinecone      Tavily     Google Fact
                                (known misinfo) (web search) Check API
                                     └────────────┼────────────┘
                                                  ▼
                                     LangGraph Agent (Claude)
                                       ┌─────────┴─────────┐
                                       ▼                    ▼
                                 Claude Analysis     LLM Validation
                                     └─────────┬─────────┘
                                                ▼
                                         SQLite Logging
                                                ▼
                                   Formatted response → N8N Chat
```

1. **Pinecone** searches a curated knowledge base of 20 known misinformation patterns
2. **Tavily** runs a real-time web search for current information
3. **Google Fact Check API** queries published fact-checks from trusted organizations
4. **LangGraph + Claude** analyzes all evidence with multi-step reasoning and returns a validated structured verdict

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Claude (Anthropic) |
| Agent Orchestration | LangGraph |
| Workflow & Chat UI | N8N (built-in chat trigger) |
| Vector Store | Pinecone |
| Embeddings | OpenAI text-embedding-3-small |
| Web Search | Tavily |
| Fact-Checking | Google Fact Check API |
| API Server | FastAPI |
| Database | SQLite |

## Setup

### Prerequisites

- Python 3.11+
- N8N instance (cloud or self-hosted)
- API keys for: Anthropic, OpenAI, Pinecone, Tavily, Google Fact Check

### Installation

```bash
git clone https://github.com/Javirum/clearcheck.git
cd clearcheck
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Required variables:

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
TAVILY_API_KEY=
GOOGLE_FACTCHECK_API_KEY=
```

### Seed the Knowledge Base

```bash
python -m src.seed_knowledge_base
```

### Start the API Server

```bash
python app.py
```

The server runs on `http://localhost:8000`. Verify with `GET /health`.

### Import the N8N Workflow

1. Open your N8N instance
2. Go to **Workflows → Import from File**
3. Select `n8n_workflow.json`
4. Update the HTTP Request node URL if your server is not on `localhost:8000`
5. Activate the workflow
6. Open the chat UI from the Chat Trigger node

### Run Evaluation

```bash
python evaluate.py
```

## Project Structure

```
clearcheck/
├── app.py                          # FastAPI server (called by N8N)
├── evaluate.py                     # Test dataset evaluation script
├── n8n_workflow.json               # N8N workflow (import into N8N)
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py                   # Environment variables & constants
│   ├── schemas.py                  # Pydantic models & LangGraph state
│   ├── evidence.py                 # Pinecone, Tavily, Google Fact Check
│   ├── agent.py                    # LangGraph agent (Claude + validation)
│   ├── audit_log.py                # SQLite audit logging
│   └── seed_knowledge_base.py      # Seed Pinecone with misinfo patterns
├── data/
│   ├── misinformation_patterns.json  # 20 curated misinfo patterns
│   └── test_dataset.json             # 20-item test set with ground truth
├── .env
├── .gitignore
├── PROJECT_PLAN.md
└── README.md
```

## License

MIT
