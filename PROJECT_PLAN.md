# ClearCheck — Deepfake & Misinformation Shield for Older Adults

## 1. Use Case

**Selected option:** Option E — Own Idea

### Problem Statement

Older adults (55+) are the most vulnerable demographic to AI-generated deepfakes and online misinformation. They share fake news at significantly higher rates than younger users, and existing solutions — platform fact-check labels, media literacy campaigns — either go unnoticed or feel patronizing. There is no simple tool that meets them where they are, verifies content on demand, and builds their critical thinking over time.

### Target Users

- **Primary:** Adults 55+ active on Facebook, WhatsApp, YouTube, and email who regularly encounter unverified content.
- **Secondary:** Family members who want to provide a helpful tool instead of constantly correcting their relatives.

### Success Criteria

- ≥85% accuracy identifying misinformation against a curated test set
- Plain-language explanations at grade 6–8 readability level
- Cross-references ≥2 independent sources per check
- Educational tip included with every verdict
- Full audit log of every check with reasoning chain

### Current Process

No effective process exists. Older adults either trust and share content without verification, ask family members who respond with frustration, ignore platform fact-check labels they distrust, or Google the claim and find more misinformation confirming the original fake.

---

## 2. Technology Stack

### Technology Requirements

| Question | Answer | Implication |
|----------|--------|-------------|
| Need external knowledge? | Yes | RAG with Pinecone for known misinfo patterns + real-time web search |
| Need external systems? | Yes | Web search (Tavily), fact-check API (Google), database (SQLite) |
| Need multi-step reasoning? | Yes, linear | N8N workflow for API orchestration + LangGraph for stateful AI agent reasoning |
| Need business system integration? | No | Standalone application |
| Need to run autonomously? | Partially | On-demand (user submits content), with light proactive education tips |

### Selected Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Core LLM | Claude (Anthropic) | Safety-oriented design, nuanced reasoning, clear explanations for non-technical users |
| Workflow Orchestration | N8N | Visual workflow, chat trigger, native API integrations, parallel execution, retry/error handling built-in |
| Agent Orchestration | LangGraph | Stateful graph-based AI agent logic, multi-step reasoning, state management, structured output |
| Vector Store | Pinecone | Cloud-hosted (demo-ready), managed, free tier sufficient |
| Embeddings | OpenAI text-embedding-3-small | Cost-effective, high quality |
| Web Search | Tavily API | Agent-ready structured results, free tier, built for LLM workflows |
| Fact-Checking | Google Fact Check API | Free, searches published fact-checks from trusted organizations |
| Output Validation | LLM-based validation | Secondary LLM call validates verdicts for consistency, factual grounding, and structural correctness |
| Interface | N8N Chat (built-in) | Native chatbot UI within N8N; no separate frontend to build or deploy |
| Database | SQLite | Lightweight audit logging, no infrastructure, portable |

### Alternatives Considered

- **LangChain alone:** LangGraph provides the graph-based state management that LangChain lacks for multi-step agent workflows
- **Streamlit:** Separate frontend to build and deploy; N8N's built-in chat eliminates the need for a custom UI
- **Pydantic:** Static schema validation; LLM validation provides richer semantic checks for consistency and factual grounding
- **Chroma:** Local-only limits demo portability vs. Pinecone's cloud hosting
- **GPT-4o:** Claude's safety design better suits a trust-and-safety tool for vulnerable users
- **SerpAPI:** Requires custom parsing vs. Tavily's agent-ready output

---

## 3. MVP Scope

### Included (Must-Have)

- Text input via N8N's built-in chat interface
- N8N workflow orchestrating the full verification pipeline, with LangGraph powering the AI agent reasoning
- Three-layer verification: Pinecone knowledge base + Tavily web search + Google Fact Check API
- Color-coded trust verdict (green / yellow / red)
- Plain-language explanation at grade 6–8 reading level
- Confidence score with explicit uncertainty handling when evidence is insufficient
- Source citations with clickable links
- Educational tip tailored to the content type
- Full audit logging in SQLite
- LLM-validated structured output (secondary LLM call checks verdict quality)

### Excluded from MVP

| Feature | Version | Why Excluded |
|---------|---------|-------------|
| URL input with content extraction | v2 | Adds complexity around paywalls, dynamic content, broken links |
| Check history in the interface | v2 | Nice UX but doesn't change core verification logic |
| Pattern detection across checks | v2 | Requires session persistence and analytics logic |
| User feedback mechanism | v2 | Valuable feedback loop but not needed for core function |
| Image / video / audio analysis | v3+ | Requires specialized ML models |
| WhatsApp / Messenger bot | v3+ | High-value channel but requires platform API setup |
| Browser extension | v3+ | Significant separate development effort |
| Family dashboard | v3+ | Adds multi-user auth and privacy complexity |
| Multi-language support | v3+ | Requires translation APIs or multilingual embeddings |

### What to Cut if Time Gets Tight

Prioritized from "cut first" to "cut last":

| Feature | Priority | Cut Rationale |
|---------|----------|---------------|
| **Educational tip per verdict** | Cut first | Nice touch but purely cosmetic. Can be a single hardcoded line instead of dynamic generation |
| **SQLite audit logging** | Cut second | Valuable for production but not needed for demo. N8N execution log + LangGraph trace serve as a de facto audit trail |
| **Google Fact Check API** (3rd source) | Cut third | Tavily + Pinecone already give 2 sources. Dropping removes one integration. Cross-reference metric drops from 3 to 2, still defensible |
| **LLM validation step** | Cut fourth | Useful guardrail but adds latency and an extra API call; can validate manually during demo |
| **Confidence score with uncertainty handling** | Keep | Core differentiator — showing the agent knows when it doesn't know |
| **Color-coded verdict + plain-language explanation** | Keep | This IS the product |
| **Pinecone RAG + Tavily search** | Keep | Without these the agent is just a prompt |

**Absolute minimum viable demo:** N8N chat → Pinecone + Tavily (parallel) → LangGraph agent (Claude verdict) → display result in chat.

---

## 4. Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Verdict accuracy | ≥85% on 20 items | Agent vs. ground truth on test set |
| Explanation readability | Grade 6–8 | Flesch-Kincaid scoring on all explanations |
| Source citation rate | 100% | LLM validation enforces ≥1 source per verdict |
| Multi-source checks | ≥80% | Audit log: checks consulting ≥2 sources |
| Uncertainty handling | 0% false confidence | 3 items with no debunking → must return uncertain |
| Response time | <20s | Timed across test set |
| Output validity | 100% | All responses pass LLM validation |

---

## 5. Risk Assessment

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|-----------|
| LLM hallucination: agent fabricates sources or gives confident verdicts without evidence | High | High | Separate retrieval from reasoning; LLM validation step checks for real source URLs; explicit uncertain verdict when evidence is insufficient; validate cited URLs exist in search results |
| API cost escalation during testing | Med | High | Cache duplicate queries in SQLite; set daily spend limits; optimize prompts. Estimated total budget: $10–26 |
| API rate limits block agent during demo or testing | Med | Med | Exponential backoff and retry; cache results aggressively; graceful degradation if one API is rate-limited |
| External API failure or interface change | Low | High | Graceful degradation: if one source fails, proceed with remaining and note the gap; pin library versions; try/except on all API calls |
| Response time exceeds 20 seconds | Med | Med | Run knowledge base and web search in parallel via N8N; show progress indicator; cache common queries; 10-second timeout per API call |
| User adoption: target demographic doesn't trust the tool | High | High | Radically simple chat interface; frame as helper not authority; always show verifiable sources; build agency through educational tips |
| Scope creep | High | Med | MVP scope documented and fixed; if it's not in the day-by-day plan, it doesn't exist until after Day 5 |
| Knowledge base contains outdated or inaccurate patterns | Med | High | Source entries from reputable fact-checkers only; tag entries with dates; web search acts as real-time correction layer |
| Privacy: users paste content containing personal information | Med | High | Minimize data retention; SQLite stored locally; no user accounts in MVP; clear disclaimer about logging |
| N8N learning curve | Med | Med | Watch one N8N webhook+API tutorial (1hr) before Day 1; the workflow is only 6–7 nodes |
| LangGraph learning curve | Med | Med | Review LangGraph quickstart and examples (1hr) before Day 1; the graph is only 4–5 nodes |
| Day 1 burns on setup/config issues | Med | High | Use N8N Cloud instead of self-hosting to eliminate Docker issues; have all API keys ready before Day 1 |

---

## 6. Implementation Plan (5 Days)

### Day 1: Setup & Data Prep

- Set up Python project, install LangGraph and dependencies (`langgraph`, `langchain-anthropic`, `langchain-openai`, `langchain-pinecone`)
- Install N8N (self-hosted via Docker or N8N Cloud)
- Create all API accounts (Anthropic, Pinecone, Tavily, Google Fact Check, OpenAI for embeddings)
- Curate knowledge base: 20 known misinformation patterns from Snopes, PolitiFact, AFP
- Create embeddings, upload to Pinecone
- Build test dataset: 20 items with ground truth (10 misinfo, 7 legit, 3 uncertain)
- Define output schema and LangGraph state schema

**Milestone:** Pinecone index populated, N8N accessible, LangGraph scaffolded, all API keys working

### Days 2–3: Core N8N + LangGraph Workflow

- Build N8N workflow (outer orchestration):
  - Chat trigger (receives content from N8N's built-in chat UI)
  - Parallel branch: Pinecone query + Tavily search + Google Fact Check
  - Merge results → call LangGraph agent (via Code node or HTTP request)
  - SQLite logging (via Code node)
  - Respond to chat
- Build LangGraph agent (inner AI reasoning):
  - Input node (receives merged evidence from N8N)
  - Claude analysis node (analyzes all evidence, applies multi-step reasoning)
  - LLM validation node (secondary Claude call checks verdict for consistency, grounding, and completeness)
  - Return structured verdict
- Test end-to-end: N8N workflow → LangGraph agent → validated response
- Tune Claude prompt for plain-language output and uncertainty handling

**Milestone:** Chat accepts text, N8N gathers evidence, LangGraph returns validated verdict with explanation, sources, and tip

### Day 4: Chat UI + Testing

- Configure N8N chat interface: customize welcome message, styling, and response formatting
- Run full test dataset through the pipeline
- Measure accuracy, readability (Flesch-Kincaid), response times
- Fix failures, tune prompts, re-test

**Milestone:** Working app, all metrics measured

### Day 5: Polish & Docs

- Deploy (N8N Cloud or prepare local demo)
- Write README with setup instructions and architecture diagram
- Compile evaluation results table
- Prepare demo script: 3 representative checks (obvious fake, legit content, uncertain)
- Code cleanup, remove debug output

**Milestone:** Demo-ready with evaluation report

### Timeline Summary

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Setup & Data Prep | Day 1 | APIs configured, Pinecone seeded, N8N running, LangGraph scaffolded, test data ready |
| Core N8N + LangGraph Workflow | Days 2–3 | N8N orchestrates evidence gathering, LangGraph agent returns structured verdicts |
| Chat UI + Testing | Day 4 | N8N chat interface configured, full pipeline tested, metrics measured |
| Polish & Docs | Day 5 | Demo-ready with evaluation report |

---

## 7. Resources & Budget

**Team:** Solo developer — responsible for all phases.

**Development tools:** Python 3.11+, N8N, LangGraph, LangChain, SQLite, Git/GitHub.

| Service | Cost | Notes |
|---------|------|-------|
| Anthropic API (Claude) | $10–25 | Primary cost — development, testing, demo |
| OpenAI Embeddings | <$1 | One-time: embedding ~20 knowledge base entries |
| Pinecone / Tavily / Google Fact Check / N8N | $0 | All free tiers sufficient for MVP |
| LangGraph / LangChain | $0 | Open-source Python libraries |
| **Total** | **$10–26** | |

---

## Workflow Architecture

```
[N8N Chat UI] → [Chat Trigger]
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
       [Pinecone] [Tavily] [Google FC]
            └─────────┼─────────┘
                      ▼
                [Merge Results]
                      ▼
          [LangGraph Agent (via Code node)]
            ┌─────────┴─────────┐
            ▼                   ▼
      [Claude Analysis]   [LLM Validation]
       (structured JSON)
            └─────────┬─────────┘
                      ▼
               [SQLite Logging]
                      ▼
            [Respond → N8N Chat UI]
```

**Division of responsibilities:**
- **N8N** handles: chat interface, chat trigger, parallel API calls (Pinecone, Tavily, Google FC), result merging, SQLite logging, response delivery
- **LangGraph** handles: multi-step AI reasoning over merged evidence, structured output generation, LLM-based validation
