# Enterprise Multi-Domain RAG Chatbot Service

---

## 1. Project Overview

### a. Project Name
Enterprise Multi-Domain RAG Chatbot Service

### b. Duration
2025

### c. Project Type
Individual Project

### d. Role
- Full system design and architecture
- LangGraph-based AI pipeline implementation (router / retriever / generator, 3-node)
- LLM-based domain router and keyword fallback logic development
- Multi-domain Chroma vector DB setup and RAG pipeline construction
- Streamlit frontend UI development (domain badges, source references, image upload)
- Docker Compose containerization setup
- Domain-specific knowledge documents (10 files) and test scenario authoring

### e. Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| AI Orchestration | LangGraph |
| RAG Framework | LangChain |
| LLM | OpenAI gpt-4o-mini |
| Embedding | OpenAI text-embedding-3-small |
| Vector DB | Chroma |
| Backend | FastAPI |
| Frontend | Streamlit |
| Infrastructure | Docker, Docker Compose |

### f. One-Line Summary
An enterprise AI chatbot that automatically classifies user queries into 5 business domains and retrieves answers from domain-specific internal knowledge documents via RAG.

---

## 2. Problem Definition & Approach

### a. Problem to Solve
Employees across various roles — field workers, office staff, sales representatives, and new hires — repeatedly had to contact the responsible department or person for every work-related question. Key pain points included:

- **Field workers**: Had to manually search through thick manuals when equipment alarms or error codes appeared
- **Office staff**: Repeatedly contacted HR for the same leave, travel expense, and remote work policy questions
- **IT issues**: Employees waited for IT helpdesk responses during VPN, SAP, or email failures
- **Sales representatives**: Could not quickly access product specs or competitor comparisons during client meetings
- **New hires**: Unclear channels to access onboarding information such as process flows and quality standards

### b. Limitations of Existing Approaches

| Existing Approach | Limitation |
|-------------------|------------|
| Directly contacting the responsible department | Delayed responses, repetitive manual work for staff |
| Manually searching large internal documents | Time-consuming; hard to locate the exact information |
| Unclear routing — employees unsure whom to ask | Wrong department contacted, followed by re-routing |
| No way for field workers to share equipment photos for remote support | Inadequate remote troubleshooting |
| Slow propagation of updated knowledge documents | Outdated information circulated |

### c. Solution & Core Idea
- **LLM-based automatic domain routing**: Without requiring users to select a domain, the LLM interprets query intent and automatically routes to one of 5 domain-specific RAG pipelines
- **Domain-isolated RAG pipelines**: Knowledge documents for each business domain are managed in separate Chroma collections, ensuring no cross-domain noise in retrieval
- **LangGraph state graph**: Explicit state management (domain, context, sources) across a router → retriever → generator 3-node pipeline
- **Multimodal support**: Field photos are Base64-encoded and analyzed alongside text queries by the LLM

---

## 3. Implementation

### a. System Architecture

```
User Query (text + optional image)
        │
        ▼
  [Streamlit Frontend]
  - Domain badge display
  - Source document references
  - Image upload (Base64 encoding)
        │ POST /chat
        ▼
  [FastAPI Backend]
        │
        ▼
  [LangGraph Pipeline]
  ┌─────────────────────────────┐
  │ router_node                 │
  │  LLM-based domain classification │
  │  + keyword fallback         │
  └────────────┬────────────────┘
               │ domain decided
               ▼
  ┌─────────────────────────────┐
  │ retriever_node              │
  │  Chroma vector DB search    │
  │  Top-K docs from domain collection │
  └────────────┬────────────────┘
               │ context + sources
               ▼
  ┌─────────────────────────────┐
  │ generator_node              │
  │  Domain-specific System Prompt │
  │  LLM answer generation      │
  └────────────┬────────────────┘
               │
               ▼
  {reply, domain, sources} → Frontend response
```

### b. Data Pipeline

```
Knowledge documents (.txt)
    │
    ▼
TextLoader (LangChain)
    │
    ▼
RecursiveCharacterTextSplitter
(chunk_size=500, overlap=80)
    │
    ▼
OpenAI Embedding
(text-embedding-3-small)
    │
    ▼
Chroma Vector DB
(independent collection per domain)
    │
    ▼
Similarity search → Top-K documents returned
```

**Domain Knowledge Document Structure (10 files total)**

| Domain | File | Content |
|--------|------|---------|
| manual | equipment_manual.txt | PA-100 specs, alarm codes, filter replacement schedule |
| manual | safety_manual.txt | LOTO procedures, safety rules, PPE standards |
| manual | troubleshooting.txt | Error codes (E-201~E-205) and remediation steps |
| hr | hr_policy.txt | Work hours, benefits, HR policy |
| hr | expense_policy.txt | Domestic/overseas travel expense standards, reimbursement process |
| hr | vacation_policy.txt | Leave accrual, carryover, allowance, remote work policy |
| it | vpn_troubleshooting.txt | VPN error types and resolution steps |
| it | email_issue.txt | Email storage management, SAP login issues |
| sales | product_info.txt | Product A/B specs, pricing, features |
| sales | competitor_analysis.txt | Competitive strengths/weaknesses, response strategies |
| education | training_manual.txt | Onboarding procedures, internal system usage |
| education | process_guide.txt | Production process flow, quality inspection standards |

### c. Key Features

1. **Automatic domain routing**: LLM classifies user queries into one of 5 domains (Field Manual / HR / IT / Sales / Education) without requiring manual selection
2. **Domain-specific RAG answers**: Retrieves relevant document chunks from the target domain's knowledge base to generate accurate answers
3. **Source transparency**: Every response includes the filenames of the knowledge documents used as evidence
4. **Multimodal support**: Field photos can be uploaded and analyzed alongside text by the LLM
5. **Conversation context**: Thread ID-based session management enables multi-turn conversations
6. **One-command deployment**: Docker Compose starts the entire service (backend, frontend, vector DB) with a single command

### d. Core Algorithm / Logic

**LLM-based Domain Router (dual safety net)**

```python
def classify_domain(query: str) -> str:
    # Primary: System Prompt forces LLM to output only one domain key
    response = _router_llm.invoke([("system", _ROUTER_SYSTEM), ("human", query)])
    domain = response.content.strip().lower()

    # Fallback: If LLM output is invalid, use Korean keyword matching
    if domain not in valid_domains:
        domain = _keyword_fallback(query)
    return domain
```

**LangGraph State Graph (ChatState)**

```python
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # Conversation history
    domain: str        # Classified domain key
    context: str       # Retrieved document text
    sources: List[str] # Source document filenames
```

### e. Data Processing

- **Chunking**: RecursiveCharacterTextSplitter (chunk_size=500, overlap=80) splits documents into semantically coherent segments
- **Embedding**: OpenAI text-embedding-3-small vectorizes each chunk
- **Storage**: Chroma with independent domain collections (`rag_manual`, `rag_hr`, `rag_it`, `rag_sales`, `rag_education`)
- **Caching**: `_retrievers` dictionary caches retriever objects in memory to prevent redundant vector DB reloads
- **Retrieval**: Top-K similarity search per domain collection

### f. Key Code Structure

```
langgraph-rag-service/
├── backend/
│   ├── app/
│   │   ├── graph.py      # LangGraph 3-node pipeline
│   │   ├── router.py     # LLM domain classifier + keyword fallback
│   │   ├── rag.py        # Multi-domain Chroma vector store
│   │   ├── tools.py      # Domain RAG tools (for agent extension)
│   │   └── config.py     # Environment variable configuration
│   ├── knowledge/        # Domain-specific knowledge documents
│   │   ├── manual/
│   │   ├── hr/
│   │   ├── it/
│   │   ├── sales/
│   │   └── education/
│   └── main.py           # FastAPI application entry point
├── frontend/
│   └── streamlit_app.py  # Chatbot UI
├── tests/
│   └── test_scenarios.md # Test scenario document
└── docker-compose.yml
```

### g. Performance Optimization

- **Retriever memory caching**: Chroma vector DB is loaded only once per domain and cached in the `_retrievers` dictionary, eliminating redundant initialization on subsequent requests
- **Domain-isolated collections**: Scoping retrieval to a single domain collection (not the full DB) reduces search space and improves both speed and accuracy
- **Chunk size tuning**: chunk_size=500, overlap=80 balances context preservation with retrieval precision
- **Docker health check**: `depends_on` + `service_healthy` ensures the frontend waits for the backend to be ready before starting, preventing connection errors on cold start

### h. Problem-Solving Process

**Issue 1: LLM router returning unexpected output**
- Cause: LLM occasionally returned domain names with explanations or hallucinated domain keys not in the valid set
- Solution: Implemented a dual safety net — strong System Prompt constraints forcing a single domain key output, plus keyword-based fallback classification when the LLM response fails validation

**Issue 2: Multimodal message parsing error**
- Cause: When an image was attached, LangChain HumanMessage content was delivered as a list, breaking the existing string-based text extraction logic
- Solution: Implemented `_get_last_human_text()` helper function that checks content type and extracts only the `type: "text"` part from list-format content

**Issue 3: Redundant vector DB loading**
- Cause: Each request re-initialized the Chroma retriever, causing unnecessary memory use and latency
- Solution: Module-level `_retrievers` dictionary caches initialized retrievers per domain; subsequent requests reuse the cached retriever

---

## 4. Results

### a. Key Results (Metrics & Performance)

| Metric | Result |
|--------|--------|
| Domain routing accuracy | **13/13 (100%)** — Exceeded 90% target |
| Requirements fulfilled | **14/14 (100%)** |
| Supported domains | 5 (Manual / HR / IT / Sales / Education) |
| Knowledge documents | 10 files |
| Test cases passed | 13/13 |

**Domain Routing Accuracy Validation**

| Query | Expected Domain | Actual Domain | Result |
|-------|----------------|---------------|--------|
| How to handle a pressure alarm on Line 3? | manual | 🔧 Field Manual | ✅ |
| Filter replacement cycle for Equipment A? | manual | 🔧 Field Manual | ✅ |
| What does error code E-204 mean? | manual | 🔧 Field Manual | ✅ |
| What are the travel expense rules? | hr | 👤 HR Policy | ✅ |
| How does leave carryover work? | hr | 👤 HR Policy | ✅ |
| How do I apply for remote work? | hr | 👤 HR Policy | ✅ |
| VPN won't connect | it | 💻 IT Helpdesk | ✅ |
| SAP account locked | it | 💻 IT Helpdesk | ✅ |
| How to fix email storage exceeded? | it | 💻 IT Helpdesk | ✅ |
| Difference between Product A and B | sales | 📊 Sales Support | ✅ |
| How to respond when a customer asks to compare us to competitors? | sales | 📊 Sales Support | ✅ |
| What are the quality inspection standards? | education | 📚 Education/Onboarding | ✅ |
| Explain the entire production process | education | 📚 Education/Onboarding | ✅ |

### b. Improvements Over Previous Approach

| Aspect | Before (Single LLM Call) | After (LangGraph RAG) |
|--------|--------------------------|------------------------|
| Answer basis | Model training data (hallucination risk) | Actual internal knowledge documents |
| Domain handling | Single undifferentiated response | Automatic routing across 5 domains |
| Source transparency | None | Source document filenames provided |
| Image analysis | Not supported | Field photo upload + multimodal analysis |
| Knowledge updates | Model retraining required | Instant update by replacing document files |

### c. Demo / Service Link

<!-- Fill in manually -->

### d. Screenshots

<!-- Fill in manually -->

---

## 5. Challenges & Solutions

### a. Technical Challenges

**LLM Router Reliability**
- Problem: LLM occasionally returned domain keys with extra explanation or produced keys not in the valid domain set
- Solution: Strictly constrained System Prompt to output only one domain keyword, with a keyword-matching fallback (`_keyword_fallback`) when the LLM response fails validation. Achieved 100% routing accuracy in all 13 test cases.

**Multimodal Message Handling**
- Problem: When an image was attached, the HumanMessage `content` field was delivered as a list instead of a string, causing the existing text extraction logic to fail
- Solution: Implemented a dedicated `_get_last_human_text()` helper that iterates messages in reverse, detects list-format content, and extracts only the `type: "text"` part

**Multi-Domain Vector DB Design**
- Problem: Storing all domain documents in a single Chroma collection caused cross-domain noise, degrading retrieval quality
- Solution: Structured independent Chroma collections per domain, so retrieval is always scoped to the collection that matches the router's decision — fully eliminating cross-domain interference

### b. Team Collaboration Challenges

N/A — Individual project

### c. Performance Challenges

- Problem: Chroma vector DB was re-initialized on every request, causing repeated latency even after the first call
- Solution: Maintained a module-level `_retrievers` dictionary that caches the retriever object for each domain after the first initialization. All subsequent requests are served from the in-memory cache, concentrating initialization cost to the first call only.

---

## 6. Retrospective

### a. Skills Learned
- **LangGraph**: Designed multi-node AI pipelines using StateGraph. Gained hands-on experience with TypedDict-based state management and the `add_messages` reducer for conversation history.
- **RAG Architecture**: Directly experimented with chunking strategies (chunk size, overlap) and observed how they affect retrieval quality to arrive at optimal values.
- **LLM Prompt Engineering**: Learned the importance of explicit constraint specification in System Prompts through iterating on the router prompt to reliably produce single-keyword outputs.
- **Multimodal LLM Integration**: Gained a deep understanding of Base64-based image-text co-processing and the structure of LangChain message types.
- **FastAPI + Streamlit Integration**: Implemented REST API design and asynchronous frontend-backend communication from scratch.
- **Docker Compose Health Checks**: Learned how to guarantee reliable service startup ordering using `depends_on` + `service_healthy` conditions.

### b. Areas for Improvement
- **Streaming responses**: The current architecture returns the complete answer in a single response. Integrating LLM streaming API would significantly improve perceived latency.
- **Routing confidence display**: Showing the router's classification confidence to users would let them manually override the domain when misclassification occurs, improving UX.
- **Knowledge management UI**: Currently, updating the knowledge base requires directly adding/editing `.txt` files on the filesystem. An admin UI for document upload and deletion via the web would improve operational convenience.
- **Evaluation pipeline**: Building an automated RAG evaluation pipeline (e.g., using RAGAS) to measure routing accuracy and answer quality (relevance, faithfulness) would enable continuous quality monitoring.

### c. Extensibility
- **New domain addition**: The system is designed so that adding a new folder with `.txt` files under `knowledge/` automatically triggers vector indexing and routing — making domain expansion straightforward.
- **Agent architecture upgrade**: Domain-specific RAG tools are already implemented in `tools.py`, making migration to a LangGraph agent architecture straightforward.
- **Real-time data integration**: Adding live data sources (MES, production DB, quality data) alongside document RAG would support advanced queries like "Why did the defect rate spike this week?"
- **Multi-tenant support**: Adding department-level independent knowledge bases and access control would make the system enterprise-grade for large organizations.
- **Automated report generation**: Combining accumulated conversation history with KPI documents could extend the system to an automated management report generation feature.

---

## 7. References

### a. GitHub

<!-- Fill in manually -->

### b. Documentation

<!-- Fill in manually -->

### c. Presentation Materials

<!-- Fill in manually -->

### d. Papers / References

<!-- Fill in manually -->
