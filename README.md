## Multi-Doctor Collaborative Diagnosis Simulator: Architecture & File Structure

This project simulates a panel of medical experts collaborating to diagnose a patient. Each expert leverages domain‑specific knowledge, a shared medical knowledge graph, and retrieval‑augmented generation (RAG) via LlamaIndex. A Mixture of Experts (MoE) architecture orchestrates the experts, resolves conflicts, and produces a final diagnosis.

---

### System Architecture

The architecture is modular, allowing independent development and scaling of components. It consists of the following layers:

```
┌─────────────────┐
│   Input Layer   │  Patient data (symptoms, history, test results)
└────────┬────────┘
         │
┌────────▼────────┐
│  Expert Modules │  Parallel reasoning by specialised doctors
│  (Cardiology,   │  Each expert: LLM + domain KB + RAG
│   Neurology,…)  │
└────────┬────────┘
         │
┌────────▼────────┐
│ Knowledge Graph │  Central graph (UMLS/SNOMED) linking entities
│   Integration   │  Queried by all experts for consistent context
└────────┬────────┘
         │
┌────────▼────────┐
│  Mixture of     │  Router selects relevant experts & weights
│   Experts (MoE) │  Gate combines outputs; conflict resolver
└────────┬────────┘
         │
┌────────▼────────┐
│   Output Layer  │  Final diagnosis, differentials, confidence,
│                 │  reasoning traces from each expert
└─────────────────┘
```

#### Detailed Component Descriptions

1. **Input Layer**  
   - Accepts structured (JSON) or free‑text patient data.  
   - Normalises and validates input (e.g., symptom extraction, unit conversion).

2. **Expert Modules**  
   - Each expert corresponds to a medical specialty (cardiology, neurology, etc.).  
   - **Internal structure**:  
     - A base LLM (e.g., GPT‑4, fine‑tuned medical model) with a specialty‑specific prompt.  
     - Access to a **domain‑specific knowledge base** (local guidelines, textbooks).  
     - **RAG via LlamaIndex**: retrieves relevant literature or guidelines from indexed medical corpora.  
     - Optional connection to the **knowledge graph** for entity disambiguation and relationship retrieval.  
   - Experts run in parallel (simulated concurrency) and produce:  
     - A differential diagnosis list with confidence scores.  
     - Supporting evidence (citations, reasoning steps).  
     - Any uncertainty or need for additional tests.

3. **Knowledge Graph Integration**  
   - Central graph database (e.g., Neo4j) populated with medical ontologies (SNOMED CT, UMLS, etc.).  
   - Provides a unified vocabulary and relationships (e.g., “symptom X is associated with disease Y”).  
   - All experts query the graph to ground their reasoning in established medical knowledge.  
   - Graph updates can be periodic or on‑demand from trusted sources.

4. **Mixture of Experts (MoE) Layer**  
   - **Router**: Based on patient data, selects which experts to consult and assigns initial weights (e.g., symptoms pointing to cardiology give higher weight to cardiologist).  
   - **Gate**: Collects expert outputs and combines them. Could be a simple weighted average of confidence scores or a learned model that synthesises free‑text rationales.  
   - **Conflict Resolver**: Handles disagreements (e.g., two experts propose mutually exclusive diagnoses). Strategies include:  
     - **Debate simulation**: Experts exchange evidence and refine opinions.  
     - **Meta‑expert**: A separate model reviews conflicting outputs and decides.  
     - **Consensus voting**: With confidence thresholds.

5. **Output Layer**  
   - Final diagnosis (primary and differential).  
   - Confidence interval.  
   - Traceability: each expert’s reasoning and the conflict‑resolution process.

---

### Technology Stack Suggestions

- **Backend**: Python (FastAPI for APIs, Celery for async expert tasks)  
- **LLM Integration**: LangChain or LlamaIndex (for RAG), HuggingFace Transformers (for local models)  
- **Knowledge Graph**: Neo4j (graph DB) + py2neo or similar  
- **MoE Implementation**: Custom router/gate; could use a small neural network or rule‑based logic initially  
- **Data Storage**: PostgreSQL for patient records (optional), vector DB (Chroma/FAISS) for RAG indexes  
- **Containerisation**: Docker, docker‑compose for orchestration

---

### File Structure

A clean, modular directory layout facilitates collaboration and scaling.

```
multi-doctor-diagnosis/
│
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env.example
│
├── config/                     # Configuration files
│   ├── experts.yaml            # List of experts and their parameters
│   ├── kg_config.yaml          # Knowledge graph connection details
│   └── moe_config.yaml          # Router/gate settings
│
├── data/                        # Data assets (not code)
│   ├── knowledge_graph/         # Dumps / Cypher scripts for graph population
│   ├── medical_corpus/          # Raw text files for RAG (guidelines, textbooks)
│   ├── indexes/                  # Pre‑built LlamaIndex indexes (saved to disk)
│   └── patient_examples/         # Sample patient cases for testing
│
├── src/
│   ├── __init__.py
│   │
│   ├── api/                      # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py                # API endpoints
│   │   ├── dependencies.py        # Dependency injection
│   │   └── schemas.py             # Pydantic models for request/response
│   │
│   ├── core/                      # Core orchestration logic
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # Main pipeline: input → experts → MoE → output
│   │   └── pipeline.py             # Async task management
│   │
│   ├── experts/                    # Expert modules
│   │   ├── __init__.py
│   │   ├── base_expert.py          # Abstract base class for all experts
│   │   ├── cardiology_expert.py
│   │   ├── neurology_expert.py
│   │   ├── ... (other specialties)
│   │   ├── expert_factory.py       # Creates expert instances from config
│   │   └── utils.py                 # Shared helper functions (prompt templates)
│   │
│   ├── knowledge_graph/            # Graph integration
│   │   ├── __init__.py
│   │   ├── connector.py             # Neo4j connection and query execution
│   │   ├── queries.py                # Predefined Cypher queries
│   │   └── entity_resolver.py        # Map free text to graph entities
│   │
│   ├── rag/                         # Retrieval-Augmented Generation
│   │   ├── __init__.py
│   │   ├── index_builder.py          # Build LlamaIndex from corpus
│   │   ├── retriever.py              # Retrieve relevant documents
│   │   └── generator.py              # Generate text with context
│   │
│   ├── moe/                          # Mixture of Experts
│   │   ├── __init__.py
│   │   ├── router.py                  # Select experts and assign weights
│   │   ├── gate.py                     # Combine expert outputs
│   │   ├── conflict_resolver.py        # Handle disagreements
│   │   └── models/                      # Optional: trained models for routing
│   │
│   ├── models/                       # (If using custom ML models)
│   │   └── ...
│   │
│   └── utils/                        # General utilities
│       ├── logger.py
│       ├── text_processor.py          # Symptom extraction, normalisation
│       └── validators.py               # Input validation
│
├── tests/                            # Unit and integration tests
│   ├── test_experts/
│   ├── test_moe/
│   ├── test_api/
│   └── conftest.py
│
└── notebooks/                        # Exploration and prototyping
    ├── kg_exploration.ipynb
    ├── rag_demo.ipynb
    └── moe_experiments.ipynb
```

### Key Design Decisions Explained

- **Parallel Expert Execution**: Each expert runs independently, enabling true multi‑doctor simulation. Asynchronous task queues (Celery) or Python’s `asyncio` can be used.  
- **Knowledge Graph as Shared Memory**: All experts query the same graph, ensuring consistency and avoiding contradictory entity definitions.  
- **LlamaIndex for RAG**: Provides flexible indexing over heterogeneous medical texts and integrates easily with LLMs.  
- **MoE with Explicit Conflict Resolution**: Moving beyond simple averaging, we include a dedicated resolver to mimic real‑world medical consensus‑building.  
- **Configuration‑Driven**: `config/` files allow adding/removing experts, tuning weights, and swapping knowledge sources without code changes.
