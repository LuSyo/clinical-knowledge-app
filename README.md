# Clinical Knowledge Graph RAG Framework

An agentic, model-agnostic clinical decision support system built with **LangChain**, **LangGraph**, and **ChromaDB**. Designed to translate unstructured NICE guidelines into a reliable, self-correcting knowledge pipeline.

## Overview

This project implements a **Corrective RAG (CRAG)** architecture to solve the "hallucination problem" in clinical AI. Unlike standard RAG, this system autonomously evaluates the relevance of retrieved evidence before generating medical advice, ensuring that responses are grounded strictly in validated clinical guidelines.

### Key features
* **Agentic orchestration:** Uses `LangGraph` to manage a stateful workflow with conditional branching.
* **Multi-layered guardrails:** Features an Entry Router (Intent) and a Document Grader (Grounding) (and more safety and evaluation features to come!)
* **Structured ingestion (EDC):** UPCOMING: Implements the *Extract-Define-Canonicalize* framework to transform text into relational triplets.

---

## Graph architecture

The system follows a non-linear directed acyclic graph (DAG):

1.  **Entry routing:** Analyses the query to determine if it requires clinical guidelines retrieval or a direct response.
2.  **Context retrieval:** Performs semantic search against a ChromaDB vector store.
3.  **Corrective grading:** A dedicated "Grader" node evaluates if the retrieved documents actually answer the specific clinical question.
4.  **Conditional generation:** 
    * **Success:** If relevant, the LLM generates a grounded response.
    * **Failure:** If irrelevant, the system triggers a rejection response to prevent hallucinations and unsafe clinical recommendations.

```text
[ START ]
           |
           v
    +--------------+
    | route_query  | (Router Node: Intent Guardrail)
    +--------------+
           |
           +--------------------------+
           |                          |
    [ vectorstore ]           [ direct_response ]
           |                          |
           v                          |
    +--------------+                  |
    |   retrieve   |                  |
    +--------------+                  |
           |                          |
           v                          |
    +--------------+                  v
    |grade_document|          +--------------+
    +--------------+          |   generate   |
           |                  +--------------+
           v                          |
    +--------------+                  |
    |  generate_   |                  |
    |  or_reject   |                  |
    +--------------+                  |
           |                          |
    +------+-------+                  |
    |              |                  |
[ generate ]    [ reject ]            |
    |              |                  |
    v              v                  |
+----------+   +--------------+       |
| generate |   |   reject     |       |
+----------+   +--------------+       |
    |              |                  |
    +------+-------+------------------+
           |
           v
        [ END ]
```

---

## Tech Stack

* **Orchestration:** LangChain & LangGraph
* **State management:** Pydantic
* **Vector database:** ChromaDB
* **LLMs:** GPT-4o (Extraction/Grading), GPT-4o-mini (Routing/Generation)
* **Embeddings:** OpenAI `text-embedding-3-small`

---

## Project Structure

```text
├── src/
│   ├── pipeline/
│   │   ├── graph.py        # Graph definition and state machine logic
│   │   ├── nodes.py        # Functional logic for Retrieve, Grade, Generate
│   ├── data_processing/
│   │   ├── extractor.py    # PDF text extraction and splitting
│   │   └── vector_store.py # ChromaDB management
│   ├── evaluation/    # UPCOMING!
│   ├── schema.py           # GraphState
│   ├── main.py             # Entry point for the agent
│   └── rag_setup.py        # Entry point for the RAG setup
├── data/                   # ChromaDB persistence and raw guidelines
└── requirements.txt        # Production dependencies