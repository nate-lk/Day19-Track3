# Tech Company GraphRAG Pipeline

A robust implementation of a Graph-based Retrieval-Augmented Generation (GraphRAG) system, designed to extract knowledge from unstructured text and perform multi-hop reasoning over a Knowledge Graph.

## 🚀 Overview

This project implements a complete GraphRAG pipeline that:
1.  **Extracts Triples:** Uses GPT-4o to extract entities and relationships from a text corpus.
2.  **Builds Graphs:** Constructs a Knowledge Graph in both **Neo4j** (for persistent storage and querying) and **NetworkX** (for in-memory analysis).
3.  **Semantic Entity Matching:** Uses vector embeddings (FAISS) to map natural language entities from questions to actual nodes in the graph.
4.  **Multi-hop Retrieval:** Performs 2-hop traversal in Neo4j to gather rich relational context for answering complex questions.
5.  **Evaluation:** Compares GraphRAG performance against standard "Flat" Vector RAG.

## 🛠️ Tech Stack

*   **LLM:** OpenAI GPT-4o
*   **Orchestration:** LangChain
*   **Graph Databases:** Neo4j, NetworkX
*   **Vector Database:** FAISS (for entity resolution)
*   **Data Science:** Pandas, Matplotlib

## 📋 Prerequisites

*   Python 3.10+
*   A running Neo4j instance (local or AuraDB)
*   OpenAI API Key

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Day19-Track3
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_openai_key_here
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password
    ```

## 🚀 Running the Pipeline

To run the full indexing, construction, and evaluation process:

```bash
python graph_rag_pipeline.py
```

## 🧠 Key Features

### Semantic Entity Matching
One of the most robust features of this pipeline is its ability to handle "out-of-graph" references. If a user asks about a "competitor" not explicitly named in the graph, the system uses semantic similarity to find the most relevant nodes (e.g., matching "competitor" or "rival" to "Google").

### Multi-hop Reasoning
Unlike standard RAG, which only finds similar text chunks, this system traverses relationships. It can answer questions like:
> "What is the relationship between the company that makes chips for OpenAI and the company that made Gemini?"

By traversing `(OpenAI) -> (Nvidia) -> (GPU Hardware) <- (Google) <- (Gemini)`, the system provides a structured, relational answer.

## 📂 Project Structure

*   `graph_rag_pipeline.py`: The main entry point and core logic.
*   `tech_company_corpus.txt`: The source dataset for building the graph.
*   `instruction.md`: Original lab requirements and objectives.
*   `requirements.txt`: Python dependencies.

## 📊 Evaluation Results
The system is evaluated on 5 complex questions where GraphRAG typically outperforms Flat RAG by providing more structured and contextually accurate answers for relational queries.
