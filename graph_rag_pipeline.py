import os
import networkx as nx
import matplotlib.pyplot as plt

# FORCE TIKTOKEN TO USE LOCAL CACHE
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "tiktoken_cache")

from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# 0. SETUP & CONFIGURATION
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class Triple(BaseModel):
    subject: str = Field(description="The source entity (Node). Use concise names, e.g., 'OpenAI' not 'OpenAI founded in 2015'")
    predicate: str = Field(description="The relationship between subject and object, e.g., 'FOUNDED_BY'")
    obj: str = Field(description="The target entity (Node). Use concise names.")

class KnowledgeGraph(BaseModel):
    triples: List[Triple] = Field(description="List of extracted triples from the text")

class EntityList(BaseModel):
    entities: List[str] = Field(description="List of important entities identified in the text")

class GraphRAGPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.structured_llm = self.llm.with_structured_output(KnowledgeGraph)
        self.entity_extractor = self.llm.with_structured_output(EntityList)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.embeddings = OpenAIEmbeddings()
        self.nx_graph = nx.MultiDiGraph()
        self.vectorstore = None # For semantic entity matching

    def close(self):
        self.driver.close()

    # STEP 1: INDEXING (EXTRACTION)
    def extract_triples(self, text: str) -> List[Triple]:
        print("--- Extracting Triples ---")
        prompt = ChatPromptTemplate.from_template(
            "Extract entities and their relationships from the following text as a knowledge graph.\n"
            "Include key facts like headquarters, founders, investments, and competitors.\n"
            "Keep entity names concise but meaningful (e.g., '2019 Investment' or '$1 Billion' can be nodes if they represent a specific fact).\n\n"
            "Text: {text}"
        )
        chain = prompt | self.structured_llm
        result = chain.invoke({"text": text})
        return result.triples

    # STEP 2: CONSTRUCTION (NEO4J & NETWORKX)
    def build_graph(self, triples: List[Triple]):
        print("--- Building Graphs (Neo4j & NetworkX) ---")
        all_entities = set()
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
            for triple in triples:
                # Neo4j
                query = (
                    "MERGE (s:Entity {name: $subject}) "
                    "MERGE (o:Entity {name: $object}) "
                    "MERGE (s)-[r:RELATION {type: $predicate}]->(o)"
                )
                session.run(query, subject=triple.subject, object=triple.obj, predicate=triple.predicate)
                
                # NetworkX
                self.nx_graph.add_edge(triple.subject, triple.obj, relation=triple.predicate)
                all_entities.add(triple.subject)
                all_entities.add(triple.obj)
        
        # Build a vector store of entity names for semantic matching
        entity_docs = [Document(page_content=e) for e in all_entities]
        self.vectorstore = FAISS.from_documents(entity_docs, self.embeddings)
        
        print(f"Successfully indexed {len(triples)} triples.")

    # STEP 3: SEMANTIC ENTITY MATCHING
    def get_relevant_entities(self, question: str) -> List[str]:
        # 1. Ask LLM what entities it thinks are relevant
        prompt = ChatPromptTemplate.from_template(
            "Identify 2-3 key entities (companies, people, products) in this question: {question}. Return them as a list."
        )
        llm_entities = (prompt | self.entity_extractor).invoke({"question": question}).entities
        
        # 2. Use vector search to find the actual names used in our graph
        matched_entities = []
        for ent in llm_entities:
            # Find the top 2 matches for each extracted entity to be safe
            matches = self.vectorstore.similarity_search(ent, k=2)
            for m in matches:
                matched_entities.append(m.page_content)
        
        return list(set(matched_entities))

    # STEP 4: QUERYING (MULTI-HOP RETRIEVAL)
    def query_graph(self, question: str) -> str:
        print(f"--- Querying Graph: {question} ---")
        
        main_entities = self.get_relevant_entities(question)
        print(f"Matched Entities in Graph: {main_entities}")

        if not main_entities:
            return "No relevant entities found in the graph."

        # 4.2 Fetch broader neighborhood from Neo4j
        relationships = []
        with self.driver.session() as session:
            for entity in main_entities:
                # Use a slightly broader match to get more context
                query = (
                    "MATCH (e:Entity {name: $name})-[r1]-(n1) "
                    "OPTIONAL MATCH (n1)-[r2]-(n2) "
                    "RETURN e.name as s, type(r1) as p1, n1.name as o1, type(r2) as p2, n2.name as o2"
                )
                result = session.run(query, name=entity)
                for record in result:
                    relationships.append(f"{record['s']} --{record['p1']}--> {record['o1']}")
                    if record['o2']:
                        relationships.append(f"{record['o1']} --{record['p2']}--> {record['o2']}")
            
        context = "\n".join(list(set(relationships)))

        if not context:
            return "No relevant graph relationships found."

        # 4.3 Generate final answer
        qa_prompt = ChatPromptTemplate.from_template(
            "You are a GraphRAG assistant. Use the following knowledge graph context to answer the question.\n"
            "Each line represents a relationship: Entity1 --RELATION--> Entity2.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        answer = self.llm.invoke(qa_prompt.format(context=context, question=question)).content
        return answer

# FLAT RAG IMPLEMENTATION
class FlatRAG:
    def __init__(self, text: str):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def query(self, question: str) -> str:
        docs = self.vectorstore.similarity_search(question, k=2)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = ChatPromptTemplate.from_template(
            "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        return self.llm.invoke(prompt.format(context=context, question=question)).content

def main():
    # Load corpus
    with open("tech_company_corpus.txt", "r") as f:
        corpus = f.read()

    pipeline = GraphRAGPipeline()
    flat_rag = FlatRAG(corpus)

    try:
        # Indexing & Construction
        triples = pipeline.extract_triples(corpus)
        pipeline.build_graph(triples)

        # Evaluation Questions
        questions = [
            "Who founded OpenAI and where is it headquartered?",
            "What is the relationship between the company that makes chips for OpenAI and the company that made Gemini?",
            "How has Microsoft's investment in OpenAI evolved over time?",
            "Why did Elon Musk leave OpenAI and what company was he focused on then?",
            "Who is the CEO of the company that competes with OpenAI in the LLM space?"
        ]

        print("\n=== STARTING EVALUATION ===\n")
        for q in questions:
            print(f"Question: {q}")
            print("-" * 20)
            
            print("Flat RAG Answer:")
            print(flat_rag.query(q))
            print("-" * 10)
            
            print("GraphRAG Answer:")
            print(pipeline.query_graph(q))
            print("=" * 40 + "\n")

    finally:
        pipeline.close()

if __name__ == "__main__":
    main()
