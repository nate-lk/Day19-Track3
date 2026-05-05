from graph_rag_pipeline import GraphRAGPipeline
import os
from dotenv import load_dotenv

load_dotenv()

def list_entities():
    pipeline = GraphRAGPipeline()
    try:
        with pipeline.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN n.name as name")
            entities = [record["name"] for record in result]
            print("Entities in Graph:")
            for e in sorted(entities):
                print(f"- {e}")
            
            print("\nRelationships in Graph:")
            result = session.run("MATCH (s:Entity)-[r]->(o:Entity) RETURN s.name as s, type(r) as p, o.name as o")
            for record in result:
                print(f"({record['s']}) --[{record['p']}]--> ({record['o']})")
    finally:
        pipeline.close()

if __name__ == "__main__":
    list_entities()
