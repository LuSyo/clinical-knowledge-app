import os
import hashlib
from dotenv import load_dotenv
from collections import Counter
from utils import setup_logger
from data_processing.extraction import process_source_files, extract_clinical_triplets, triage_chunk
from data_processing.vector_store import create_vector_store

logger = setup_logger(log_dir="./logs", exp_name="ingestion_pipeline")

def main():
  load_dotenv()
  api_key = os.getenv("OPENAI_API_KEY")

  if not api_key:
    raise ValueError("OPENAI_API_KEY not found.")
  
  source_folder= "./data/sources"
  db_path= "./data/chroma_db"
  
  chunks = process_source_files(source_folder)

  if chunks:
    discovered_predicates = Counter()
    discovered_entities = Counter()
    processed_chunks = []

    for i, chunk in enumerate(chunks):
      
      if i > 10: break # TEST LIMIT

      if not triage_chunk(chunk.page_content):
        logger.info(f"Skipping Chunk {i}: Non-clinical/Boilerplate.")
        continue

      logger.info(f"Extracting clinical knowledge from Chunk {i}...")

      triplets = extract_clinical_triplets(chunk.page_content)
      triplet_list = []

      for t in triplets:
        discovered_predicates[t.predicate] += 1
        discovered_entities[t.subject] += 1 
        discovered_entities[t.object] += 1
        triplet_list.append(f"[{t.subject} -> {t.predicate} -> {t.object}]")

      if triplet_list:
        triplet_str = "; ".join(triplet_list)
        chunk.page_content = f"Relationships: {triplet_str}\n\nText: {chunk.page_content}"
        chunk.metadata["triplets"] = triplet_str
        logger.info(triplet_str)

      processed_chunks.append(chunk)
    
    if processed_chunks:
      vector_store = create_vector_store(processed_chunks, persist_directory=db_path)
      logger.info(f"Knowledge Graph-augmented Vector Store created successfully with {len(processed_chunks)} chunks.")
    
    logger.info("--- AUTONOMOUS SCHEMA DISCOVERY REPORT ---")
    logger.info(f"Top 10 Discovered Predicates: {discovered_predicates.most_common(10)}")
    logger.info(f"Unique entities found: {len(discovered_entities)}")

  else:
    logger.info("No documents found or processed. Check your data/sources folder.")

if __name__ == "__main__":
  main()