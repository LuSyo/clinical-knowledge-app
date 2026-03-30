import os
from dotenv import load_dotenv
from collections import Counter
from utils import setup_logger
from data_processing.extraction import canonicalise_fact, process_source_files, extract_clinical_triplets, triage_chunk
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
      
      if i > 12: break # TEST LIMIT

      if not triage_chunk(chunk.page_content):
        logger.info(f"Skipping Chunk {i}: Non-clinical/Boilerplate.")
        continue

      logger.info(f"Extracting clinical knowledge from Chunk {i}...")

      triplets = extract_clinical_triplets(chunk.page_content)
      triplet_list = []

      for t in triplets:
        aligned_item = t.item
        if aligned_item.type == 'fact':
          aligned_item = canonicalise_fact(aligned_item, chunk)

          if aligned_item.type == 'fact':
            discovered_predicates[aligned_item.predicate] += 1
            discovered_entities[aligned_item.subject] += 1 
            discovered_entities[aligned_item.object] += 1
            triplet_list.append(f"[Fact: {aligned_item.subject} -> {aligned_item.predicate} -> {aligned_item.object}]")
        
        if aligned_item.type == 'logic':
          discovered_predicates["if_then"] += 1
          discovered_entities[aligned_item.condition] += 1 
          discovered_entities[aligned_item.object] += 1
          triplet_list.append(f"[Logic: IF {aligned_item.condition} -> THEN {aligned_item.action} -> {aligned_item.object}]")

      if triplet_list:
        triplet_str = "; ".join(triplet_list)
        chunk.page_content = f"Structured knowledge: {triplet_str}\n\nText: {chunk.page_content}"
        chunk.metadata["triplets"] = triplet_str
        logger.info(f"Extracted knowledge: {triplet_str}")

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