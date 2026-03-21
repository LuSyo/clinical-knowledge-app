import os
from dotenv import load_dotenv

from data_processing.extractor import process_source_files
from data_processing.vector_store import create_vector_store

def main():
  load_dotenv()
  api_key = os.getenv("OPENAI_API_KEY")

  if not api_key:
    raise ValueError("OPENAI_API_KEY not found.")

  source_folder= "./data/sources"
  db_path= "./data/chroma_db"
  
  chunks = process_source_files(source_folder)

  if chunks:
    print(f'Successfully created {len(chunks)} chunks')

    vector_store = create_vector_store(chunks, persist_directory=db_path)

  else:
    print("No documents found or processed. Check your data/sources folder.")

if __name__ == "__main__":
  main()