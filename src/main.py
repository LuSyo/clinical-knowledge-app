import os
from dotenv import load_dotenv

from data_processing.extractor import process_source_files

def main():
  load_dotenv()
  api_key = os.getenv("OPENAI_API_KEY")

  if not api_key:
    raise ValueError("OPENAI_API_KEY not found.")

  source_folder= "./data/sources"
  process_source_files(source_folder)

if __name__ == "__main__":
  main()