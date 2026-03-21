from data_processing.extractor import process_source_files

def main():
  source_folder= "./data/sources"
  process_source_files(source_folder)

if __name__ == "__main__":
  main()