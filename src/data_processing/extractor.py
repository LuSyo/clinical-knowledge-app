import os

from PyPDF2 import PdfReader

def extract(file: str):
  try:
    reader = PdfReader(file)
    pages = len(reader.pages)
    return pages
  except Exception as e:
    return f"Error: {e}"
  
def process_source_files(folder_path: str):
  if not os.path.exists(folder_path):
    print(f'Folder not found: {folder_path}')
    return
  
  for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)
    print(f'{filename}: {extract(full_path)} pages')