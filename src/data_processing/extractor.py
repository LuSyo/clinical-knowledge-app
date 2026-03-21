import os

from PyPDF2 import PdfReader
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file: str):
  try:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
      content = page.extract_text()
      if content:
        text += content + "\n"
    
    return text
  except Exception as e:
    return f"Error: {e}"


def process_source_files(folder_path: str):
  if not os.path.exists(folder_path):
    print(f'Folder not found: {folder_path}')
    return []
  
  md = MarkItDown()
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=100,
      separators=["\n\n", "\n", ".", " "]
  )

  all_chunks = []
  for filename in os.listdir(folder_path):
    if filename.startswith('.'): continue
    full_path = os.path.join(folder_path, filename)

    try:
      result = md.convert(full_path)
      markdown_text = result.text_content

      chunks = text_splitter.create_documents(
        texts=[markdown_text],
        metadatas=[{"source": filename}]
      )
      all_chunks.extend(chunks)

    except Exception as e:
      print(f"Error processing {filename}: {e}")

  return all_chunks