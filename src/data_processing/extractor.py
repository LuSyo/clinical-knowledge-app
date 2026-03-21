import os

from PyPDF2 import PdfReader

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
  
  extracted_data = []
  for filename in os.listdir(folder_path):
    full_path = os.path.join(folder_path, filename)
    text = extract_text_from_pdf(full_path)
    extracted_data.append({"filename":{filename}, "content":text})

  return extracted_data