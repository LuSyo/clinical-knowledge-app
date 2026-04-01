import os
import json

from markitdown import MarkItDown
from typing import List, Union, cast, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from schema import ClinicalLogic, Triplet, TripletExtraction, ClinicalFact
from utils import setup_logger

logger = setup_logger(log_dir="./logs", exp_name="ingestion_pipeline")

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

def extract_clinical_triplets(text_chunk: str) -> List[Triplet]:
    """
    Performs Open Information Extraction (OIE) to discover entities 
    and relationships dynamically from clinical text.
    """
    # LLM with structured output
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(TripletExtraction)

    # extraction prompt
    system_prompt = """
    You are a clinical data engineer extracting actionable knowledge for a decision support system.

    TASK: Extract clinical knowledge items and categorize them as either 'fact' or 'logic'.

    1. CLINICAL FACTS:
      - Used for static relationships (e.g. 'Aspirin is_a NSAID', 'Metformin treats Type 2 Diabetes').
      - Format: [Subject] -> [Predicate] -> [Target].

    2. CLINICAL LOGIC:
      - Used for conditional instructions or diagnostic pathways (e.g. 'If symptoms persist, increase dosage').
      - Format: IF [Condition] -> THEN [Action] -> ON [Target].

    CRITICAL FILTER:
    1. IGNORE generic meta-information (e.g., 'medicine is used for treatment', 'report events to MHRA').
    2. EXTRACT ONLY actionable clinical instructions or factual definitions.
    
    Be specific: use 'first-line-treatment' or 'contraindicated-with' instead of 'related-to'
    Ensure Subject and Object are concise nouns.
    """

    # invoke and return triplets
    try:
      result = cast(TripletExtraction, structured_llm.invoke([
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": text_chunk}
      ]))
      return result.triplets
    except Exception as e:
        print(f"Extraction failed for chunk: {e}")
        return []
    
# def batch_extract(chunks, batch_size=5, logger=None):
#   all_triplets = []
#   for i in range(0, len(chunks), batch_size):
#     batch = chunks[i:i + batch_size]
#     # Combine text for the LLM
#     combined_text = "\n---\n".join([c.page_content for c in batch])
    
#     if logger: logger.info(f"Batch processing chunks {i} to {i+batch_size}...")
#     triplets = extract_clinical_triplets(combined_text)
#     all_triplets.append(triplets)

#   return all_triplets

class TriageResult(BaseModel):
    """Binary decision on whether a chunk contains actionable clinical knowledge."""
    is_actionable: bool = Field(
        description="True if the text contains specific clinical instructions, diagnostic criteria, or treatment pathways. False otherwise (e.g. it is boilerplate, TOC, or generic info)."
    )

def triage_chunk(text_chunk: str) -> bool:
    """
    Checks if a chunk contains actionable clinical relationships.
    """
    # cheap model for binary classification
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(TriageResult)
    
    system_prompt = "You are a clinical data triage officer. Your job is to determine if a text chunk is worth the cost of deep extraction."
    
    try:
      result = cast(TriageResult, structured_llm.invoke([
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": text_chunk[:900]}
      ]))
      
      # Log the decision
      logger.info(f"TRIAGE DECISION: {result.is_actionable}")
      return result.is_actionable
    except Exception as e:
      logger.error(f"Triage failed, defaulting to True for safety: {e}")
      return True 

def canonicalise_fact(fact: ClinicalFact, source_chunk: str) -> Union[ClinicalFact, ClinicalLogic]:
  """
  Uses a small LLM call to map a raw predicate to the canonical list and re-classify as logic if necessary
  """
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
  structured_llm = llm.with_structured_output(Triplet)

  with open('./config/canonical_predicates.json', 'r') as f:
    CANONICAL_PREDICATES = json.load(f)
  
  system_prompt = f"""
  You are a clinical ontologist. 

  1. VERIFY: Ensure the 'Fact' accurately reflects the 'Source Text'. If the text implies a condition (e.g., 'If', 'When', 'In cases of'), you MUST promote it to Logic. Pay special attention to generic relationships such as "IS_ASSOCIATED_WITH"
  
  2. CANONICALIZE: If it remains a fact, map the raw relationship to the BEST fit from this list of canonical predicates:
  {CANONICAL_PREDICATES}. If no good fit exists, or if the relationship is the reverse to a canonical predicate, return the raw predicate in ALL_CAPS.
  """
  
  user_prompt = f"""
  Source text: {source_chunk}
  Raw relationship: {fact.subject} -> {fact.predicate} -> {fact.object}
  """
  
  try:
    response = cast(Triplet, structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]))
    
    return response.item
  except Exception as e:
    logger.error(f"Canonicalization failed: {e}")
    return fact