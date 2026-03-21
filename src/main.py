import os
from dotenv import load_dotenv

from pipeline.rag import get_clinical_answer
from evaluation.judge import evaluate_response

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
      raise ValueError("OPENAI_API_KEY not found.")
    
    db_path = "./data/chroma_db"
    
    # 1. Run a Test Query
    query = "What are the recommendations for anticoagulation in atrial fibrillation?"
    print(f"\n--- Processing Query: {query} ---")
    
    result = get_clinical_answer(query, db_path)
    
    answer = result["answer"]
    # Combine the text of all retrieved chunks for the judge
    context_text = "\n".join([doc.page_content for doc in result["context"]])
    
    print(f"\nAI Response:\n{answer}")

    # 2. Phase 1b: Basic LLM-as-a-Judge Evaluation
    print("\n--- Starting Evaluation ---")
    evaluation = evaluate_response(query, answer, context_text)
    print(f"Judge's Verdict:\n{evaluation}")

if __name__ == "__main__":
    main()