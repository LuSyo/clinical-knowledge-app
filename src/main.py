import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from pipeline.rag import get_clinical_answer
from pipeline.graph import build_graph
from evaluation.judge import evaluate_response
from schema import GraphState

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
      raise ValueError("OPENAI_API_KEY not found.")
    
    app = build_graph()
    print(app.get_graph().draw_ascii())
    
    # test query
    query = "What are the recommendations for anticoagulation in atrial fibrillation?"
    
    # initial state for the graph
    initial_state = GraphState(
      messages=[HumanMessage(content=query)]
    )

    # invoke the graph
    final_state = app.invoke(initial_state)

    answer = final_state["messages"][-1].content
    print(f"\nAI Response:\n{answer}")
    
    print(f"\nRetrieved Context Chunks: {len(final_state['context'])}")

    # db_path = "./data/chroma_db"
    # print(f"\n--- Processing Query: {query} ---")
    
    # result = get_clinical_answer(query, db_path)
    
    # answer = result["answer"]
    # # Combine the text of all retrieved chunks for the judge
    # context_text = "\n".join([doc.page_content for doc in result["context"]])
    
    # print(f"\nAI Response:\n{answer}")

    # # 2. Phase 1b: Basic LLM-as-a-Judge Evaluation
    # print("\n--- Starting Evaluation ---")
    # evaluation = evaluate_response(query, answer, context_text)
    # print(f"Judge's Verdict:\n{evaluation}")

if __name__ == "__main__":
    main()