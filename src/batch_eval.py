import json
import os
from dotenv import load_dotenv
from datetime import datetime
from pipeline.graph import build_graph
from schema import GraphState
from langchain_core.messages import HumanMessage
from evaluation.judge import evaluate_response

def run_batch_evaluation(dataset_path: str, output_dir: str = "./results"):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
      raise ValueError("OPENAI_API_KEY not found.")
    
    os.makedirs(output_dir, exist_ok=True)
    app = build_graph()
    
    with open(dataset_path, 'r') as f:
        eval_set = json.load(f)

    results = []
    print(f"--- Starting Eval on {len(eval_set)} cases ---")

    for entry in eval_set:
        query = entry["query"]
        ground_truth = entry["ground_truth"]
        
        print(f"Testing: {query[:50]}...")

        # Run the app
        initial_state = GraphState(messages=[HumanMessage(content=query)])
        final_state = app.invoke(initial_state)
        system_response = final_state["messages"][-1].content

        # Grade the response
        grade = evaluate_response(
          query=query, 
          ground_truth=ground_truth,
          response=system_response 
        )

        # Store metrics
        results.append({
          "query": query,
          "category": entry["category"],
          "response": system_response,
          "grade": grade.score,
          "ich": grade.ich_detected,
          'fch': grade.fch_detected,
          "fn": grade.fn_detected,
          "justification": grade.justification
        })

    # 5. Save the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"eval_report_{timestamp}.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"--- Eval Complete. Report saved to {report_path} ---")

if __name__ == "__main__":
    run_batch_evaluation("data/eval/eval_set.json")