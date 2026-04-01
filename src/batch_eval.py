import json
import os
from dotenv import load_dotenv
from datetime import datetime
from utils import setup_logger, parse_args, Config
from pipeline.graph import build_graph
from schema import GraphState
from langchain_core.messages import HumanMessage
from evaluation.judge import evaluate_response

def run_batch_evaluation():
    load_dotenv()
    
    args = parse_args()

    logger = setup_logger(log_dir=Config.LOG_DIR, exp_name=args.exp_name)

    if not os.getenv("OPENAI_API_KEY"):
      raise ValueError("OPENAI_API_KEY not found.")
    
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    app = build_graph()
    
    dataset_path = os.path.join(Config.EVAL_DIR, args.eval_dataset)
    with open(dataset_path, 'r') as f:
        eval_set = json.load(f)

    results = []
    logger.info(f"--- Starting Eval on {len(eval_set)} cases ---")

    for entry in eval_set:
        query = entry["query"]
        ground_truth = entry["ground_truth"]
         
        logger.info(f"Testing: {query[:50]}...")

        # Run the app
        initial_state = GraphState(
            messages=[HumanMessage(content=query)],
            use_kg=args.use_kg,
            seed=args.seed
            )
        final_state = app.invoke(initial_state)
        system_response = final_state["messages"][-1].content

        # Grade the response
        grade = evaluate_response(
          query=query, 
          ground_truth=ground_truth,
          response=system_response,
          seed=args.seed
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
    report_path = os.path.join(Config.RESULTS_DIR, f"{args.exp_name}.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"--- Eval Complete. Report saved to {report_path} ---")

if __name__ == "__main__":
    run_batch_evaluation()