import os
from dotenv import load_dotenv

from utils import setup_logger, parse_args, Config 
from langchain_core.messages import HumanMessage
from pipeline.graph import build_graph
from schema import GraphState

def main():
  load_dotenv()

  args = parse_args()

  logger = setup_logger(log_dir=Config.LOG_DIR, exp_name=args.exp_name)

  if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found.")
  
  app = build_graph()
  # print(app.get_graph().draw_ascii())
  
  # initial state for the graph
  initial_state = GraphState(
    messages=[HumanMessage(content=args.query)]
  )

  # invoke the graph
  logger.info(f"QUERY: {args.query}")
  final_state = app.invoke(initial_state)

  answer = final_state["messages"][-1].content
  logger.info(f"AI RESPONSE: {answer}")
  
  logger.info(f"\nRetrieved Context Chunks: {len(final_state['context'])}")

if __name__ == "__main__":
    main()