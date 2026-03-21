import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from pipeline.graph import build_graph
from schema import GraphState

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
      raise ValueError("OPENAI_API_KEY not found.")
    
    app = build_graph()
    print(app.get_graph().draw_ascii())
    
    # test query
    query = "What are the surgical steps for a bilateral lung transplant?"
    
    # initial state for the graph
    initial_state = GraphState(
      messages=[HumanMessage(content=query)]
    )

    # invoke the graph
    final_state = app.invoke(initial_state)

    answer = final_state["messages"][-1].content
    print(f"\nAI Response:\n{answer}")
    
    print(f"\nRetrieved Context Chunks: {len(final_state['context'])}")

if __name__ == "__main__":
    main()