from langgraph.graph import StateGraph, END
from schema import GraphState
from pipeline.nodes import retrieve, generate

def build_graph():
  # initialise the graph with the schema
  workflow = StateGraph(GraphState)

  # add nodes
  workflow.add_node("retrieve", retrieve)
  workflow.add_node("generate", generate)

  # add edges
  workflow.set_entry_point("retrieve")
  workflow.add_edge("retrieve", "generate")
  workflow.add_edge("generate", END)

  return workflow.compile()

# For testing purposes
# app = build_graph()
# print(app.get_graph().draw_ascii())