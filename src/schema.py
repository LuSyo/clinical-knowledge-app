from typing import TypedDict, List, Annotated
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
  sources: List[str]
  messages: Annotated[list, add_messages]
  evaluation_score: int
  evaluation_feedback: str
  hallucination_flag: bool