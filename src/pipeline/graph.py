from typing import cast
from langgraph.graph import StateGraph, END, START
from pipeline.nodes import retrieve, generate, grade_documents, handle_rejection
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from schema import GraphState

def build_graph():
  # initialise the graph with the schema
  workflow = StateGraph(GraphState)

  # add nodes
  workflow.add_node("retrieve", retrieve)
  workflow.add_node("generate", generate)
  workflow.add_node("grade_documents", grade_documents)
  workflow.add_node("reject", handle_rejection)

  # add START routing logic
  workflow.add_conditional_edges(
    START,
    route_question,
    {
      "vectorstore": "retrieve",
      "direct_response": "generate"
    }
  )

  # add edges
  workflow.add_edge("retrieve", "grade_documents")
  
  # add generate/STOP routing after grading
  workflow.add_conditional_edges(
      'grade_documents',
      generate_or_stop,
      {
          "generate": "generate",
          "reject": "reject"
      }
  )

  workflow.add_edge("reject", END)
  workflow.add_edge("generate", END)

  return workflow.compile()

# --- ROUTING FUCNTIONS ---

class RouteQuery(BaseModel):
    """Route a user query to the most appropriate datasource."""
    datasource: str = Field(
        description="Given a user question choose to route it to 'vectorstore' or 'direct_response'."
    )

def generate_or_stop(state: GraphState):
    """
    Determines whether to generate an answer or skip based on document relevance.
    """
    print("--- DECIDING TO GENERATE ---")
    
    if state.relevance_score == 1:
        # Documents were relevant, proceed to generation
        return "generate"
    else:
        # Stop
        return "reject"
    
def route_question(state: GraphState):
    """
    Route the question to determine if it is clinical or general.
    """
    print("--- ROUTING REPONSE ---")
    query = state.messages[-1].content

    # Initialise a structured LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)

    # Ask the router to categorize the query
    system_prompt = "You are an expert at routing user queries to a vectorstore or direct response. If the question is about medical guidelines or clinical advice, use vectorstore."
    
    result = cast(RouteQuery, structured_llm.invoke([
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": query}
    ]))

    print(result.datasource)

    return result.datasource