from typing import List, Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(BaseModel):
    # conversation history
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    
    # Context retrieved from the vector store
    context: List[str] = Field(default_factory=list)
    
    # evaluation metrics
    relevance_score: Optional[int] = None
    hallucination_flag: bool = False
    
    # Helper to print the state clearly during debugging
    def __repr__(self):
        return f"GraphState(messages_count={len(self.messages)}, context_len={len(self.context)})"
    
class Triplet(BaseModel):
    """Refers to a single relationship extracted from the text."""
    subject: str = Field(description="The entity that the relationship starts from")
    predicate: str = Field(description="The relationship type (e.g., 'is_used_for', 'causes', 'contraindicated_with')")
    object: str = Field(description="The entity that the relationship points to")

class TripletExtraction(BaseModel):
    """The collection of all extracted triplets from a document."""
    triplets: List[Triplet] = Field(description="A list of relational triplets found in the text.")