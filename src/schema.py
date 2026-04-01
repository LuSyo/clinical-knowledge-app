from typing import List, Annotated, Optional, Literal, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(BaseModel):
    # Random seed
    seed: int = Field(default=4)

    # conversation history
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

    # Config: use Knowledge Graph retrieval or not
    use_kg: bool = Field(default=True, description="Toggle for KG-augmented retrieval logic")
    
    # Context retrieved from the vector store
    context: List[str] = Field(default_factory=list)
    
    # evaluation metrics
    relevance_score: Optional[int] = None
    hallucination_flag: bool = False
    
    # Helper to print the state clearly during debugging
    def __repr__(self):
        return f"GraphState(messages_count={len(self.messages)}, context_len={len(self.context)})"
    
class ClinicalFact(BaseModel):
    """Clinical knowledge (e.g. 'Amlodipine is a Calcium Channel Blocker')"""
    type: Literal["fact"] = "fact"
    subject: str = Field(description="The clinical entity that the relationship starts from")
    predicate: str = Field(description="The relationship (e.g., 'is_used_for', 'causes', 'contraindicated_with')")
    object: str = Field(description="The clinical entity that the relationship points to")

class ClinicalLogic(BaseModel):
    """Conditional clinical instruction (e.g., 'If pulse is irregular, perform ECG')"""
    type: Literal["logic"] = "logic"
    condition: str = Field(description="The clinical trigger (the 'If')")
    action: str = Field(description="The required action (the 'Then'), e.g. 'perform'")
    object: str = Field(description="The object of the action")

class Triplet(BaseModel):
    """Structured knowledge triplet"""
    item: Union[ClinicalFact, ClinicalLogic]

class TripletExtraction(BaseModel):
    """The collection of all extracted structured knowledge triplets from a document."""
    triplets: List[Triplet] = Field(description="A list of relational triplets (clinical facts and conditional clinical logic) found in the text")
