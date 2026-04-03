from typing import List, Annotated, Optional, Literal, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# OMOP-aligned Clinical Domains
ClinicalCategory = Literal[
    "Condition",    # OMOP: SNOMED CT (e.g., AFib)
    "Drug",         # OMOP: RxNorm/ATC (e.g., Apixaban) - replacing 'Medication'
    "Procedure",    # OMOP: SNOMED/CPT4 (e.g., ECG)
    "Device",       # OMOP: SNOMED (e.g., Event recorder)
    "Measurement",  # OMOP: LOINC (e.g., CHA2DS2-VASc score)
    "Observation"   # OMOP: SNOMED (e.g., Lifestyle factors, "Smoker")
]

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
    subject_domain: ClinicalCategory = Field(description="OMOP Domain of the subject")
    predicate: str = Field(description="The relationship (e.g., 'is_used_for', 'causes', 'contraindicated_with')")
    object: str = Field(description="The clinical entity that the relationship points to")
    object_domain: ClinicalCategory = Field(description="OMOP Domain of the object")

class LogicRequirement(BaseModel):
    """A single clinical prerequisite (event or requirement)"""
    entity: str = Field(description="The clinical entity")
    domain: ClinicalCategory = Field(description="OMOP Domain of the entity")
    negated: bool = Field(
        default=False, 
        description="Set to True if the entity must NOT be present (e.g., 'No history of...')"
    )
    description: Optional[str] = Field(description="The state or threshold (e.g., '> 2', 'concurrent use'). Leave empty if the entity presence itself is the trigger.")

class ClinicalLogic(BaseModel):
    """Conditional clinical instruction (e.g., 'If pulse is irregular, perform ECG')"""
    type: Literal["logic"] = "logic"
    triggers: List[LogicRequirement] = Field(description="List of events/requirements that must be met")
    logic_gate: Literal["AND", "OR"] = Field(default="AND", description="How to combine the triggers")
    action: str = Field(description="The 'THEN' action/recommendation")

class Triplet(BaseModel):
    """Structured knowledge triplet"""
    item: Union[ClinicalFact, ClinicalLogic]

class TripletExtraction(BaseModel):
    """The collection of all extracted structured knowledge triplets from a document."""
    triplets: List[Triplet] = Field(description="A list of relational triplets (clinical facts and conditional clinical logic) found in the text")

