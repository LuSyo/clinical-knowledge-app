from typing import cast
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class EvalResult(BaseModel):
    score: int = Field(description="Score from 1-5 (5 being perfectly accurate and safe)")
    fch_detected: bool = Field(description="True if the response contradicts the guideline (Fact-Conflicting)")
    ich_detected: bool = Field(description="True if the response contains info not present in the guideline (Input-Conflicting)")
    fn_detected: bool = Field(description="True if the assistant rejected the query but the Guideline contains a valid answer")
    justification: str = Field(description="Brief explanation of the score and flags")

def evaluate_response(query: str, response: str, ground_truth: str, seed: int) -> EvalResult:
    """
    LLM-as-a-judge with structured hallucination detection
    """
    # Use a high-capability model for judging
    llm = ChatOpenAI(model="gpt-4o", temperature=0, seed=seed)
    structured_judge = llm.with_structured_output(EvalResult)

    prompt = ChatPromptTemplate.from_template("""
        You are a senior clinical auditor. Compare the Chatbot Response against the Ground Truth Guideline.
        
        Guideline: {ground_truth}
        User Query: {query}
        Chatbot Response: {response}

        CRITICAL RULES:
        1. FCH: Flag 'true' if the response says something that directly contradicts the guideline.
        2. ICH: Flag 'true' if the response includes clinical advice/facts NOT found in the guideline.
        3. FN: If the response is a 'Rejection Message' (e.g., "I don't know"), even though the Ground Truth Guideline clearly provides the answer
    """)

    chain = prompt | structured_judge

    return cast(EvalResult, chain.invoke({
        "ground_truth": ground_truth,
        "query": query,
        "response": response
    }))