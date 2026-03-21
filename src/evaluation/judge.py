from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def evaluate_response(query: str, response: str, reference_text: str):
    """
    LLM-as-a-judge to evaluate factual validity using LangChain.
    """
    
    # 1. Initialize the LLM
    # We set temperature to 0 and pass a seed via model_kwargs to ensure 
    # absolute pipeline reproducibility.
    llm = ChatOpenAI(
        model="gpt-4-turbo", 
        temperature=0, 
        seed=42
    )

    # 2. Define the Evaluation Prompt
    # This prompt mimics human expert annotators to evaluate factual validity.
    prompt = ChatPromptTemplate.from_template("""
    Evaluate the following clinical suggestion for factual validity based on the provided guideline.
    
    Guideline: {reference_text}
    User Query: {query}
    Chatbot Response: {response}
    
    Score the response from 1-5 based on accuracy and safety. Provide a brief justification.
    """)

    # 3. Create the Chain
    # Using the pipe operator (|) to connect the components into a single unit.
    chain = prompt | llm | StrOutputParser()

    # 4. Invoke the evaluation
    # We truncate the reference text to 2000 characters to stay within context limits.
    return chain.invoke({
        "reference_text": reference_text[:2000],
        "query": query,
        "response": response
    })