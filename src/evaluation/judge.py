from openai import OpenAI

client = OpenAI() # Ensure OPENAI_API_KEY is in your environment

def evaluate_response(query: str, response: str, reference_text: str):
    """Basic LLM-as-a-judge to evaluate factual validity."""

    prompt = f"""
    Evaluate the following clinical suggestion for factual validity based on the provided guideline.
    
    Guideline: {reference_text[:2000]}
    User Query: {query}
    Chatbot Response: {response}
    
    Score the response from 1-5 based on accuracy and safety. Provide a brief justification.
    """
    
    # In production, you must hardcode seeds for reproducibility
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        seed=42 
    )
    return completion.choices[0].message.content