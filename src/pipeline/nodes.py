from typing import cast
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from schema import GraphState

def retrieve(state: GraphState):
    """
    Retrieve documents from the vector store based on the last message.
    """
    print("--- RETRIEVING CONTEXT ---")

    # latest user query from the message history
    messages = state.messages
    last_query = str(messages[-1].content)

    # initialize the vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)

    # 3. Perform the search
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(last_query)

    # 4. Format for the state
    context_text = [doc.page_content for doc in docs]

    # Return the updated context field in the state
    return {"context": context_text}

def generate(state: GraphState):
    """
    Generate an answer using the retrieved context and the query history.
    """
    print("--- GENERATING RESPONSE ---")
    
    # set up llm
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=state.seed)

    # clinical prompt template
    template = """
    You are a clinical decision support assistant. 
    Use the following pieces of context from clinical guidelines to answer the question. 
    If you don't know the answer based on the context, say that you don't know. 
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # prepare inputs
    last_query = state.messages[-1].content
    context_str = "\n\n".join(state.context)

    # 4. Run the chain
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_str, "question": last_query})

    # 5. Return the response as a new message to be added to the state
    return {"messages": [("assistant", response)]}

# ----- CRAG (Corrective RAG) ------

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    print("--- CHECKING DOCUMENT RELEVANCE ---")
    question = state.messages[-1].content
    docs = state.context

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=state.seed)
    structured_llm = llm.with_structured_output(GradeDocuments)

    system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keywords or semantic meaning related to the user question, grade it as relevant. 
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

    # grade the combined context
    combined_docs = "\n\n".join(docs)
    score = cast(GradeDocuments, structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User question: {question} \n\nRetrieved docs: {combined_docs}"}
    ]))

    print(f'Are documents relevant? {score.binary_score}')

    return {"relevance_score": 1 if score.binary_score == "yes" else 0}

def handle_rejection(state: GraphState):
    print("--- HANDLING REJECTION ---")
    return {
        "messages": [("assistant", "I found some clinical information, but it doesn't specifically address your question in the context of the available NICE guidelines. I cannot provide a safe recommendation.")]
    }