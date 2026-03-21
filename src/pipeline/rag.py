import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

def get_clinical_answer(query: str, db_path: str = "./data/chroma_db"):
    # load_dotenv()
    # api_key = os.getenv("OPENAI_API_KEY")

    # load vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # define prompt template
    template = """
    You are a clinical decision support assistant. 
    Use the following pieces of context from clinical guidelines to answer the question. 
    If you don't know the answer based on the context, say that you don't know. 
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # llm
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # rag chain
    # rag_chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    rag_chain = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    }).assign(answer=prompt | llm | StrOutputParser())

    return rag_chain.invoke(query)