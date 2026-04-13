from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

docs = [
    "LangGraph is a framework for LLM workflows.",
    "RAG improves LLM by adding external knowledge.",
    "FAISS is used for similarity search."
]

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.create_documents(docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

query = "What is RAG?"

retrieved_docs = db.similarity_search(query, k=2)
context = "\n".join([d.page_content for d in retrieved_docs])

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = f"""
Use this context to answer:

{context}

Question: {query}
"""

response = llm.invoke(prompt)
print(response.content)