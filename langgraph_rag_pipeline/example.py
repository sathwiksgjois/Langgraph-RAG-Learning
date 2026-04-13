from langgraph.graph import StateGraph
from typing import TypedDict
from 03_rag_pipeline_module.rag_pipeline import RAGPipeline

class State(TypedDict):
    query: str
    context: str
    answer: str

documents = [
    "LangGraph is used for building AI workflows.",
    "RAG combines retrieval with generation.",
    "FAISS is used for similarity search."
]

rag = RAGPipeline(documents)

def retrieve(state: State):
    state["context"] = rag.retrieve(state["query"])
    return state

def generate(state: State):
    state["answer"] = rag.generate(state["query"], state["context"])
    return state

graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.set_finish_point("generate")

app = graph.compile()

result = app.invoke({"query": "Explain RAG"})

print("\n🧠 Answer:\n", result["answer"])