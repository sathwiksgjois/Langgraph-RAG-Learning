from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class RAGPipeline:
    def __init__(self, documents):
        self.documents = documents
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        self._build_index()

    def _build_index(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        docs = splitter.create_documents(self.documents)
        self.vector_db = FAISS.from_documents(docs, self.embeddings)

    def retrieve(self, query, k=2):
        docs = self.vector_db.similarity_search(query, k=k)
        return "\n".join([d.page_content for d in docs])

    def generate(self, query, context):
        prompt = f"""
        Use ONLY the context:

        {context}

        Question: {query}
        """
        response = self.llm.invoke(prompt)
        return response.content

    def query(self, query):
        context = self.retrieve(query)
        return self.generate(query, context)