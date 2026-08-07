from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from app.graph.graph import ResearchGraph
from app.graph.nodes import GraphNodes
from app.graph.routes import GraphRouter
from app.graph.state import GraphState

from app.retrieval.query_expander import QueryExpander
from app.retrieval.retriever import Retriever
from app.retrieval.mmr_retriever import MMRRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.advanced_retriever import AdvancedRetriever

from app.embeddings.embedding_generator import EmbeddingGenerator
from app.vectorstore.chroma_store import ChromaStore


load_dotenv()


def main():

    print("\n========== Initializing ==========")

    # LLM for query expansion
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
    )

    # Vector store
    vector_store = ChromaStore(
        persist_directory="./chroma_db",
        embedding_function=EmbeddingGenerator,
    )


    print("\n========== Direct Chroma Test ==========")

    docs = vector_store.similarity_search(
        query="Attention Is All You Need",
        k=5
    )

    print("Documents found:", len(docs))


    for i, doc in enumerate(docs):

        print("\n-------------------------")
        print("Document:", i+1)

        print("Metadata:")
        print(doc.metadata)

        print("\nContent:")
        print(doc.page_content[:500])


    print("\n========== Retriever Test ==========")


    query_expander = QueryExpander(
        llm=llm
    )


    retriever = AdvancedRetriever(
        vector_store=vector_store,
        query_expander=query_expander
    )


    results = retriever.retrieve(
        query="Summarize Attention Is All You Need paper",
        k=5
    )


    print("\nRetriever returned:", len(results))


    for i, doc in enumerate(results):

        print("\n====================")
        print("Chunk:", i+1)

        print(doc.metadata)

        print(doc.page_content[:300])


if __name__ == "__main__":
    main()