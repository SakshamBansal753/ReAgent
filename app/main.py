from dotenv import load_dotenv

from langchain_groq import ChatGroq
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
import os
load_dotenv()


def build_agent():



    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )


    embedding_generator = EmbeddingGenerator()



    vector_store = ChromaStore(
        embedding_function=embedding_generator,
        persist_directory="./chroma_db"
    )



    query_expander = QueryExpander(llm)

    retriever = Retriever(
        vector_store=vector_store,
    )

    mmr = MMRRetriever(
        embedding_generator=embedding_generator,
    )

    reranker = Reranker()

    advanced_retriever = AdvancedRetriever(
        query_expander=query_expander,
        retriever=retriever,
        mmr_retriever=mmr,
        reranker=reranker,
    )



    nodes = GraphNodes(
        llm=llm,
        retriever=advanced_retriever,
    )

    router = GraphRouter()

    graph = ResearchGraph(
        nodes=nodes,
        router=router,
    ).build()

    return graph


def main():

    graph = build_agent()

    while True:

        query = input("\nAsk: ")

        if query.lower() in {"exit", "quit"}:
            break

        state = GraphState(
            question=query,
        )

        result = graph.invoke(state)

        print("\n")
        print("=" * 80)
        print(result['answer'])
        print("=" * 80)


if __name__ == "__main__":
    main()