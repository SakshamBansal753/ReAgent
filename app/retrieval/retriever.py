from typing import List

from langchain_core.documents import Document
from app.vectorstore.base_store import BaseVectorStore


class Retriever:

    def __init__(
        self,
        vector_store: BaseVectorStore,
    ):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:

        return self.vector_store.similarity_search(
            query=query,
            k=k,
        )