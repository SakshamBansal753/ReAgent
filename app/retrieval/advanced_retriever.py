from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.retrieval.query_expander import QueryExpander
from app.retrieval.retriever import Retriever
from app.retrieval.mmr_retriever import MMRRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.filters import DocumentFilter


class AdvancedRetriever:
    """
    Production retrieval pipeline.

    Query
      ↓
    Query Expansion
      ↓
    Similarity Search
      ↓
    Deduplication
      ↓
    MMR
      ↓
    Reranking
      ↓
    Metadata Filters (optional)
      ↓
    Final Documents
    """

    def __init__(
        self,
        query_expander: QueryExpander,
        retriever: Retriever,
        mmr_retriever: MMRRetriever,
        reranker: Reranker,
    ):
        self.query_expander = query_expander
        self.retriever = retriever
        self.mmr_retriever = mmr_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        candidate_k: int = 20,
        mmr_k: int = 10,
        final_k: int = 5,
    ) -> List[Document]:

        expanded_queries = self.query_expander.expand(query)

        candidate_docs = []

        for q in expanded_queries:
            candidate_docs.extend(
                self.retriever.retrieve(
                    query=q,
                    k=candidate_k,
                )
            )

        # Remove duplicates
        unique_docs = []
        seen = set()

        for doc in candidate_docs:

            key = (
                doc.page_content,
                tuple(sorted(doc.metadata.items()))
            )

            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        # MMR
        mmr_docs = self.mmr_retriever.retrieve(
            query=query,
            candidate_docs=unique_docs,
            k=mmr_k,
        )

        # Rerank
        reranked_docs = self.reranker.rerank(
            query=query,
            documents=mmr_docs,
            top_k=final_k,
        )

        return reranked_docs