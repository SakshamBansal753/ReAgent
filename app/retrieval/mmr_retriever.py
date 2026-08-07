from __future__ import annotations
from typing import List
import numpy as np
from langchain_core.documents import Document

from app.embeddings.embedding_generator import EmbeddingGenerator

class MMRRetriever:
    def __init__(self,embedding_generator:EmbeddingGenerator,lambda_mult:float=0.7):
        self.embedding_generator=embedding_generator
        self.lambda_mult=lambda_mult
    def retrieve(self,query:str,candidate_docs:List[Document],k:int=5)->List[Document]:
        if not candidate_docs:
            return []

        if len(candidate_docs) <= k:
            return candidate_docs

        query_embedding = self.embedding_generator.embed_query(query)

        doc_embeddings = self.embedding_generator.embed_documents(
            candidate_docs
        )

        selected = []

        remaining = list(range(len(candidate_docs)))

        relevance_scores = np.dot(doc_embeddings, query_embedding)

        while remaining and len(selected) < k:

            if not selected:

                best = max(
                    remaining,
                    key=lambda i: relevance_scores[i]
                )

                selected.append(best)
                remaining.remove(best)

                continue

            mmr_scores = []

            for idx in remaining:

                redundancy = max(
                    np.dot(
                        doc_embeddings[idx],
                        doc_embeddings[selected_doc],
                    )
                    for selected_doc in selected
                )

                score = (
                    self.lambda_mult * relevance_scores[idx]
                    - (1 - self.lambda_mult) * redundancy
                )

                mmr_scores.append((idx, score))

            best = max(
                mmr_scores,
                key=lambda x: x[1]
            )[0]

            selected.append(best)
            remaining.remove(best)

        return [
            candidate_docs[i]
            for i in selected
        ]