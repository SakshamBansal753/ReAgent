from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document


class DocumentFilter:
    """
    Filters retrieved documents using metadata.
    """

    @staticmethod
    def by_year(
        documents: List[Document],
        year: int,
    ) -> List[Document]:

        filtered = []

        for doc in documents:

            published = doc.metadata.get("published")

            if published and str(year) in str(published):
                filtered.append(doc)

        return filtered

    @staticmethod
    def by_author(
        documents: List[Document],
        author: str,
    ) -> List[Document]:

        author = author.lower()

        filtered = []

        for doc in documents:

            authors = doc.metadata.get("authors", [])

            if any(author in a.lower() for a in authors):
                filtered.append(doc)

        return filtered

    @staticmethod
    def by_title(
        documents: List[Document],
        keyword: str,
    ) -> List[Document]:

        keyword = keyword.lower()

        return [
            doc
            for doc in documents
            if keyword in doc.metadata.get(
                "title",
                "",
            ).lower()
        ]

    @staticmethod
    def by_source(
        documents: List[Document],
        source: str,
    ) -> List[Document]:

        source = source.lower()

        return [
            doc
            for doc in documents
            if source in doc.metadata.get(
                "source",
                "",
            ).lower()
        ]

    @staticmethod
    def limit(
        documents: List[Document],
        limit: int,
    ) -> List[Document]:

        return documents[:limit]