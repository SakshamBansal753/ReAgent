from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict


class GraphState(BaseModel):
    """
    Shared state for the AI Research Agent.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    # User's original question
    question: str

    # Conversation history
    messages: List[BaseMessage] = Field(default_factory=list)

    # Retrieved documents
    documents: List[Document] = Field(default_factory=list)

    # Final generated answer
    answer: Optional[str] = None

    # Query after rewriting
    rewritten_query: Optional[str] = None

    # Number of retrieval attempts
    retry_count: int = 0

    # Maximum allowed retries
    max_retries: int = 2

    # Whether retrieved context is sufficient
    context_sufficient: bool = False