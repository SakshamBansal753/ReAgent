from __future__ import annotations

from typing import Literal

from app.graph.state import GraphState


class GraphRouter:
    """
    Routing logic for the AI Research Agent.
    """

    def route_after_evaluation(
        self,
        state: GraphState,
    ) -> Literal["generate", "rewrite"]:

        if state.context_sufficient:
            return "generate"

        return "rewrite"

    def route_after_rewrite(
        self,
        state: GraphState,
    ) -> Literal["retrieve", "generate"]:

        if state.retry_count >= state.max_retries:
            return "generate"

        return "retrieve"