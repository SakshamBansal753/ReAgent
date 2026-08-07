from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.graph.state import GraphState
from app.graph.nodes import GraphNodes
from app.graph.routes import GraphRouter


class ResearchGraph:

    def __init__(
        self,
        nodes: GraphNodes,
        router: GraphRouter,
    ):
        self.nodes = nodes
        self.router = router

    def build(self):

        workflow = StateGraph(GraphState)

        # ----------------------------
        # Nodes
        # ----------------------------

        workflow.add_node(
            "retrieve",
            self.nodes.retrieve,
        )

        workflow.add_node(
            "evaluate",
            self.nodes.evaluate,
        )

        workflow.add_node(
            "rewrite",
            self.nodes.rewrite,
        )

        workflow.add_node(
            "generate",
            self.nodes.generate,
        )

        # ----------------------------
        # Edges
        # ----------------------------

        workflow.add_edge(
            START,
            "retrieve",
        )

        workflow.add_edge(
            "retrieve",
            "evaluate",
        )

        workflow.add_conditional_edges(
            "evaluate",
            self.router.route_after_evaluation,
            {
                "generate": "generate",
                "rewrite": "rewrite",
            },
        )

        workflow.add_conditional_edges(
            "rewrite",
            self.router.route_after_rewrite,
            {
                "retrieve": "retrieve",
                "generate": "generate",
            },
        )

        workflow.add_edge(
            "generate",
            END,
        )

        return workflow.compile()