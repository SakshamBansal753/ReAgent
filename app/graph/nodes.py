from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from app.graph.prompts import (
    GENERATE_PROMPT,
    REWRITE_PROMPT,
    EVALUATE_PROMPT,)
from app.graph.state import GraphState
from app.retrieval.advanced_retriever import AdvancedRetriever
class GraphNodes:
    def __init__(self,llm:BaseChatModel,retriever:AdvancedRetriever):
        self.retriever=retriever
        self.generate_chain=(
            GENERATE_PROMPT|llm|StrOutputParser()
        )
        self.rewrite_chain=(
            REWRITE_PROMPT|llm|StrOutputParser()
        )
        self.evaluvate_chain=(
            EVALUATE_PROMPT|llm|StrOutputParser()
        )
    def retrieve(self,state:GraphState)->GraphState:
        query=(
            state.rewritten_query if state.rewritten_query else state.question
        )
        documents=self.retriever.retrieve(query=query)
        state.documents=documents
        return state
    def evaluate(self,state:GraphState)->GraphState:
        context="\n\n".join(doc.page_content for doc in state.documents)
        result=self.evaluvate_chain.invoke({
            "question":state.question,
            "context":context
        })
        state.context_sufficient=(
            result.strip().upper()=="YES"
        )
        return state
    def rewrite(
        self,
        state: GraphState,
    ) -> GraphState:

        rewritten_query = self.rewrite_chain.invoke(
            {
                "question": state.question,
            }
        )

        state.rewritten_query = rewritten_query.strip()

        state.retry_count += 1

        return state



    def generate(
        self,
        state: GraphState,
    ) -> GraphState:

        context = "\n\n".join(
            doc.page_content
            for doc in state.documents
        )

        answer = self.generate_chain.invoke(
            {
                "question": state.question,
                "context": context,
            }
        )
        state.answer = answer

        state.messages.append(
            HumanMessage(content=state.question)
        )

        state.messages.append(
            AIMessage(content=answer)
        )

        return state


