from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
class QueryExpander:
    def __init__(self,llm:BaseChatModel,num_queries:int=4):
        self.llm=llm
        self.num_queries=num_queries
        self.prompt=ChatPromptTemplate.from_messages(
            [(
                "system",(
                    "You are an expert AI research assistant.\n"
                        "Generate alternative search queries that preserve "
                        "the original meaning.\n"
                        f"Generate exactly {num_queries} queries.\n"
                        "Return ONLY one query per line.\n"
                        "Do not number them."
                ),
            ),
            (
                "human",
                "{query}"
            )]
        )
        self.chain=self.prompt|self.llm|StrOutputParser()
    def expand(self,query:str)->List[str]:
            response=self.chain.invoke(
            {"query":query})
            queries=[
                q.strip()
                for q in response.split("\n")
                if q.strip()
            ]
            queries.insert(0,query)
            return list(dict.fromkeys(queries))