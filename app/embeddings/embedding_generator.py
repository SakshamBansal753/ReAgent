from typing import List
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """
    Generate embedding  for langchain document  
    """
    def __init__(self,model_name: str = "BAAI/bge-base-en-v1.5",
        normalize_embeddings: bool = True,):
        self.model=SentenceTransformer(model_name)
        self.normalize_embeddings=normalize_embeddings
    def embed_doc(self,documents:List[Document])->np.ndarray:
        texts=[doc.page_content for doc in documents]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        return embeddings
    def embed_query(self,query:str)->np.ndarray:
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embedding
        