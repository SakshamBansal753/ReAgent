from sentence_transformers import SentenceTransformer
import numpy as np


class SentenceEmbedding:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences: list[str]) -> np.ndarray:
        """
        Convert sentences into embedding vectors.
        """
        return self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )