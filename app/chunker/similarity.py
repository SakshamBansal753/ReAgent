import numpy as np


class SimilarityCalculator:

    @staticmethod
    def cosine(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        """

        return float(
            np.dot(vec1, vec2)
            / (
                np.linalg.norm(vec1)
                * np.linalg.norm(vec2)
            )
        )