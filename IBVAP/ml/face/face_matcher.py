from typing import Optional

import numpy as np


class FaceMatcher:
    """
    IBVAP ArcFace embedding matcher.

    Compares an input face embedding against
    enrolled face embeddings using cosine similarity.

    Identity is returned only when the similarity
    exceeds the configured threshold.
    """

    def __init__(
        self,
        threshold: float = 0.45,
    ):

        self.threshold = threshold

        self.embeddings = {}
        self.names = {}

    def add_identity(
        self,
        identity_id: str,
        name: str,
        embedding: np.ndarray,
    ):
        """
        Add an enrolled identity.
        """

        normalized = self._normalize(
            embedding
        )

        self.embeddings[
            identity_id
        ] = normalized

        self.names[
            identity_id
        ] = name

    def load_identity(
        self,
        identity_id: str,
        name: str,
        embedding,
    ):
        """
        Load an identity from persistent storage.
        """

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        self.add_identity(
            identity_id=identity_id,
            name=name,
            embedding=embedding,
        )

    def _normalize(
        self,
        embedding: np.ndarray,
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            return embedding

        return embedding / norm

    def similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity.
        """

        a = self._normalize(
            embedding_a
        )

        b = self._normalize(
            embedding_b
        )

        return float(
            np.dot(a, b)
        )

    def match(
        self,
        embedding: np.ndarray,
    ) -> Optional[dict]:
        """
        Match an embedding against all enrolled
        identities.

        Returns the best candidate and whether
        the similarity passed the threshold.
        """

        if not self.embeddings:
            return None

        best_identity = None
        best_similarity = -1.0

        for (
            identity_id,
            stored_embedding,
        ) in self.embeddings.items():

            score = self.similarity(
                embedding,
                stored_embedding,
            )

            if score > best_similarity:

                best_similarity = score
                best_identity = identity_id

        if best_identity is None:
            return None

        matched = (
            best_similarity
            >= self.threshold
        )

        return {
            "identity_id": (
                best_identity
                if matched
                else None
            ),
            "name": (
                self.names[best_identity]
                if matched
                else None
            ),
            "similarity": best_similarity,
            "matched": matched,
        }