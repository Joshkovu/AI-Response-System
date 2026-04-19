from __future__ import annotations

import math
from typing import Callable


class VectorIndex:
    """
    In-memory vector store with cosine or Euclidean distance search.

    This class keeps vectors and associated document payloads in lists.
    It supports query-time embedding via an injected embedding function.
    """

    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        metric = distance_metric.lower().strip()
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")

        self.distance_metric = metric
        self.embedding_fn = embedding_fn
        self.vectors: list[list[float]] = []
        self.documents: list[dict] = []

    def add_vector(self, vector: list[float], document: dict) -> None:
        """Add one vector and its associated document metadata."""
        if not vector:
            raise ValueError("vector must not be empty")
        if not isinstance(document, dict):
            raise TypeError("document must be a dict")

        if self.vectors and len(vector) != len(self.vectors[0]):
            raise ValueError("all vectors must have the same dimensionality")

        self.vectors.append(vector)
        self.documents.append(document)

    def search(self, query: str, k: int = 3) -> list[tuple[dict, float]]:
        """
        Search top-k closest documents to a query string.

        Returns:
            List of (document, distance) sorted ascending by distance.
        """
        if self.embedding_fn is None:
            raise ValueError("embedding_fn is required for string-query search")
        if k <= 0 or not self.vectors:
            return []

        query_vector = self.embedding_fn(query)
        if len(query_vector) != len(self.vectors[0]):
            raise ValueError("query vector dimensionality does not match index")

        scored_results: list[tuple[dict, float]] = []
        for vector, document in zip(self.vectors, self.documents):
            distance = self._distance(query_vector, vector)
            scored_results.append((document, distance))

        scored_results.sort(key=lambda item: item[1])
        return scored_results[: min(k, len(scored_results))]

    def _distance(self, a: list[float], b: list[float]) -> float:
        if self.distance_metric == "cosine":
            return self._cosine_distance(a, b)
        return self._euclidean_distance(a, b)

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            # Maximum distance when one vector is all zeros.
            return 1.0
        cosine_similarity = dot / (norm_a * norm_b)
        return 1.0 - cosine_similarity

    @staticmethod
    def _euclidean_distance(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
