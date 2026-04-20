import os
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from vector_index import VectorIndex 


class LocalEmbedding:

    def __init__(self, model_name: str = "gemini-embedding-001", distance_metric: str = "cosine"):
        """
        Initialise the Gemini embedding client and an empty vector index.

        Args:
            model_name:      Gemini embedding model ID to use.
            distance_metric: Distance metric for the index ('cosine' or 'euclidean').
        """
        load_dotenv()
        self.model_name = model_name or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")

        self._candidate_models = [self.model_name]
        for fallback_model in ("gemini-embedding-001", "text-embedding-004"):
            if fallback_model not in self._candidate_models:
                self._candidate_models.append(fallback_model)

        print(f"[LocalEmbedding] Initialising Gemini embedding model {self.model_name}...")
        self.client = genai.Client(api_key=api_key)
        print("[LocalEmbedding] Gemini embedding client ready.")

        self.store = VectorIndex(distance_metric=distance_metric, embedding_fn=self._embed_one,)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_embedding_values(self, embedding_response: object) -> list[float]:
        """Extract embedding values across compatible google-genai response shapes."""
        embedding = getattr(embedding_response, "embedding", None)
        if embedding is not None and hasattr(embedding, "values"):
            return list(embedding.values)

        embeddings = getattr(embedding_response, "embeddings", None)
        if embeddings and hasattr(embeddings[0], "values"):
            return list(embeddings[0].values)

        raise ValueError("Unexpected Gemini embedding response format")

    def _embed_one(self, text: str) -> list[float]:
        """
        Embed a single string and return it as a plain Python list.

        Used internally by VectorIndex to embed query strings at search time.

        Args:
            text: The string to embed.

        Returns:
            List of floats representing the text embedding.
        """
        response = self._embed_content_with_fallback(text)
        return self._extract_embedding_values(response)

    def _embed_content_with_fallback(self, text: str):
        """Try configured embedding model(s), switching automatically on 404 model errors."""
        last_error = None

        for candidate_model in self._candidate_models:
            try:
                response = self.client.models.embed_content(model=candidate_model, contents=text)
                if candidate_model != self.model_name:
                    print(f"[LocalEmbedding] Switching embedding model to {candidate_model}.")
                    self.model_name = candidate_model
                return response
            except genai_errors.ClientError as error:
                last_error = error
                if getattr(error, "status_code", None) == 404:
                    continue
                raise

        raise RuntimeError(
            "No supported embedding model was found for this API key/version. "
            "Set GEMINI_EMBEDDING_MODEL in your .env to a model returned by ListModels."
        ) from last_error

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings using Gemini embeddings.

        Args:
            texts: One or more strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for text in texts:
            response = self._embed_content_with_fallback(text)
            embeddings.append(self._extract_embedding_values(response))
        return embeddings

    def build_index(self, chunks: list[str]) -> None:
        """
        Embed all chunks and store them in the vector index.

        Args:
            chunks: List of text strings to index.
        """
        embeddings = self.get_embeddings(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            self.store.add_vector(embedding, {"content": chunk})
        print(f"[LocalEmbedding] Indexed {len(chunks)} chunks.")

    def search(self, question: str, k: int = 3) -> list[tuple[dict, float]]:
        """
        Find the k chunks most semantically similar to the question.

        The question is embedded automatically by VectorIndex using the
        _embed_one function provided at initialisation.

        Args:
            question: Natural-language question string.
            k:        Number of results to return (default 3).

        Returns:
            List of (document_dict, distance) tuples, sorted by
            ascending cosine distance (lower = more similar).
        """
        return self.store.search(question, k=k)

    def get_context(self, question: str, k: int = 3) -> str:
        """
        Retrieve the most relevant chunks for a question as a single string.

        Intended for RAG: pass the returned string directly as context
        to your LLM prompt alongside the user's question.

        Args:
            question: Natural-language question string.
            k:        Number of chunks to include (default 3).

        Returns:
            A single string with the k most relevant chunks
            joined by a separator, ready to embed in a prompt.
        """
        results = self.search(question, k=k)
        return "\n\n---\n\n".join(doc["content"] for doc, _ in results)
