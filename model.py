import os
from typing import Iterator

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from local_embedding import LocalEmbedding
from pdf_reader import PdfReader


class AiModel:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini client for answer generation.

        Args:
            model_name: Gemini text generation model ID.
        """
        load_dotenv()
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")

        print("Running checks to make sure everything is good...")
        self.gemini_auth(api_key)
        self._candidate_models = self._build_candidate_models(self.model_name)
        print("Model client ready.")

    def gemini_auth(self, api_key: str) -> None:
        """Create a Gemini client using the API key from environment."""
        self.client = genai.Client(api_key=api_key)

    def _normalize_model_name(self, model_name: str) -> str:
        if model_name.startswith("models/"):
            return model_name.split("/", 1)[1]
        return model_name

    def _list_generate_models(self) -> list[str]:
        """Discover models that support generateContent for this API key."""
        discovered: list[str] = []
        try:
            for model in self.client.models.list():
                actions = getattr(model, "supported_actions", []) or []
                if "generateContent" in actions:
                    name = self._normalize_model_name(getattr(model, "name", ""))
                    if name:
                        discovered.append(name)
        except Exception:
            # Non-fatal: hardcoded fallback list still allows operation.
            return []
        return discovered

    def _build_candidate_models(self, preferred_model: str) -> list[str]:
        candidates: list[str] = []

        for name in (
            preferred_model,
            os.getenv("GEMINI_MODEL", ""),
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-2.5-pro",
        ):
            normalized = self._normalize_model_name(name) if name else ""
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        for discovered in self._list_generate_models():
            if discovered not in candidates:
                candidates.append(discovered)

        return candidates

    def _is_retryable_model_error(self, error: Exception) -> bool:
        if isinstance(error, genai_errors.ClientError):
            return getattr(error, "status_code", None) in (404, 429)
        if isinstance(error, genai_errors.ServerError):
            return getattr(error, "status_code", None) in (503,)
        return False

    def _quota_help_text(self) -> str:
        return (
            "Gemini API quota is exhausted for all configured free-tier models. "
            "Wait for quota reset or use a billed Gemini project/API key. "
            "You can also set GEMINI_MODEL in .env to a model available in your account."
        )

    def _openrouter_configured(self) -> bool:
        return bool(os.getenv("OPENROUTER_API_KEY") and os.getenv("OPENROUTER_MODEL"))

    def _openrouter_generate_text(self, prompt: str) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL")
        if not api_key or not model:
            raise RuntimeError("OpenRouter fallback is not configured")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        site_url = os.getenv("OPENROUTER_SITE_URL", "")
        app_name = os.getenv("OPENROUTER_APP_NAME", "RAG Assistant")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "") or ""

    def _generate_content_with_fallback(self, prompt: str):
        last_error = None

        for candidate_model in self._candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=candidate_model,
                    contents=prompt,
                )
                if candidate_model != self.model_name:
                    print(f"[AiModel] Switching generation model to {candidate_model}.")
                    self.model_name = candidate_model
                return getattr(response, "text", "") or ""
            except Exception as error:
                last_error = error
                if self._is_retryable_model_error(error):
                    continue
                raise

        if self._openrouter_configured():
            print("[AiModel] Falling back to OpenRouter generation.")
            return self._openrouter_generate_text(prompt)

        if isinstance(last_error, genai_errors.ClientError) and getattr(last_error, "status_code", None) == 429:
            raise RuntimeError(self._quota_help_text()) from last_error

        raise RuntimeError(
            "No supported generation model is currently available for this API key/version. "
            "To enable non-Gemini fallback, set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env."
        ) from last_error

    def _generate_content_stream_with_fallback(self, prompt: str):
        last_error = None

        for candidate_model in self._candidate_models:
            try:
                stream = self.client.models.generate_content_stream(
                    model=candidate_model,
                    contents=prompt,
                )
                if candidate_model != self.model_name:
                    print(f"[AiModel] Switching generation model to {candidate_model}.")
                    self.model_name = candidate_model
                return stream
            except Exception as error:
                last_error = error
                if self._is_retryable_model_error(error):
                    continue
                raise

        if self._openrouter_configured():
            print("[AiModel] Falling back to OpenRouter generation.")
            fallback_text = self._openrouter_generate_text(prompt)

            def _single_chunk_stream():
                if fallback_text:
                    yield fallback_text

            return _single_chunk_stream()

        if isinstance(last_error, genai_errors.ClientError) and getattr(last_error, "status_code", None) == 429:
            raise RuntimeError(self._quota_help_text()) from last_error

        raise RuntimeError(
            "No supported generation model is currently available for this API key/version. "
            "To enable non-Gemini fallback, set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env."
        ) from last_error

    def ask_a_question(self, prompt: str = "Hello there!") -> str:
        """
        Generate an answer for a direct user prompt.

        Returns:
            Full generated response text.
        """
        response_text = self._generate_content_with_fallback(prompt)
        return response_text

    def ask_a_question_from_pdf(
        self,
        pdf_path: str,
        prompt: str = "tell me what is this pdf about",
    ) -> str:
        """
        Run a basic RAG flow on a PDF and return a grounded answer.
        """
        pdf_reader = PdfReader(pdf_path)
        pdf_paragraphs = pdf_reader.get_paragraphs()

        local_embedding = LocalEmbedding()
        local_embedding.build_index(pdf_paragraphs)

        relevant_sections = local_embedding.get_context(prompt, 10)
        full_prompt_for_rag = self.full_prompt_for_rag(
            relevant_sections=relevant_sections,
            question_prompt=prompt,
        )

        response_text = self._generate_content_with_fallback(full_prompt_for_rag)
        return response_text

    def ask_a_question_from_pdf_stream(
        self,
        pdf_path: str,
        prompt: str = "tell me what is this pdf about",
        local_embedding: LocalEmbedding | None = None,
    ) -> Iterator[str]:
        """
        Streaming variant of PDF question answering for UI token streaming.

        Yields:
            Response text chunks.
        """
        if local_embedding is None:
            pdf_reader = PdfReader(pdf_path)
            pdf_paragraphs = pdf_reader.get_paragraphs()
            local_embedding = LocalEmbedding()
            local_embedding.build_index(pdf_paragraphs)

        relevant_sections = local_embedding.get_context(prompt, k=10)
        full_prompt = self.full_prompt_for_rag(
            relevant_sections=relevant_sections,
            question_prompt=prompt,
        )

        try:
            stream = self._generate_content_stream_with_fallback(full_prompt)

            for chunk in stream:
                text = chunk if isinstance(chunk, str) else getattr(chunk, "text", "")
                if text:
                    yield text
        except Exception as error:
            if self._is_retryable_model_error(error):
                try:
                    fallback_text = self._generate_content_with_fallback(full_prompt)
                    if fallback_text:
                        yield fallback_text
                    return
                except Exception as fallback_error:
                    yield f"Error: {str(fallback_error)}"
                    return

            yield f"Error: {str(error)}"

    def full_prompt_for_rag(
        self,
        relevant_sections: str = "",
        question_prompt: str = "",
        relevent_sections: str | None = None,
    ) -> str:
        """
        Construct the grounded prompt used for RAG.

        Note:
            relevent_sections is kept for backward compatibility with older calls.
        """
        if relevent_sections is not None:
            relevant_sections = relevent_sections

        return f"""
You are an AI assistant. Answer the question based only on the provided document text.
If the answer is not found in the document, say: "The document does not contain information on this topic."
Do not use outside knowledge.

Document Text:
---
{relevant_sections}
---

Question: {question_prompt}
Answer:
""".strip()
