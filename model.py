import os
from typing import Iterator

from dotenv import load_dotenv
from google import genai

from local_embedding import LocalEmbedding
from pdf_reader import PdfReader


class AiModel:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize Gemini client for answer generation.

        Args:
            model_name: Gemini text generation model ID.
        """
        load_dotenv()
        self.model_name = model_name

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")

        print("Running checks to make sure everything is good...")
        self.gemini_auth(api_key)
        print("Model client ready.")

    def gemini_auth(self, api_key: str) -> None:
        """Create a Gemini client using the API key from environment."""
        self.client = genai.Client(api_key=api_key)

    def ask_a_question(self, prompt: str = "Hello there!") -> str:
        """
        Generate an answer for a direct user prompt.

        Returns:
            Full generated response text.
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return getattr(response, "text", "") or ""

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

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt_for_rag,
        )
        return getattr(response, "text", "") or ""

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

        stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=full_prompt,
        )

        for chunk in stream:
            text = getattr(chunk, "text", "")
            if text:
                yield text

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
