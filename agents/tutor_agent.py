"""Grounded tutoring agent."""

from collections.abc import Iterator

from langchain_ollama import OllamaLLM

from agents.retrieval_agent import Evidence, RetrievalAgent
from config import Settings, settings
from utils.prompts import TUTOR_PROMPT


class TutorAgent:
    """Answer questions using only retrieved textbook evidence."""

    def __init__(self, retrieval: RetrievalAgent, app_settings: Settings = settings):
        self.retrieval = retrieval
        self.settings = app_settings
        self.llm = OllamaLLM(
            model=app_settings.chat_model,
            base_url=app_settings.ollama_base_url,
            temperature=app_settings.temperature,
            num_predict=app_settings.max_output_tokens,
        )

    def stream(self, question: str) -> tuple[Iterator[str], list[Evidence], float]:
        """Retrieve evidence and return a token stream immediately."""

        evidence = self.retrieval.run(question)
        if not evidence:
            return iter(["I could not find this information in the uploaded document."]), [], 0.0
        prompt = TUTOR_PROMPT.format(
            question=question,
            context=self.retrieval.format_context(evidence),
        )
        confidence = sum(item.score for item in evidence) / len(evidence)
        return (str(chunk) for chunk in self.llm.stream(prompt)), evidence, confidence

    def run(self, question: str) -> tuple[str, list[Evidence], float]:
        """Generate a complete answer for non-streaming callers and tests."""

        stream, evidence, confidence = self.stream(question)
        return "".join(stream), evidence, confidence
