"""Student answer evaluation agent."""

from collections.abc import Iterator

from langchain_ollama import OllamaLLM

from agents.retrieval_agent import RetrievalAgent
from config import settings
from utils.prompts import EVALUATION_PROMPT


class EvaluationAgent:
    """Evaluate student answers and surface misconceptions."""

    def __init__(self, retrieval: RetrievalAgent):
        self.retrieval = retrieval
        self.llm = OllamaLLM(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            num_predict=settings.max_output_tokens,
        )

    def stream(self, question: str, answer: str) -> Iterator[str]:
        """Return constructive feedback as tokens become available."""

        evidence = self.retrieval.run(question)
        if not evidence:
            return iter(["I could not find this information in the uploaded document."])
        prompt = EVALUATION_PROMPT.format(
            question=question,
            answer=answer,
            context=self.retrieval.format_context(evidence),
        )
        return (str(chunk) for chunk in self.llm.stream(prompt))

    def run(self, question: str, answer: str) -> str:
        """Generate complete feedback for non-streaming callers."""

        return "".join(self.stream(question, answer))
