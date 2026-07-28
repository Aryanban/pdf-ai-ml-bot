"""Quiz generation agent."""

from collections.abc import Iterator

from langchain_ollama import OllamaLLM

from agents.retrieval_agent import RetrievalAgent
from config import Settings, settings
from utils.prompts import QUIZ_PROMPT


class QuizAgent:
    """Generate simple source-grounded quizzes."""

    def __init__(self, retrieval: RetrievalAgent, app_settings: Settings = settings):
        self.retrieval = retrieval
        self.llm = OllamaLLM(
            model=app_settings.chat_model,
            base_url=app_settings.ollama_base_url,
            temperature=app_settings.temperature,
            num_predict=app_settings.max_output_tokens,
        )

    def stream(self, topic: str, question_type: str, difficulty: str, count: int) -> Iterator[str]:
        """Generate quiz text as tokens become available."""

        evidence = self.retrieval.run(topic)
        if not evidence:
            return iter(["I could not find this information in the uploaded document."])
        prompt = QUIZ_PROMPT.format(
            count=count,
            question_type=question_type,
            difficulty=difficulty,
            context=self.retrieval.format_context(evidence),
        )
        return (str(chunk) for chunk in self.llm.stream(prompt))

    def run(self, topic: str, question_type: str, difficulty: str, count: int) -> str:
        """Generate a complete quiz for non-streaming callers."""

        return "".join(self.stream(topic, question_type, difficulty, count))
