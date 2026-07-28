"""Summary generation agent."""

from collections.abc import Iterator

from langchain_ollama import OllamaLLM

from agents.retrieval_agent import RetrievalAgent
from config import settings
from utils.prompts import SUMMARY_PROMPT


class SummaryAgent:
    """Create revision notes grounded in uploaded documents."""

    def __init__(self, retrieval: RetrievalAgent):
        self.retrieval = retrieval
        self.llm = OllamaLLM(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            num_predict=settings.max_output_tokens,
        )

    def stream(self, topic: str) -> Iterator[str]:
        """Summarize retrieved material as tokens become available."""

        evidence = (
            self.retrieval.all_documents()
            if topic.lower() in {"the uploaded textbook", "the uploaded document"}
            else self.retrieval.run(topic)
        )
        if not evidence:
            return iter(["I could not find this information in the uploaded document."])
        prompt = SUMMARY_PROMPT.format(context=self.retrieval.format_context(evidence))
        return (str(chunk) for chunk in self.llm.stream(prompt))

    def run(self, topic: str) -> str:
        """Generate a complete summary for non-streaming callers."""

        return "".join(self.stream(topic))
