"""Evidence retrieval agent."""

from __future__ import annotations

from dataclasses import dataclass

from config import Settings, settings
from rag.vector_store import VectorStoreManager


@dataclass(frozen=True)
class Evidence:
    """A retrieved chunk with source information."""

    text: str
    filename: str
    page: int
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.filename}, page {self.page}]"


class RetrievalAgent:
    """Retrieve and format relevant textbook evidence."""

    def __init__(self, vector_store: VectorStoreManager, app_settings: Settings = settings):
        self.vector_store = vector_store
        self.settings = app_settings

    def run(self, query: str) -> list[Evidence]:
        """Return evidence above the configured distance threshold."""

        results = self.vector_store.search(query)
        evidence: list[Evidence] = []
        for document, distance in results:
            # Chroma returns a distance, not a probability. Keep the score
            # bounded while preserving relevant results across distance types.
            similarity = 1.0 / (1.0 + max(0.0, float(distance)))
            if similarity >= self.settings.confidence_threshold:
                evidence.append(
                    Evidence(
                        text=document.page_content,
                        filename=document.metadata.get("source_filename", "Unknown PDF"),
                        page=int(document.metadata.get("page_number", 0)),
                        score=similarity,
                    )
                )
        return evidence

    def all_documents(self) -> list[Evidence]:
        """Return all indexed chunks for whole-book summaries."""

        return [
            Evidence(
                text=document.page_content,
                filename=document.metadata.get("source_filename", "Unknown PDF"),
                page=int(document.metadata.get("page_number", 0)),
                score=1.0,
            )
            for document in self.vector_store.all_documents()
        ]

    @staticmethod
    def format_context(evidence: list[Evidence]) -> str:
        """Format retrieved chunks for downstream prompts."""

        return "\n\n".join(f"{item.citation}\n{item.text}" for item in evidence)
