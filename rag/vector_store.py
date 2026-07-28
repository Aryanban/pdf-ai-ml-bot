"""ChromaDB persistence and retrieval adapter."""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import Settings, settings


class VectorStoreManager:
    """Small adapter around ChromaDB and local Ollama embeddings."""

    def __init__(self, app_settings: Settings = settings):
        self.settings = app_settings
        self._store: Chroma | None = None

    @property
    def store(self) -> Chroma:
        """Return a lazily initialized persistent Chroma collection."""

        if self._store is None:
            self.settings.ensure_directories()
            embeddings = OllamaEmbeddings(
                model=self.settings.embedding_model,
                base_url=self.settings.ollama_base_url,
            )
            self._store = Chroma(
                collection_name="edugraph_documents",
                persist_directory=str(self.settings.chroma_dir),
                embedding_function=embeddings,
            )
        return self._store

    def add_documents(self, documents: list[Document], document_id: str) -> None:
        """Add chunks, replacing any previous copy of the same document."""

        self.delete_document(document_id)
        ids = [
            f"{document_id}:{index}"
            for index, _document in enumerate(documents)
        ]
        self.store.add_documents(documents, ids=ids)

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document if they exist."""

        self.store.delete(where={"document_id": document_id})

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Return relevant chunks and Chroma similarity distances."""

        return self.store.similarity_search_with_score(
            query,
            k=k or self.settings.retrieval_k,
        )

    def document_count(self) -> int:
        """Return the number of indexed chunks."""

        return self.store._collection.count()

    def all_documents(self, limit: int = 100) -> list[Document]:
        """Return indexed chunks for whole-document operations."""

        payload = self.store.get(limit=limit, include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(
                payload.get("documents", []), payload.get("metadatas", [])
            )
        ]

    def raw_collection_metadata(self) -> dict[str, Any]:
        """Expose basic collection metadata for the sidebar."""

        return {"chunks": self.document_count()}
