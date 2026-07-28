"""PDF ingestion and chunk preparation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, settings
from rag.vector_store import VectorStoreManager


@dataclass(frozen=True)
class IngestionResult:
    """Summary of a completed PDF ingestion operation."""

    filename: str
    pages: int
    chunks: int
    document_id: str


class PDFIngestor:
    """Extract, split, and index PDF content while preserving page metadata."""

    def __init__(self, vector_store: VectorStoreManager, app_settings: Settings = settings):
        self.vector_store = vector_store
        self.settings = app_settings

    def ingest(self, file_name: str, file_bytes: bytes) -> IngestionResult:
        """Persist and index one uploaded PDF."""

        if not file_bytes:
            raise ValueError("The uploaded PDF is empty.")
        if not file_name.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        document_id = hashlib.sha256(file_bytes).hexdigest()[:16]
        destination = self.settings.upload_dir / f"{document_id}.pdf"
        self.settings.ensure_directories()
        destination.write_bytes(file_bytes)

        pages = PyPDFLoader(str(destination)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(pages)
        for index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "document_id": document_id,
                    "source_filename": file_name,
                    "page_number": int(metadata.get("page", 0)) + 1,
                    "chunk_index": index,
                }
            )
            chunk.metadata = metadata

        self.vector_store.add_documents(chunks, document_id=document_id)
        return IngestionResult(file_name, len(pages), len(chunks), document_id)
