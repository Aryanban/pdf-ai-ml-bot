"""Application configuration for EduGraph AI.

The project is intentionally local-first. Configuration is read from
environment variables when present and otherwise falls back to sensible
development defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _read_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback."""

    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def _read_float(name: str, default: float) -> float:
    """Read a floating-point environment variable with a safe fallback."""

    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings used by the EduGraph AI application."""

    project_dir: Path
    upload_dir: Path
    chroma_dir: Path
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    confidence_threshold: float
    temperature: float
    max_output_tokens: int

    def ensure_directories(self) -> None:
        """Create local data directories required by the application."""

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Build application settings from environment variables and defaults."""

    project_dir = Path(
        os.getenv("EDUGRAPH_PROJECT_DIR", str(Path(__file__).resolve().parent))
    ).expanduser().resolve()

    return Settings(
        project_dir=project_dir,
        upload_dir=Path(
            os.getenv("EDUGRAPH_UPLOAD_DIR", str(project_dir / "data" / "uploads"))
        ).expanduser(),
        chroma_dir=Path(
            os.getenv("EDUGRAPH_CHROMA_DIR", str(project_dir / "data" / "chroma"))
        ).expanduser(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        chat_model=os.getenv("EDUGRAPH_CHAT_MODEL", "qwen2.5:7b"),
        embedding_model=os.getenv(
            "EDUGRAPH_EMBEDDING_MODEL", "qllama/bge-small-en-v1.5"
        ),
        chunk_size=_read_int("EDUGRAPH_CHUNK_SIZE", 800),
        chunk_overlap=_read_int("EDUGRAPH_CHUNK_OVERLAP", 120),
        retrieval_k=_read_int("EDUGRAPH_RETRIEVAL_K", 5),
        confidence_threshold=_read_float("EDUGRAPH_CONFIDENCE_THRESHOLD", 0.20),
        temperature=_read_float("EDUGRAPH_TEMPERATURE", 0.2),
        max_output_tokens=_read_int("EDUGRAPH_MAX_OUTPUT_TOKENS", 700),
    )


settings = load_settings()
