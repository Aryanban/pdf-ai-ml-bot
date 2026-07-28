"""Tests for configuration defaults."""

from config import settings


def test_default_settings_are_local() -> None:
    """The default configuration points at local services and data."""

    assert settings.chat_model == "qwen2.5:7b"
    assert settings.ollama_base_url.startswith("http://localhost")
    assert settings.embedding_model == "qllama/bge-small-en-v1.5"
    assert settings.chunk_overlap < settings.chunk_size
