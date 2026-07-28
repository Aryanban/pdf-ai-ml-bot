"""Streamlit entry point for the single-conversation EduGraph AI app."""

from __future__ import annotations

import re
from collections.abc import Iterator

import streamlit as st

from agents.evaluation_agent import EvaluationAgent
from agents.quiz_agent import QuizAgent
from agents.retrieval_agent import RetrievalAgent
from agents.summary_agent import SummaryAgent
from agents.tutor_agent import TutorAgent
from config import settings
from rag.ingest import PDFIngestor
from rag.vector_store import VectorStoreManager
from utils.logger import get_logger


logger = get_logger(__name__)


st.set_page_config(page_title="EduGraph AI", page_icon="📚", layout="centered")

st.markdown(
    """
    <style>
        :root { color-scheme: light; }
        .stApp { background: #f7f7f8 !important; color: #202124 !important; }
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span,
        .stApp label, .stApp li, .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stChatMessage"], .stApp [data-testid="stFileUploader"] {
            color: #202124 !important;
        }
        .stApp small, .stApp [data-testid="stCaptionContainer"] { color: #5f6368 !important; }
        .stApp code { color: #303134 !important; background: #ececef !important; }
        [data-testid="stHeader"] { background: rgba(247,247,248,0.92) !important; }
        [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e6e6e8; }
        .block-container { max-width: 920px; padding-top: 2.5rem; padding-bottom: 7rem; }
        h1 { letter-spacing: -0.04em; font-size: 2.35rem !important; }
        .eyebrow { color: #6b6b73; font-size: 0.82rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
        .hero-copy { color: #55555d; font-size: 1.05rem; margin-top: -.7rem; margin-bottom: 1.4rem; }
        .upload-card { background: white; border: 1px solid #e2e2e6; border-radius: 16px; padding: 1.15rem 1.25rem .6rem; box-shadow: 0 2px 10px rgba(0,0,0,.03); }
        .upload-title { font-weight: 650; color: #202024; margin-bottom: .2rem; }
        .upload-help { color: #73737c; font-size: .9rem; margin-bottom: .65rem; }
        .doc-pill { display: inline-block; background: #f0f0f2; color: #4f4f58; border-radius: 999px; padding: .28rem .65rem; margin: .2rem .25rem 0 0; font-size: .78rem; }
        .suggestion { color: #707079; font-size: .9rem; margin-top: .8rem; }
        [data-testid="stFileUploaderDropzone"] {
            background: #ffffff !important;
            border: 1px dashed #aeb4bf !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploaderDropzone"] * { color: #202124 !important; }
        [data-testid="stFileUploaderDropzone"] small { color: #5f6368 !important; }
        [data-testid="stFileUploader"] button,
        .stButton > button {
            background: #ffffff !important;
            color: #202124 !important;
            border: 1px solid #b8bbc2 !important;
            border-radius: 10px !important;
        }
        [data-testid="stFileUploader"] button:hover,
        .stButton > button:hover { background: #f1f3f4 !important; border-color: #7b8190 !important; }
        [data-testid="stChatMessage"] { background: transparent !important; border-radius: 14px; }
        [data-testid="stChatMessage"] * { color: #202124 !important; }
        [data-testid="stChatInput"] {
            background: #ffffff !important;
            border: 1px solid #aeb4bf !important;
            border-radius: 14px !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #202124 !important;
            background: #ffffff !important;
            caret-color: #202124 !important;
        }
        [data-testid="stChatInput"] textarea::placeholder { color: #6b7280 !important; opacity: 1 !important; }
        input, textarea { color: #202124 !important; background: #ffffff !important; }
        input::placeholder, textarea::placeholder { color: #6b7280 !important; opacity: 1 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_components() -> tuple[VectorStoreManager, PDFIngestor, RetrievalAgent, TutorAgent, QuizAgent, SummaryAgent, EvaluationAgent]:
    """Create shared application components once per Streamlit process."""

    store = VectorStoreManager(settings)
    retrieval = RetrievalAgent(store, settings)
    return (
        store,
        PDFIngestor(store, settings),
        retrieval,
        TutorAgent(retrieval, settings),
        QuizAgent(retrieval, settings),
        SummaryAgent(retrieval),
        EvaluationAgent(retrieval),
    )


def _topic_from_request(request: str, marker: str) -> str:
    """Remove an action marker from a natural-language request."""

    topic = re.sub(marker, "", request, count=1, flags=re.IGNORECASE)
    topic = re.sub(
        r"\b(please|give|make|create|generate|me|a|an|the|on|about)\b",
        " ",
        topic,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", topic).strip(" .,:;") or "the uploaded textbook"


def _stream_request(
    request: str,
    tutor: TutorAgent,
    quiz: QuizAgent,
    summary: SummaryAgent,
    evaluation: EvaluationAgent,
) -> tuple[Iterator[str], str]:
    """Route one request and return a stream plus optional trailing metadata."""

    lowered = request.lower()
    if "quiz" in lowered or "test me" in lowered:
        topic = _topic_from_request(request, r"\b(quiz|test me)\b")
        question_type = "True/False" if "true false" in lowered or "true/false" in lowered else "MCQ"
        difficulty = next((level for level in ("hard", "medium", "easy") if level in lowered), "medium")
        return quiz.stream(topic, question_type, difficulty.title(), 5), ""

    if any(word in lowered for word in ("summary", "summarize", "summarise", "revision notes")):
        topic = _topic_from_request(request, r"\b(summary|summarize|summarise|revision notes)\b")
        return summary.stream(topic), ""

    if any(word in lowered for word in ("feedback", "evaluate my answer", "check my answer", "grade my answer")):
        previous_user = next(
            (message["content"] for message in reversed(st.session_state.messages[:-1]) if message["role"] == "user"),
            request,
        )
        return evaluation.stream(previous_user, request), ""

    stream, evidence, confidence = tutor.stream(request)
    source_line = f"\n\n*Confidence: {confidence:.0%}*" if evidence else ""
    return stream, source_line


def main() -> None:
    """Render the focused EduGraph AI workspace."""

    store, ingestor, _retrieval, tutor, quiz, summary, evaluation = get_components()
    st.markdown('<div class="eyebrow">Local study assistant</div>', unsafe_allow_html=True)
    st.title("EduGraph AI")
    st.markdown('<div class="hero-copy">Upload your textbook, then ask anything about it.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="upload-title">Add your textbook</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-help">PDFs stay on your computer and become searchable study material.</div>', unsafe_allow_html=True)
        uploads = st.file_uploader(
            "Upload PDF textbooks",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads:
            for upload in uploads:
                marker = f"processed_{upload.name}_{upload.size}"
                if marker not in st.session_state:
                    try:
                        with st.spinner(f"Indexing {upload.name}…"):
                            result = ingestor.ingest(upload.name, upload.getvalue())
                        st.session_state[marker] = result
                    except Exception as exc:  # UI boundary: show safe error and keep app alive.
                        st.error(f"Could not index {upload.name}: {exc}")

        indexed = [value for key, value in st.session_state.items() if key.startswith("processed_")]
        if indexed:
            pills = "".join(f'<span class="doc-pill">📄 {item.filename}</span>' for item in indexed)
            st.markdown(pills, unsafe_allow_html=True)
            st.caption(f"{sum(item.pages for item in indexed)} pages · {sum(item.chunks for item in indexed)} searchable chunks")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state.messages and not indexed:
        st.markdown('<div class="suggestion">Try: “Summarize chapter 2” · “Make me a quiz on photosynthesis” · “What is binary search?”</div>', unsafe_allow_html=True)

    has_documents = bool(indexed) or store.document_count() > 0
    request = st.chat_input(
        "Ask a question, request a quiz, summary, or feedback…",
        disabled=not has_documents,
    )
    if request:
        st.session_state.messages.append({"role": "user", "content": request})
        with st.chat_message("user"):
            st.markdown(request)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your textbook…"):
                    response_stream, suffix = _stream_request(
                        request, tutor, quiz, summary, evaluation
                    )
                generated = st.write_stream(response_stream)
                response = (
                    generated
                    if isinstance(generated, str)
                    else "".join(str(part) for part in generated)
                )
                response += suffix
                if suffix:
                    st.markdown(suffix)
            except Exception as exc:
                logger.exception("Request failed")
                if "connection" in str(exc).lower() or "11434" in str(exc):
                    response = "Ollama is not responding. Please open Ollama and verify the model is installed."
                else:
                    response = "I couldn't complete that request. Please try again."
                st.error(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.markdown("### EduGraph AI")
        st.caption("A private, local-first study workspace.")
        st.caption("GitHub portfolio project · Active development")
        st.divider()
        st.caption(f"Model  ·  `{settings.chat_model}`")
        st.caption(f"Embeddings  ·  `{settings.embedding_model}`")
        st.info("Currently refining the UI and natural-language request handling.")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
