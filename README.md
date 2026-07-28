# EduGraph AI

EduGraph AI is a local intelligent tutoring assistant for asking questions about educational PDF textbooks. It uses retrieval-augmented generation so answers are grounded in textbook passages and include page citations.

> **Project status:** GitHub portfolio project in active development. The user interface and natural-language request handling are still being refined.

## Features

- Upload one or more PDF textbooks
- Extract, chunk, embed, and index textbook pages locally
- Ask grounded questions with source pages and confidence
- Generate MCQ, true/false, and short-answer quizzes
- Create revision summaries
- Evaluate student answers and identify misconceptions
- Run the language model and embeddings through Ollama

## Architecture

```text
Streamlit UI
    ├── PDF Ingestion → ChromaDB
    └── Retrieval Agent → Tutor / Quiz / Summary / Evaluation Agents → Ollama
```

Each generated response receives retrieved textbook context. When relevant evidence cannot be found, the application refuses to invent an answer.

## Requirements

- Python 3.12+
- Ollama
- At least 8 GB RAM recommended for Qwen2.5:7B

## Installation

```bash
git clone <your-repository-url>
cd EduGraph-AI
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Install Ollama from [ollama.com](https://ollama.com/), then download the models:

```bash
ollama pull qwen2.5:7b
ollama pull qllama/bge-small-en-v1.5
```

If you use a different embedding model, update `EDUGRAPH_EMBEDDING_MODEL` in `.env`.

## Run

```bash
ollama serve
STREAMLIT_CONFIG_DIR="$PWD/.streamlit" streamlit run app.py
```

Open the local URL shown by Streamlit, upload a textbook, and use the tabs to chat, generate quizzes, summarize material, or evaluate an answer.

## Project Structure

```text
app.py                 Streamlit interface
config.py              Environment-backed settings
rag/ingest.py          PDF extraction and chunking
rag/vector_store.py    ChromaDB and embedding adapter
agents/                Retrieval and educational agents
utils/prompts.py       Grounded prompt templates
data/                  Local uploads and vector database
tests/                 Lightweight automated tests
```

## Testing

```bash
pytest
```

The included tests do not require Ollama. End-to-end usage requires the local models to be installed.

## Screenshots

Add screenshots of the Chat, Quiz, Summary, and Evaluation tabs here after running the application locally.

## Future Improvements

- Conversation history and export
- Streaming responses
- OCR for scanned PDFs
- Hybrid keyword and semantic retrieval
- Adaptive quizzes based on previous answers
- Docker and GPU deployment

## License

MIT License. Add your preferred copyright notice before publishing.
