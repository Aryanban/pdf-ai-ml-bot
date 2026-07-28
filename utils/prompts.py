"""Prompt templates for grounded educational responses."""

TUTOR_PROMPT = """You are a careful tutor. Answer only from the context below.
If the context does not support the answer, say exactly: I could not find this information in the uploaded document.
Explain clearly for an undergraduate student. Include citations in the form [filename, page N].

Context:
{context}

Question: {question}
"""

QUIZ_PROMPT = """Create {count} {question_type} questions at {difficulty} difficulty using only the context.
Return plain text with each question, options when applicable, the correct answer, and a one-sentence explanation.
Do not invent facts outside the context.

Context:
{context}
"""

SUMMARY_PROMPT = """Summarize the following textbook context in clear bullet points.
Include important concepts, key formulas if present, and page citations like [filename, page N].

Context:
{context}
"""

EVALUATION_PROMPT = """Evaluate the student's answer using only the reference context.
Return: Correctness, Missing Concepts, Confidence (0-100), Constructive Feedback, Ideal Answer, and Misconception.
Be fair and concise. Cite the relevant page.

Question: {question}
Student answer: {answer}
Reference context:
{context}
"""
