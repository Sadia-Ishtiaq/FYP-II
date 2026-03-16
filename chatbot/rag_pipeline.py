"""
RAG pipeline: retrieves relevant document chunks and generates a response via Google Gemini API.
"""

import os
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 4  # number of chunks to retrieve

# Load once at module level to avoid reloading on every request
_embedder = None
_index = None
_chunks = None
_gemini = None

def _load_resources():
    global _embedder, _index, _chunks, _gemini

    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)

    if _index is None:
        _index = faiss.read_index(f"{VECTOR_STORE_DIR}/index.faiss")

    if _chunks is None:
        with open(f"{VECTOR_STORE_DIR}/chunks.pkl", "rb") as f:
            _chunks = pickle.load(f)

    if _gemini is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        _gemini = genai.GenerativeModel(GEMINI_MODEL)


def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """Embed the query and retrieve the top-k most relevant document chunks."""
    _load_resources()
    query_vec = _embedder.encode([query]).astype("float32")
    _, indices = _index.search(query_vec, top_k)
    retrieved = [_chunks[i]["text"] for i in indices[0] if i < len(_chunks)]
    return "\n\n".join(retrieved)


def build_prompt(report_text: str, question: str, context: str) -> str:
    """Construct the prompt sent to the LLM."""
    return f"""You are a helpful medical assistant for DeepNeuro, an AI-powered brain tumor detection system.
Your role is to explain AI-generated MRI reports to patients in simple, clear, and reassuring language.

Important rules:
- Do NOT provide a medical diagnosis or recommend specific treatments.
- Always remind the patient to consult their doctor or specialist.
- Use plain, easy-to-understand language — avoid heavy medical jargon.
- Base your answer only on the report and the medical information provided below.

--- MRI REPORT ---
{report_text}

--- RELEVANT MEDICAL INFORMATION ---
{context}

--- PATIENT QUESTION ---
{question}

--- YOUR RESPONSE ---
Answer the patient's question in simple language using the report and medical information above.
End your response with a reminder to follow up with their doctor."""


def ask(report_text: str, question: str) -> str:
    """Full RAG pipeline: retrieve context, build prompt, call Gemini."""
    _load_resources()
    context = retrieve_context(question)
    prompt = build_prompt(report_text, question, context)
    response = _gemini.generate_content(prompt)
    return response.text
