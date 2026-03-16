"""
Flask API for the DeepNeuro RAG chatbot.
Endpoints:
  GET  /          - Chat UI
  POST /chat      - Send a question with an uploaded PDF report
"""

import io
from flask import Flask, request, jsonify, render_template
from pypdf import PdfReader
from rag_pipeline import ask

app = Flask(__name__)
MAX_PDF_SIZE_MB = 5


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400

    pdf_file = request.files.get("report")
    if not pdf_file:
        return jsonify({"error": "A PDF report file is required."}), 400

    pdf_bytes = pdf_file.read()
    if len(pdf_bytes) > MAX_PDF_SIZE_MB * 1024 * 1024:
        return jsonify({"error": f"PDF exceeds {MAX_PDF_SIZE_MB}MB limit."}), 400

    try:
        report_text = extract_text_from_pdf(pdf_bytes)
        if not report_text:
            return jsonify({"error": "Could not extract text from the PDF."}), 400

        answer = ask(report_text, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
