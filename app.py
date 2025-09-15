import base64
import os
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.language_models import TextEmbeddingModel
import vertexai

# Init Flask app
app = Flask(__name__)

# Initialize Vertex AI
vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location="us-central1")
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko")

# In-memory storage
document_chunks = []
document_embeddings = []

# Create embedding
def get_embedding(text: str) -> np.ndarray:
    embeddings = embedding_model.get_embeddings([text])
    return np.array(embeddings[0].values)

# Simple chunking
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

@app.route("/upload", methods=["POST"])
def upload_document():
    global document_chunks, document_embeddings
    data = request.get_json()
    base64_doc = data.get("document")

    if not base64_doc:
        return jsonify({"error": "Missing base64 document"}), 400

    try:
        decoded_bytes = base64.b64decode(base64_doc)
        text = decoded_bytes.decode("utf-8")
    except Exception as e:
        return jsonify({"error": f"Invalid base64 input: {str(e)}"}), 400

    document_chunks = chunk_text(text)
    document_embeddings = [get_embedding(chunk) for chunk in document_chunks]

    return jsonify({"message": "Document uploaded and embeddings created.", "chunks": len(document_chunks)})

@app.route("/ask", methods=["POST"])
def ask_question():
    global document_chunks, document_embeddings

    if not document_chunks:
        return jsonify({"error": "No document uploaded yet."}), 400

    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    q_embedding = get_embedding(question).reshape(1, -1)
    doc_matrix = np.array(document_embeddings)
    sims = cosine_similarity(q_embedding, doc_matrix)[0]

    best_idx = int(np.argmax(sims))
    best_chunk = document_chunks[best_idx]

    return jsonify({"answer": best_chunk, "similarity": float(sims[best_idx])})

@app.route("/test", methods=["GET"])
def test():
    return "API is running (Vertex AI version)"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
