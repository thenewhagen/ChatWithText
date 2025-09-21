import base64
import os
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

# --- Conditional Imports based on Provider ---
# These libraries will be used depending on the BACKEND_PROVIDER setting.
# On GCP, you would not need to install the Ollama library.
import vertexai
from vertexai.language_models import TextEmbeddingModel
import ollama

# Init Flask app
app = Flask(__name__)

# --- Environment Variable Configuration ---
# Use this variable to switch between backends.
# Set it to "ollama" for local development, or "vertexai" for GCP.
# The default is "ollama" for easy local testing.
PROVIDER = os.environ.get("BACKEND_PROVIDER", "ollama")

# --- Model Initialization ---
# This block handles the conditional loading of the correct model
embedding_model = None
try:
    if PROVIDER == "vertexai":
        # Read the project ID and location from environment variables,
        # which are automatically set by GCP services like Cloud Run.
        PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
        LOCATION = os.environ.get("LOCATION", "us-central1")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko")
        print(f"Using Vertex AI for embeddings. Project: '{PROJECT_ID}', Location: '{LOCATION}'.")
    elif PROVIDER == "ollama":
        print("Using Ollama for embeddings and generation. Make sure Ollama server is running locally.")
    else:
        raise ValueError(f"Unsupported BACKEND_PROVIDER: {PROVIDER}")
except Exception as e:
    print(f"Error initializing backend model: {e}")
    # We set the embedding_model to None to handle this gracefully later.
    embedding_model = None

# In-memory storage
document_chunks = []
document_embeddings = []

# Create embedding
def get_embedding(text: str) -> np.ndarray:
    """Gets the text embedding using the selected provider."""
    if PROVIDER == "vertexai":
        if not embedding_model:
            raise RuntimeError("Vertex AI embedding model is not initialized.")
        embeddings = embedding_model.get_embeddings([text])
        return np.array(embeddings[0].values)
    elif PROVIDER == "ollama":
        embedding = ollama.embeddings(
            model='nomic-embed-text',
            prompt=text
        )['embedding']
        return np.array(embedding)
    else:
        raise RuntimeError("Invalid backend provider.")

# Simple chunking (word-based)
def chunk_text(text, chunk_size=300):
    """Breaks a large text into smaller chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

@app.route("/upload", methods=["POST"])
def upload_document():
    """Endpoint to upload a base64 encoded document and create embeddings."""
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
    
    # We generate embeddings only if the provider is initialized
    try:
        document_embeddings = [get_embedding(chunk) for chunk in document_chunks]
    except Exception as e:
        return jsonify({"error": f"Failed to generate embeddings: {str(e)}"}), 500

    return jsonify({"message": "Document uploaded and embeddings created.", "chunks": len(document_chunks)})

@app.route("/ask", methods=["POST"])
def ask_question():
    """Endpoint to ask a question against the uploaded document."""
    global document_chunks, document_embeddings

    if not document_chunks:
        return jsonify({"error": "No document uploaded yet."}), 400

    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        # Get embedding for the question
        q_embedding = get_embedding(question).reshape(1, -1)
        doc_matrix = np.array(document_embeddings)
        sims = cosine_similarity(q_embedding, doc_matrix)[0]
    except Exception as e:
        return jsonify({"error": f"Failed to generate question embedding: {str(e)}"}), 500

    # Find the best chunk
    best_idx = int(np.argmax(sims))
    best_chunk = document_chunks[best_idx]
    
    if PROVIDER == "ollama":
        # For Ollama, we use the retrieved chunk as context for a generative model
        prompt = f"Using the following context, answer the question:\n\nContext: {best_chunk}\n\nQuestion: {question}"
        try:
            response = ollama.generate(
                model='mistral',
                prompt=prompt
            )['response']
            return jsonify({"answer": response, "similarity": float(sims[best_idx]), "source_chunk": best_chunk})
        except Exception as e:
            return jsonify({"error": f"Failed to generate response with Ollama: {str(e)}"}), 500
    else: # This covers Vertex AI and any other non-Ollama backend
        # For other backends, we return the most relevant chunk directly
        return jsonify({"answer": best_chunk, "similarity": float(sims[best_idx])})

@app.route("/test", methods=["GET"])
def test():
    """Simple test endpoint to check if the API is running."""
    return f"API is running. Backend provider is: {PROVIDER}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)




#curl -X POST localhost:8080/upload -H "Content-Type: application/json" -d '{"document":"RGVyIEJTQyBTcHJvY2tow7Z2ZWwgaXN0IGVpbiByZWluZXIgQm9nZW5zcG9ydHZlcmVpbiB1bmQgd3VyZGUgaW0gSmFociAyMDAxIGdlZ3LDvG5kZXQuIERlciBWZXJlaW4gaGF0IHJ1bmQgMTgwIE1pdGdsaWVkZXIsIGVpbiBuaWNodCB1bmVyaGVibGljaGVyIFRlaWwgc2luZCBLaW5kZXIgdW5kIEp1Z2VuZGxpY2hlLiBEaWUgw7xiZXJ3aWVnZW5kZSBNZWhyaGVpdCBzdGFtbXQgYXVzIFNwcm9ja2jDtnZlbCwgSGF0dGluZ2VuIHVuZCBXdXBwZXJ0YWwuIEF1Y2ggYXVzIETDvHNzZWxkb3JmLCBCb2NodW0sIEVzc2VuIHVuZCB2aWVsZW4gYW5kZXJlbiBTdMOkZHRlbiBmaW5kZW4gR2xlaWNoZ2VzaW5udGUgcmVnZWxtw6TDn2lnIGRlbiBXZWcgYXVmIHVuc2VyIEdlbMOkbmRlLgogICAgRGVyIEJTQyBpc3QgTWl0Z2xpZWQgaW0gV2VzdGbDpGxpc2NoZW4gU2Now7x0emVuYnVuZCAoV1NCKSwgaW0gQm9nZW5zcG9ydHZlcmJhbmQgTm9yZHJoZWluLVdlc3RmYWxlbiAoQlZOVyksIGltIExhbmRlc3Nwb3J0YnVuZCAoTFNCKSB1bmQgaW0gU3RhZHRzcG9ydHZlcmJhbmQgU3Byb2NraMO2dmVsIChTU1YpLgogICAgSW4gdW5zZXJlbSBWZXJlaW4gaXN0IG5haGV6dSBkaWUgdm9sbHN0w6RuZGlnZSBQYWxldHRlIGRlcyBCb2dlbnNwb3J0cyB2ZXJ0cmV0ZW4sIHZvbSBQcmltaXRpdmJvZ2VuIE1hcmtlIEVpZ2VuYmF1IGJpcyBoaW4genVtIHRlY2huaXNpZXJ0ZW4gQ29tcG91bmRib2dlbi4gRGllc2UgVmllbGZhbHQgaXN0IEF1c2RydWNrIHVuc2VyZXIgZ3J1bmRzw6R0emxpY2hlbiBFaW5zdGVsbHVuZywgbmFjaCBkZXIgZGVyIFNwYcOfIGFtIEJvZ2Vuc2NoaWXDn2VuIGltIFZvcmRlcmdydW5kIHN0ZWhlbiBzb2xsIHVuZCBkZW1lbnRzcHJlY2hlbmQgYmVpIHVucyBqZWRlciBkZW4gU3BvcnQgbmFjaCBzZWluZW4gVm9yc3RlbGx1bmdlbiBhdXPDvGJlbiBrYW5uLgogICAgT2J3b2hsIGRlciBMZWlzdHVuZ3NnZWRhbmtlIG5pY2h0IGltIFZvcmRlcmdydW5kIHN0ZWh0LCBuZWhtZW4gTWl0Z2xpZWRlciByZWdlbG3DpMOfaWcgYW4gTWVpc3RlcnNjaGFmdGVuIGRlcyBXU0IgdW5kIGRlcyBCVk5XIHNvd2llIGFuIFR1cm5pZXJlbiBhbmRlcmVyIFZlcmVpbmUgdGVpbC4gSGllcmJlaSBzaW5kIGRpZSBhYnNvbHZpZXJ0ZW4gRGlzemlwbGluZW4gdW5kIGRpZSB6dW0gRWluc2F0eiBrb21tZW5kZW4gQsO2Z2VuIHZpZWxzY2hpY2h0aWcuIEJlc3VjaHQgd2VyZGVuIEZlbGRib2dlbi0gb2RlciAzRC1UdXJuaWVyZSBtaXQgZGVtIExhbmctIHVuZCBKYWdkYm9nZW4gb2RlciBGaXRhLVR1cm5pZXJlIG1pdCBDb21wb3VuZCB1bmQgQmxhbmtib2dlbi4gRHVyY2ggd2llZGVyaG9sdCBndXRlIFBsYXR6aWVydW5nZW4gYmlzIGhpbiB6dSBtZWhyZmFjaGVuIFRpdGVsbiBlaW5lcyBMYW5kZXNtZWlzdGVycyB1bmQgZWluZXMgRGV1dHNjaGVuIE1laXN0ZXJzIGhhdCBzaWNoIGRlciBCU0MgZWluZW4gTmFtZW4gZ2VtYWNodC4KICAgIE5lYmVuIGRlbiBzcG9ydGxpY2hlbiBBa3Rpdml0w6R0ZW4gYmlldGVuIHdpciBydW5kIHVtIGRhcyBKYWhyIHZlcnNjaGllZGVuZSBWZXJlaW5zYWt0aXZpdMOkdGVuIGbDvHIgSnVuZyB1bmQgQWx0LCBmw7xyIE1pdGdsaWVkZXIgdW5kIEfDpHN0ZSBhbi4KICAgIFplbnRyYWxlIFZlcmFuc3RhbHR1bmdlbiBydW5kIHVtIGRhcyBTY2jDvHR6ZW5qYWhyIHNpbmQgZGFzIE9zdGVyZmV1ZXIgdW5kIGVpbiBqw6RocmxpY2hlcyBTb21tZXJmZXN0LiBTZWl0IDIwMDkgZ2lidCBlcyBkYXMgSGVyYnN0ZmVzdCB1bmQgc2VpdCAyMDExIGVpbiBGYW1pbGllbi1UdXJuaWVyLCB3ZWxjaGUgZ3Jvw59lbiBadXNwcnVjaCBmaW5kZW4uCiAgICBOZWJlbiBkZW0gSMO8Z2VsbGFuZHR1cm5pZXIsIHVuc2VyZW0gc2VpdCBKYWhyZW4gw7xiZXIgMiBUYWdlbiBhdXNnZXJpY2h0ZXRlbiAzMG0tQm9nZW5zcG9ydHR1cm5pZXIgbWl0IG1laHIgYWxzIDEwMCBCb2dlbnNjaMO8dHplbiwgdW5kIFZlcmVpbnNtZWlzdGVyc2NoYWZ0ZW4gM0QsIGltIEZyZWllbiB1bmQgaW4gZGVyIEhhbGxlIHNjaGxpZcOfdCBkYXMgZ2VtZWluc2FtZSBKYWhyZXNhYnNjaGx1c3NmZXN0IGRhcyBzcG9ydGxpY2hlIEphaHIgYmVpbSBCU0MgU3Byb2NraMO2dmVsIGFiLgogICAgU2VpdCBBcHJpbCAyMDE4IGhhYmVuIHdpciBlaW5lIG5ldWUgSGVpbWF0IGF1ZiBkZW0gU3BvcnRwbGF0eiBpbiBIYXR0aW5nZW4tT2JlcnN0w7x0ZXIgZ2VmdW5kZW4uIEhpZXIgaGFiZW4gd2lyIG5lYmVuIGJlc3RlbiBQbGF0emJlZGluZ3VuZ2VuIFLDpHVtbGljaGtlaXRlbiB6dXIgVmVyZsO8Z3VuZywgZGllIGRlbiBBdWZlbnRoYWx0IGFicnVuZGVuLiBFcnN0bWFscyBoYWJlbiB3aXIgaGllciBkaWUgTcO2Z2xpY2hrZWl0LCB1bnMgaW4gZWluZW0ga2xlaW5lbiAzRC1UcmFpbmluZ3BhcmNvdXJzIChudXIgZsO8ciBNaXRnbGllZGVyISkgYXVmIGVudHNwcmVjaGVuZGUgVHVybmllcmUgdm9yenViZXJlaXRlbi4KICAgIERhcyBHZWzDpG5kZSBzdGVodCB1bnNlcmVuIE1pdGdsaWVkZXJuIGdydW5kc8OkdHpsaWNoIGplZGVyemVpdCB6dXIgVmVyZsO8Z3VuZy4gTnVyIGFtIERpZW5zdGFnIGFiIDE2LjAwIFVociB1bmQgYW0gU2Ftc3RhZyB2b24gMTQuMDAgLSBjYS4gMTYuMDAgVWhyIChKdWdlbmR0cmFpbmluZykgaXN0IGtlaW4gU2NoaWXDn2JldHJpZWIgbcO2Z2xpY2guCiAgICBVbnNlciBDbHViaGF1cyBpc3QgamVkZW4gRG9ubmVyc3RhZyBhYiAxODowMCBVaHIgdW5kIGplZGVuIFNvbm50YWcgdm9uIDEwOjAwIGJpcyAxNDowMCBVaHIgZ2XDtmZmbmV0LiAKICAgIA=="}'
