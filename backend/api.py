from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chain import rag_pipeline  # Your existing RAG logic

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required."}), 400

    response = rag_pipeline(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
