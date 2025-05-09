from flask import Flask, request, jsonify
from rag_chain import rag_pipeline

app = Flask(__name__)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query missing"}), 400

    response = rag_pipeline(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
