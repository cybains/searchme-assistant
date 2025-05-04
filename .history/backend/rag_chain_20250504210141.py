from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
from transformers import pipeline

# Initialize the model for generating responses
generator = pipeline('text-generation', model='gpt2')  # You can change this to any other model like GPT-3/4 if needed

# Load the FAISS index
def load_faiss_index():
    index = faiss.read_index('vectorstore/faiss.index')
    return index

# Embed the query
def embed_query(query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], convert_to_tensor=True)
    return query_embedding

# Retrieve the most relevant documents from FAISS
def retrieve_documents(query_embedding, index, top_k=5):
    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)
    return indices, distances

# Generate a response based on retrieved documents
def generate_response(query, indices):
    # Load documents (assuming they are stored in text files in the 'data/' folder)
    documents = []
    data_folder = "data/"
    for idx in indices[0]:
        filename = os.listdir(data_folder)[idx]
        with open(os.path.join(data_folder, filename), 'r') as file:
            documents.append(file.read())

    # Combine retrieved documents to form the context
    context = "\n".join(documents)

    # Generate response
    response = generator(f"Answer this question based on the following context:\n{context}\n\nQuestion: {query}")
    return response[0]['generated_text']

# RAG pipeline function
def rag_pipeline(query):
    # Load the FAISS index
    index = load_faiss_index()

    # Embed the query
    query_embedding = embed_query(query)

    # Retrieve relevant documents from FAISS index
    indices, _ = retrieve_documents(query_embedding, index)

    # Generate response using the retrieved documents
    response = generate_response(query, indices)
    return response

# Example usage
if __name__ == "__main__":
    query = "What is the purpose of RAG in AI?"
    print(rag_pipeline(query))
