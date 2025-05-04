from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def retrieve_documents(query, index):
    # Create embeddings for the query
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Same model as used in ingest
    query_embedding = model.encode([query], convert_to_tensor=True)

    # Search in the FAISS index
    _, indices = index.search(query_embedding, k=5)  # Retrieve top 5 relevant docs
    return indices

def generate_response(documents):
    # Use Hugging Face for generating responses
    summarizer = pipeline("summarization")
    response = summarizer(documents)
    return response

def rag_pipeline(query):
    # Load FAISS index
    index = faiss.read_index("vectorstore/faiss.index")
    
    # Retrieve documents
    retrieved_docs = retrieve_documents(query, index)
    
    # Retrieve actual document text based on indices
    documents = [open(f"data/{idx}.txt").read() for idx in retrieved_docs]

    # Generate response
    response = generate_response(documents)
    return response
