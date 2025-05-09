import os
import faiss
import numpy as np
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import re

# Define constants for your directory and model
DATA_FOLDER = "data/"  # Folder where your documents are located
VECTORSTORE_PATH = "vectorstore/faiss.index"  # Path to your FAISS index
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
embed_model = SentenceTransformer(EMBEDDING_MODEL)
MODEL_NAME = 'facebook/bart-base'

# Function to clean and normalize text
def clean_text(text):
    text = text.replace('\n', ' ')  # Remove new lines
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into one
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Function to preprocess and chunk the text into manageable sizes
def preprocess_and_chunk(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        cleaned = clean_text(chunk)
        if cleaned:
            chunks.append(cleaned)
    return chunks

# Initialize the Hugging Face model for text generation
generator = pipeline('text2text-generation', model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the FAISS index
index = faiss.read_index(VECTORSTORE_PATH)

# Embed documents
def embed_documents(documents):
    embeddings = embed_model.encode(documents, convert_to_tensor=True)
    return embeddings

# Retrieve the most relevant documents from FAISS index
def retrieve_documents(query, top_k=5):
    with open("vectorstore/documents.txt", "r", encoding="utf-8") as f:
        all_chunks = f.readlines()

    query_embedding = embed_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)

    documents = [all_chunks[idx].strip() for idx in indices[0]]

    # LOGGING: Show retrieved chunks with rank and distance
    print("\n🔍 Top Retrieved Chunks for Query: '{}'".format(query))
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        print(f"\n#{rank} [Index {idx}] (Distance: {dist:.4f}):")
        print(all_chunks[idx].strip())

    return documents

# Generate a more structured and relevant response
def generate_response(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are an assistant answering user questions based on Portuguese immigration laws.

Context:
{context}

Question: {query}

Answer:"""
    response = generator(prompt, max_length=512, do_sample=False)[0]['generated_text']
    return response

# Main function for the RAG pipeline
def rag_pipeline(query):
    # Retrieve relevant documents from FAISS
    relevant_docs = retrieve_documents(query)
    
    # Generate a response based on the query and the retrieved documents
    response = generate_response(query, relevant_docs)
    
    return response

if __name__ == "__main__":
    query = "What is the validity of a resident permit for students?"  # Example query
    print(rag_pipeline(query))
