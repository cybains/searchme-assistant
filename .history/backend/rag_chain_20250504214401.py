import os
import faiss
import numpy as np
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

# Define constants for your directory and model
DATA_FOLDER = "data/"  # Folder where your documents are located
VECTORSTORE_PATH = "vectorstore/faiss.index"  # Path to your FAISS index
MODEL_NAME = 'facebook/bart-base'  # Change to your desired Hugging Face model (e.g., 'facebook/bart-base', 't5-base', etc.)
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Use a sentence transformer model for embedding

# Initialize the Hugging Face model for text generation
generator = pipeline('text-generation', model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the SentenceTransformer model for document embeddings
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# Load the FAISS index
index = faiss.read_index(VECTORSTORE_PATH)

# Embed documents
def embed_documents(documents):
    embeddings = embed_model.encode(documents, convert_to_tensor=True)
    return embeddings

# Retrieve the most relevant documents from FAISS index
def retrieve_documents(query, top_k=5):
    query_embedding = embed_model.encode([query], convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()
    
    # Search FAISS index for the top_k most relevant documents
    distances, indices = index.search(query_embedding, top_k)
    
    documents = []
    for idx in indices[0]:
        filename = os.listdir(DATA_FOLDER)[idx]
        with open(os.path.join(DATA_FOLDER, filename), 'r') as f:
            documents.append(f.read())
    
    return documents

def generate_response(query, documents):
    # Concatenate the documents (context) into a single string
    context = " ".join(documents)
    
    # Combine context and query into a single input text
    input_text = context + " " + query  # Ensure it's a single string
    
    # Generate the output using the pipeline
outputs = generator(input_text, max_new_tokens=50, num_return_sequences=1, do_sample=True)
    
    # Get the generated response
    generated_text = outputs[0]['generated_text']
    
    return generated_text

# Main function for the RAG pipeline
def rag_pipeline(query):
    # Retrieve relevant documents from FAISS
    relevant_docs = retrieve_documents(query)
    
    # Generate a response based on the query and the retrieved documents
    response = generate_response(query, relevant_docs)
    
    return response

if __name__ == "__main__":
    query = "What is the best approach for starting a business in Portugal?"  # Example query
    print(rag_pipeline(query))

