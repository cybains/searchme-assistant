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
/*************  ✨ Windsurf Command 🌟  *************/
# Check if the code is correct
try:

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

# Generate a response using the retrieved documents
def generate_response(query, documents):
    context = " ".join(documents)  # Concatenate the top documents to form context
    input_text = context + " " + query  # Concatenate context and query as strings
    
    print(f"Input text: {input_text}")  # Debugging line: Check the concatenated input
    
    # Tokenize the input text (context + query)
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024)
    
    # Generate a response with the Hugging Face model
    generated = generator(inputs['input_ids'], max_length=150, num_return_sequences=1, do_sample=True)
    
    return generated[0]['generated_text']

# Main function for the RAG pipeline
def rag_pipeline(query):
    # Retrieve relevant documents from FAISS
    relevant_docs = retrieve_documents(query)
    
    # Generate a response based on the query and the retrieved documents
    response = generate_response(query, relevant_docs)
    
    return response

if __name__ == "__main__":
    query = "What is the best approach for starting a business in Portugal?"  # Example query
    response = rag_pipeline(query)
    print("Code is correct!")
except Exception as e:
    print(f"Code is incorrect: {e}")
    print(rag_pipeline(query))

/*******  752c1ab3-c561-4959-aadf-a1c3d24646d0  *******/