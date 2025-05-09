import os
import faiss
import numpy as np
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

# Define constants for your directory and model
DATA_FOLDER = "../data/"  # Folder where your documents are located
VECTORSTORE_PATH = "vectorstore/faiss.index"  # Path to your FAISS index
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
def preprocess_and_chunk(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        cleaned = chunk.replace('\n', ' ').strip()
        if cleaned:
            chunks.append(cleaned)
    return chunks


# Initialize the Hugging Face model for text generation
generator = pipeline('text2text-generation', model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
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
# Retrieve the most relevant documents from FAISS index
def retrieve_documents(query, top_k=5):
    # Load document chunks
    with open("vectorstore/documents.txt", "r", encoding="utf-8") as f:
        all_chunks = f.readlines()

    query_embedding = embed_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)

    documents = [all_chunks[idx].strip() for idx in indices[0]]
    return documents



def generate_response(query, documents):
    # Combine documents into one context string
    context = " ".join(documents)

    # Build full input
    input_text = context + " " + query

    # Tokenize with truncation
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,  # BART's max
    )

    # Move to correct device
    inputs = {k: v.to(generator.model.device) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        outputs = generator.model.generate(**inputs, max_length=150)

    # Decode result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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

