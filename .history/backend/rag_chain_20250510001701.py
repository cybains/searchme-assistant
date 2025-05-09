import os
import re
import faiss
import numpy as np
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

# Define constants for your directory and model
DATA_FOLDER = "data/"  # Folder where your documents are located
VECTORSTORE_PATH = "vectorstore/faiss.index"  # Path to your FAISS index
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
MODEL_NAME = 'facebook/bart-base'

# Load models
embed_model = SentenceTransformer(EMBEDDING_MODEL)
generator = pipeline('text2text-generation', model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load FAISS index
index = faiss.read_index(VECTORSTORE_PATH)



def preprocess_and_chunk(text, chunk_size=500, overlap=100):
    # Split into sentences (rough but effective)
    sentences = re.split(r'(?<=[.!?]) +', text.strip())

    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap between chunks
    overlapped_chunks = []
    for i in range(0, len(chunks), 1):
        chunk = ' '.join(chunks[max(0, i - 1):i + 1])  # Previous + current
        if chunk not in overlapped_chunks:
            overlapped_chunks.append(chunk)

    return overlapped_chunks


def embed_documents(documents):
    embeddings = embed_model.encode(documents, convert_to_tensor=True)
    return embeddings


def retrieve_documents(query, top_k=5):
    with open("vectorstore/documents.txt", "r", encoding="utf-8") as f:
        all_chunks = f.readlines()

    query_embedding = embed_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Logging retrieved chunks
    print("\n🔍 Top Retrieved Chunks for Query: '{}'".format(query))
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        print(f"\n#{rank} [Index {idx}] (Distance: {dist:.4f}):")
        print(all_chunks[idx].strip())

    documents = [all_chunks[idx].strip() for idx in indices[0]]
    return documents


def generate_response(query, documents, max_context=3):
    context = " ".join(documents[:max_context])
    input_text = f"Context: {context}\n\nQuestion: {query}"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(generator.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = generator.model.generate(**inputs, max_length=200)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def rag_pipeline(query):
    relevant_docs = retrieve_documents(query)
    response = generate_response(query, relevant_docs)
    return response


if __name__ == "__main__":
    query = "What is the validity of a resident permit for students?"
    print("\n🧠 Response:\n", rag_pipeline(query))
