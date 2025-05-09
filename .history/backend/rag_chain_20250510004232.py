import os
import re
import faiss
import numpy as np
import textwrap
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

# Constants
DATA_FOLDER = "data/"
VECTORSTORE_PATH = "vectorstore/faiss.index"
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
MODEL_NAME = 'google/flan-t5-large'  # For example

# Load models
embed_model = SentenceTransformer(EMBEDDING_MODEL)
generator = pipeline('text2text-generation', model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load FAISS index
index = faiss.read_index(VECTORSTORE_PATH)

# CrossEncoder for ranking chunks
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_and_chunk(text, chunk_size=500, overlap=100):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlaps
    overlapped_chunks = []
    for i in range(0, len(chunks), 1):
        chunk = ' '.join(chunks[max(0, i - 1):i + 1])
        if chunk not in overlapped_chunks:
            overlapped_chunks.append(chunk)
    return overlapped_chunks

def embed_documents(documents):
    return embed_model.encode(documents, convert_to_tensor=True)

def retrieve_documents(query, top_k=5):
    with open("vectorstore/documents.txt", "r", encoding="utf-8") as f:
        all_chunks = f.readlines()

    query_embedding = embed_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)

    print(f"\n🔍 Top Retrieved Chunks for Query: '{query}'\n")
    retrieved_docs = []
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        chunk = clean_text(all_chunks[idx])
        print(f"# {i + 1} [Index {idx}] (Distance: {score:.4f}):\n{textwrap.fill(chunk, width=100)}\n")
        retrieved_docs.append(chunk)
    return retrieved_docs

def filter_chunks(chunks, query_keywords):
    """
    Filters out irrelevant chunks by checking for keyword overlap.
    """
    filtered = []
    for chunk in chunks:
        if any(keyword.lower() in chunk.lower() for keyword in query_keywords):
            filtered.append(chunk)
    return filtered or chunks  # Fallback: if nothing matches, use all

def rerank_chunks(chunks, query):
    """
    Reranks chunks using a cross-encoder model.
    """
    pairs = [[query, chunk] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    sorted_chunks = [x for _, x in sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)]
    return sorted_chunks

def format_prompt(context_docs, user_query):
    context = "\n\n".join(context_docs)
    prompt = f"""You are an assistant answering user questions based on Portuguese immigration laws. 
You are not an official of the Ministry of Education.

Answer the following question strictly based on the context provided. Be clear and avoid generic responses.

Context:
{context}

Question: {user_query}

Answer:"""
    return prompt

def generate_response(query, documents, max_context=3):
    selected_docs = documents[:max_context]
    prompt = format_prompt(selected_docs, query)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(generator.model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            max_length=300,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def rag_pipeline(query):
    relevant_docs = retrieve_documents(query)
    
    # Keywords for filtering relevant chunks
    keywords = ["student", "residence permit", "higher education", "article 91", "study programme"]
    filtered_docs = filter_chunks(relevant_docs, keywords)
    
    # Optional: Rerank the filtered chunks using the cross-encoder
    reranked_docs = rerank_chunks(filtered_docs, query)
    
    return generate_response(query, reranked_docs)

if __name__ == "__main__":
    query = "What is the validity of a resident permit for students?"
    print("\n🧠 Response:\n", rag_pipeline(query))
