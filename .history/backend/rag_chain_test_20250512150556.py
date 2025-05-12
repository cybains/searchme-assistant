import os
import re
import time
import random
import faiss
import numpy as np
import textwrap
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

# Download required NLTK data
nltk.download('punkt')

# Constants
DATA_FOLDER = "data/"
VECTORSTORE_PATH = "vectorstore/faiss.index"
DOCUMENTS_PATH = "vectorstore/documents.txt"
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'
MODEL_NAME = 'google/flan-t5-large'

# Load models
embed_model = SentenceTransformer(EMBEDDING_MODEL)
generator = pipeline('text2text-generation', model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load FAISS index (create if doesn't exist)
if os.path.exists(VECTORSTORE_PATH):
    index = faiss.read_index(VECTORSTORE_PATH)
else:
    index = faiss.IndexFlatL2(embed_model.get_sentence_embedding_dimension())

def clean_text(text):
    """Clean up the text by removing extra spaces and line breaks."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_and_chunk(text, chunk_size=500, overlap=100, tokenizer=None):
    """Preprocess and tokenize text into overlapping chunks based on token count."""
    assert tokenizer is not None, "Tokenizer must be provided."

    text = clean_text(text)
    sentences = sent_tokenize(text)

    chunks, current_chunk = [], []
    current_length = 0

    for sentence in sentences:
        tokenized = tokenizer.encode(sentence, add_special_tokens=False)
        token_length = len(tokenized)

        if current_length + token_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += token_length
        else:
            chunks.append(' '.join(current_chunk))

            if overlap > 0:
                prev_tokens = tokenizer.encode(' '.join(current_chunk), add_special_tokens=False)
                overlap_tokens = prev_tokens[-overlap:]
                overlap_text = tokenizer.decode(overlap_tokens, skip_special_tokens=True)
                current_chunk = [overlap_text, sentence]
                current_length = len(tokenizer.encode(' '.join(current_chunk), add_special_tokens=False))
            else:
                current_chunk = [sentence]
                current_length = token_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def embed_documents(documents):
    """Generate embeddings with instruction tuning (BGE)."""
    instruction = "Represent this document for retrieval: "
    formatted_docs = [instruction + doc for doc in documents]
    return embed_model.encode(formatted_docs, convert_to_numpy=True)

def build_faiss_index():
    """Process all documents in data/ and build FAISS index."""
    all_chunks = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_FOLDER, filename)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = preprocess_and_chunk(text, tokenizer=tokenizer)
                all_chunks.extend(chunks)

    print(f"📚 Total Chunks: {len(all_chunks)}")

    embeddings = embed_documents(all_chunks)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    # Save index
    faiss.write_index(faiss_index, VECTORSTORE_PATH)
    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk + "\n")
    print("✅ FAISS index and document chunks saved.")

def retrieve_documents(query, top_k=5):
    """Retrieve top-k relevant documents using FAISS and BGE formatting."""
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        all_chunks = f.readlines()

    query_instruction = "Represent this sentence for retrieval: "
    query_embedding = embed_model.encode([query_instruction + query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved_docs = []
    print("\n📚 Top Retrieved Chunks:")
    for idx in indices[0]:
        chunk = clean_text(all_chunks[idx])
        retrieved_docs.append(chunk)
        print(f"- {chunk[:100]}...")

    return retrieved_docs

def format_prompt(context_docs, user_query):
    context = "\n\n".join(context_docs)
    starters = [
        "You need", "To move forward, you need", "It’s essential to",
        "In order to proceed, you need", "The next step is",
        "Here’s what you should do", "You’ll want to make sure you have",
        "To get started, you need", "The required steps are"
    ]
    start = random.choice(starters)

    prompt = f"""
You are a solution finder based on the context provided.

Your task is to answer the user's question clearly and in detail, using the context provided.

Guidelines:
1. State the source of the information if available.
2. Be grammatically correct and comprehensive.
3. Provide lists when necessary (e.g., for required documents).

Context:
{context}

User Question: {user_query}

Answer: {start}
"""
    return prompt

def generate_response(query, documents, max_context=7):
    selected_docs = documents[:max_context]
    prompt = format_prompt(selected_docs, query)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    inputs = {k: v.to(generator.model.device) for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            max_length=800,
            num_beams=5,
            do_sample=True,
            temperature=0.7,
            early_stopping=True,
        )
    end_time = time.time()
    print(f"🕒 Response Generation Time: {end_time - start_time:.2f} seconds")

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def rag_pipeline(query):
    start_time = time.time()
    relevant_docs = retrieve_documents(query)
    response = generate_response(query, relevant_docs)
    end_time = time.time()
    print(f"⏱️ Total RAG Pipeline Time: {end_time - start_time:.2f} seconds")
    return response

# === Build index once if needed ===
if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(DOCUMENTS_PATH):
    print("⚙️ Building FAISS index from scratch...")
    build_faiss_index()
    index = faiss.read_index(VECTORSTORE_PATH)

# === Run test queries ===
if __name__ == "__main__":
    test_queries = [
        "Can I apply for a residence permit if I'm doing volunteer work in Portugal?",
        "Do I need a host family confirmation for this permit?",
        "Can I get a residence permit for an internship in Portugal?",
        "Will a criminal record from my country affect my application?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n=== Test #{i}: {query} ===")
        response = rag_pipeline(query)
        print(f"\n🧠 Response:\n{response}\n")

# Testing with sample queries
if __name__ == "__main__":
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n=== Test #{i}: {query} ===")
        response = rag_pipeline(query)
        print(f"\n🧠 Response:\n{response}\n")
