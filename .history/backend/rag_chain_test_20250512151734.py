import time
import os
import re
import random
import faiss
import numpy as np
import textwrap
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

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

# Load FAISS index
index = faiss.read_index(VECTORSTORE_PATH)

def clean_text(text):
    """Clean up the text by removing extra spaces and line breaks."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_and_chunk(text, chunk_size=500, overlap=100):
    """Preprocess the text and split it into overlapping chunks."""
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

    overlapped_chunks = []
    for i in range(0, len(chunks), 1):
        chunk = ' '.join(chunks[max(0, i - 1):i + 1])
        if chunk not in overlapped_chunks:
            overlapped_chunks.append(chunk)
    return overlapped_chunks

def embed_documents(documents):
    """Generate embeddings with instruction tuning (BGE)."""
    instruction = "Represent this document for retrieval: "
    formatted_docs = [instruction + doc for doc in documents]
    return embed_model.encode(formatted_docs, convert_to_tensor=True)

def retrieve_documents(query, top_k=5):
    """Retrieve top-k relevant documents using FAISS and BGE formatting."""
    start_time = time.time()

    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        all_chunks = f.readlines()

    query_instruction = "Represent this sentence for retrieval: "
    query_embedding = embed_model.encode([query_instruction + query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved_docs = []
    print("\n📚 Top Retrieved Chunks:")
    for idx in indices[0]:
        chunk = clean_text(all_chunks[idx])
        retrieved_docs.append(chunk)
        print(f"- {chunk[:100]}...")  # Preview first 100 characters

    end_time = time.time()
    print(f"🔎 Document Retrieval Time: {end_time - start_time:.2f} seconds")

    return retrieved_docs

def format_prompt(context_docs, user_query):
    """Format the prompt using retrieved context and user query."""
    context = "\n\n".join(context_docs)
    response_starters = [
        "You need", "To move forward, you need", "It’s essential to",
        "In order to proceed, you need", "The next step is",
        "Here’s what you should do", "You’ll want to make sure you have",
        "To get started, you need", "The required steps are"
    ]
    response_start = random.choice(response_starters)

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

Answer: {response_start}
"""
    return prompt

def generate_response(query, documents, max_context=5):
    """Generate an answer using the retrieved context and FLAN-T5."""
    selected_docs = documents[:max_context]
    prompt = format_prompt(selected_docs, query)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    inputs = {k: v.to(generator.model.device) for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            max_length=500,
            num_beams=5,
            do_sample=True,
            temperature=0.7,
            early_stopping=True,
        )
    end_time = time.time()
    print(f"🕒 Response Generation Time: {end_time - start_time:.2f} seconds")

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def rag_pipeline(query):
    """Full RAG pipeline from query to answer."""
    start_time = time.time()
    relevant_docs = retrieve_documents(query)
    response = generate_response(query, relevant_docs)
    end_time = time.time()
    print(f"⏱️ Total RAG Pipeline Time: {end_time - start_time:.2f} seconds")
    return response

# Testing with sample queries
if __name__ == "__main__":
    test_queries = [
        "Can I apply for a residence permit if I'm doing volunteer work in Portugal?",
    "What documents are required for a residence permit for volunteering?",
    "Is the volunteer work permit valid for more than one year?",
    "Can a volunteer residence permit be renewed?",
    "Do I need health insurance to apply for a volunteering residence permit?",
    "What must be included in the volunteer contract?",
    "Can I be paid for the volunteer work under this permit?",
    "Is a criminal record check from my home country required?",
    "Do I need to show proof of accommodation to apply for this permit?",
    "Can someone already in Portugal legally apply for a volunteer residence permit?",
    "Can this residence permit be denied if I have a security alert in the SIS?",
    "Can high school students get a residence permit for exchange programs in Portugal?",
    "What documents are needed for a high school mobility residence permit?",
    "Is parental consent required for minors in exchange programs?",
    "Can I stay in Portugal after my high school exchange ends?",
    "Is this permit renewable after the program ends?",
    "Do I need a host family confirmation for this permit?",
    "Does the high school permit require proof of health insurance?",
    "What kind of proof of accommodation is needed for high school exchange students?",
    "Do I need to provide the name and address of my host family?",
    "Can this residence permit be applied for while already in Portugal?",
    "Can I get a residence permit for an internship in Portugal?",
    "What documents are needed for an internship residence permit?",
    "Is the internship supposed to be unpaid to qualify for the permit?",
    "Can I apply for an internship residence permit from inside Portugal?",
    "Is the internship residence permit valid for more than one year?",
    "Can I renew my internship permit if my program continues?",
    "What must be included in the internship contract?",
    "Do I need to show financial means for an internship permit?",
    "Do I need a Portuguese criminal record check for this permit?",
    "Will a criminal record from my country affect my application?",
    "Can I still apply if I don't have a visa but entered Portugal legally?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n=== Test #{i}: {query} ===")
        response = rag_pipeline(query)
        print(f"\n🧠 Response:\n{response}\n")
