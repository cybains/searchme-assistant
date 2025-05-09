import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data/"
VECTORSTORE_PATH = "vectorstore/faiss.index"
DOCSTORE_PATH = "vectorstore/documents.txt"

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


all_chunks = []

# Step 1: Read and chunk documents
for filename in os.listdir(DATA_FOLDER):
    filepath = os.path.join(DATA_FOLDER, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        chunks = preprocess_and_chunk(text)
        all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")

# Step 2: Embed chunks
embeddings = embed_model.encode(all_chunks, convert_to_tensor=False)

# Step 3: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 4: Save FAISS index and corresponding chunks
faiss.write_index(index, VECTORSTORE_PATH)
with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(chunk + "\n")

print("Indexing complete and saved.")
