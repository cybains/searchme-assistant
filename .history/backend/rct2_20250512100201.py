from transformers import BartForConditionalGeneration, BartTokenizer
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import torch
import time

# 1. Load models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
generator_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# 2. Load vector store
vectorstore = FAISS.load_local("faiss_index", embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Define a custom generation function
def generate_answer_bart(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        summary_ids = generator_model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 4. Ask question
def ask_question(question: str):
    print(f"\n=== Test: {question} ===")

    # Step 1: Retrieve top docs
    start = time.time()
    docs = retriever.get_relevant_documents(question)
    retrieval_time = time.time() - start

    # Show top docs
    print("\n📚 Top Retrieved Chunks:")
    for doc in docs:
        print(f"- {doc.page_content[:300]}...")

    # Step 2: Format context
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question clearly and naturally, starting with 'You need'. Use only the relevant facts from the following context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"

    # Step 3: Generate answer
    start_gen = time.time()
    response = generate_answer_bart(prompt)
    response_time = time.time() - start_gen

    print(f"\n🔎 Document Retrieval Time: {retrieval_time:.2f} seconds")
    print(f"🕒 Response Generation Time: {response_time:.2f} seconds")
    print(f"⏱️ Total RAG Pipeline Time: {retrieval_time + response_time:.2f} seconds\n")
    print(f"🧠 Response:\n{response.strip()}\n")

# Example usage
questions = [
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
    "Can high school students get a residence permit for exchange programs in Portugal?"
]

for q in questions:
    ask_question(q)
