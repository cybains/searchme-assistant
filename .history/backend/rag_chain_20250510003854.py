from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.llms import HuggingFacePipeline
import os

# -----------------------------
# 1. Load and Embed Documents
# -----------------------------
data_path = "./data"
documents = []

# Load plain .txt files from your data folder
for filename in os.listdir(data_path):
    if filename.endswith(".txt"):
        with open(os.path.join(data_path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(Document(page_content=text, metadata={"source": filename}))

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# Embed using all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
db = FAISS.from_documents(split_docs, embedding_model)

# -----------------------------
# 2. Define the Retriever
# -----------------------------
retriever = db.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# 3. Load BART Model
# -----------------------------
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Create generation pipeline
generator = pipeline("text2text-generation", model=bart_model, tokenizer=bart_tokenizer)

# Wrap in LangChain-compatible LLM
llm = HuggingFacePipeline(pipeline=generator)

# -----------------------------
# 4. Define Prompt Template
# -----------------------------
template = """You are a helpful assistant. Use ONLY the information in the context to answer the question clearly.

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
llm_chain = LLMChain(llm=llm, prompt=prompt)

# StuffDocumentsChain: pass context into LLMChain
rag_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

# -----------------------------
# 5. Ask a Question
# -----------------------------
query = "What is the validity of a resident permit for students?"

# Retrieve relevant docs
docs = retriever.get_relevant_documents(query)

# Run RAG chain
final_answer = rag_chain.run(input_documents=docs, question=query)

# Output
print("\n🔍 Top Documents Retrieved:\n")
for i, doc in enumerate(docs):
    print(f"#{i+1} [{doc.metadata.get('source', 'unknown')}]:\n{doc.page_content[:300]}...\n")

print("\n🧠 Final Synthesized Answer:\n")
print(final_answer)
