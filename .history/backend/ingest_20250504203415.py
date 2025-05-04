import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import faiss

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# Ensure the OpenAI API key is loaded
if openai_api_key is None:
    raise ValueError("OpenAI API key is missing. Please add it to the .env file.")

# Load your text data (replace with your actual file path)
text_file_path = './data/a.txt'

# Load the text file using the TextLoader
def load_text(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

# Function to embed and store the content in FAISS index
def embed_and_store():
    # Load the text documents
    documents = load_text(text_file_path)
    
    # Initialize OpenAI embeddings using the OpenAI API key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create a FAISS index from the documents
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save the FAISS index to the vectorstore directory
    vectorstore.save_local('./vectorstore')
    print("FAISS index saved to './vectorstore'.")

# Main function to run the embedding process
if __name__ == '__main__':
    embed_and_store()
