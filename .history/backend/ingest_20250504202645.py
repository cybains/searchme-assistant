from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load your document
def load_text():
    loader = TextLoader("./data/a.txt")  # Adjust the path if needed
    documents = loader.load()
    return documents

# Function to embed documents and store in FAISS
def embed_and_store():
    # Load documents
    documents = load_text()

    # Initialize OpenAI Embeddings model
    embeddings = OpenAIEmbeddings()

    # Create a vector store from the documents using FAISS
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save the vector store to the 'vectorstore' directory
    vector_store.save_local("./vectorstore")

if __name__ == "__main__":
    embed_and_store()
