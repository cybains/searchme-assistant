from langchain.document_loaders import TextLoader

# Path to your text file in the 'data' folder
file_path = './data/a.txt'

# Load the document using LangChain's TextLoader
def load_text():
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    return documents

# Example usage
if __name__ == "__main__":
    text_documents = load_text()
    print(text_documents)  # Check the loaded document
