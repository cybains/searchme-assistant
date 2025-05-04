import time
from rag_chain import rag_pipeline  # Import your RAG pipeline function

def test_rag_pipeline():
    query = "What is the validity of a resident permit for students?"
    print(f"Testing query: {query}")
    
    start_time = time.time()
    response = rag_pipeline(query)
    end_time = time.time()
    
    print("\n--- Response from RAG pipeline ---")
    print(response)
    print(f"\nExecution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    test_rag_pipeline()
