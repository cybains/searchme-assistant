from rag_chain import rag_pipeline  # Import your RAG pipeline function

def test_rag_pipeline():
    # Example query
    query = "What is the validity of a resident permit for students?"
    
    # Run the pipeline
    response = rag_pipeline(query)
    
    # Print the response
    print("Response from RAG pipeline:", response)

if __name__ == "__main__":
    test_rag_pipeline()
