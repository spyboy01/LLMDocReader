import os
from agents.parsing_and_stroring_embeddings import run_agent1

if __name__ == "__main__":
    directory_path = "./data"  
    chroma_db_path = "./chroma_db"  

    # Check if the data directory exists
    if not os.path.exists(directory_path):
        print(f"Error: The directory {directory_path} does not exist.")
    else:
        # Run Agent 1 to process and store embeddings
        print("Running Agent 1: Embedding Storage from Directory")
        response1 = run_agent1(directory_path, chroma_db_path)
        print(response1)
