# this is agent1 : This agent handles parsing documents and storing embeddings in Chroma DB.

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from tools.document_tools import process_and_store

# Define the tool for embedding storage
embedding_tool = Tool(
    name="EmbeddingStore",
    func=lambda file_path, chroma_db_path: process_and_store(file_path, chroma_db_path),
    description="Parse documents, convert to embeddings, and store in Chroma DB."
)

# Initialize the agent
llm = OpenAI(temperature=0)
agent = initialize_agent(tools=[embedding_tool], llm=llm, agent="zero-shot-react-description")

def run_agent1(file_path, chroma_db_path):
    """
    Run Agent 1 to process and store embeddings.
    """
    response = agent.run(f"Store the document embeddings from {file_path} into Chroma DB at {chroma_db_path}.")
    return response
