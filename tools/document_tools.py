from langchain_community.document_loaders import SimpleDirectoryReader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader

#from langchain_community.llms import OpenAI.

def load_documents_from_directory(directory_path):
    """
    Load all documents from a given directory.
    """
    loader = SimpleDirectoryReader(directory_path)
    #loader = DirectoryLoader(directory_path)

    return loader.load_data()

def parse_and_store_in_chroma(documents, chroma_db_path):
    """
    Parse the documents into embeddings and store them in Chroma DB.
    """
    embedding_model = OpenAIEmbeddings()  #class is part of the langchain.embeddings module and is used to convert text data into numerical embeddings. 
    texts = [doc.page_content for doc in documents]
    metadatas = [{"source": f"Document {i+1}"} for i in range(len(texts))]
    
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
    vectorstore.add_texts(texts, metadatas)
    vectorstore.persist()
    return f"Embeddings successfully stored in Chroma DB at {chroma_db_path}."

def process_and_store_from_directory(directory_path, chroma_db_path):
    """
    Combines directory loading, parsing, and storing embeddings in Chroma DB.
    """
    documents = load_documents_from_directory(directory_path)
    return parse_and_store_in_chroma(documents, chroma_db_path)
