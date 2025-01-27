from langchain.document_loaders import PyPDFLoader, TextLoader, DocxLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def load_document(file_path):
    """
    Load the document based on its file type.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = DocxLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF, TXT, or DOCX file.")
    return loader.load()

def parse_and_store_in_chroma(documents, chroma_db_path):
    """
    Parse the documents into embeddings and store them in Chroma DB.
    """
    embedding_model = OpenAIEmbeddings()
    texts = [doc.page_content for doc in documents]
    metadatas = [{"source": f"Page {i+1}"} for i in range(len(texts))]
    
    vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
    vectorstore.add_texts(texts, metadatas)
    vectorstore.persist()
    return f"Embeddings successfully stored in Chroma DB at {chroma_db_path}."

def process_and_store(file_path, chroma_db_path):
    """
    Combines document loading, parsing, and storing embeddings in Chroma DB.
    """
    documents = load_document(file_path)
    return parse_and_store_in_chroma(documents, chroma_db_path)
