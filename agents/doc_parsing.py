from tools.document_tools import load_document

def run_agent2(file_path):
    """
    Run Agent 2 to parse and display document content.
    """
    documents = load_document(file_path)
    result = []
    for i, doc in enumerate(documents):
        result.append(f"Page {i + 1}: {doc.page_content}")
    return "\n".join(result)
