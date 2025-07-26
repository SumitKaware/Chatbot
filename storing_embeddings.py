import os
from langchain_core.documents import Document as LangchainDocument # Alias to avoid conflict if needed
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma # Import ChromaDB
from model import embeddings, vector_db_setup

def store_embeddings_in_chromadb(processed_data_by_pdf: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """
    Stores the extracted and embedded content from multiple PDFs into ChromaDB.

    Args:
        processed_data_by_pdf (Dict[str, List[Dict[str, Any]]]): A dictionary
                                                                  where keys are PDF filenames
                                                                  and values are lists of extracted elements
                                                                  with their embeddings.
        output_dir (str): The base directory where the ChromaDB instances will be saved.
    """
    print("\n--- Storing embeddings in ChromaDB ---")
    if not processed_data_by_pdf:
        print("No processed data to store in ChromaDB.")
        return

    for pdf_file, extracted_data in processed_data_by_pdf.items():
        chroma_db_path = os.path.join(output_dir, os.path.splitext(pdf_file)[0], "chroma_db")
        os.makedirs(chroma_db_path, exist_ok=True)
        print(f"  - Initializing ChromaDB for {pdf_file} at '{chroma_db_path}'...")

        # Create Langchain Documents from the extracted elements for ChromaDB
        lc_documents_for_chroma = []
        for element in extracted_data:
            # Use the 'description' as page_content for embedding in Chroma
            # Add metadata to preserve context and original element info
            metadata = {
                "type": element["type"],
                "page_num": element["page_num"],
                "bbox": str(element["bbox"]), # Convert tuple to string for JSON compatibility in metadata
                "original_content_summary": element["original_content_summary"],
                "source_pdf": pdf_file
            }
            # Create LangchainDocument with pre-computed embedding
            lc_doc = LangchainDocument(
                page_content=element["description"],
                metadata=metadata
            )
            lc_documents_for_chroma.append(lc_doc)

        if lc_documents_for_chroma:
            vectorstore = vector_db_setup(lc_documents_for_chroma, persist_directory=chroma_db_path)
            # Persist the database to disk
            vectorstore.persist()
            print(f"  - Stored {len(lc_documents_for_chroma)} elements in ChromaDB for {pdf_file}.")
        else:
            print(f"  - No elements to store in ChromaDB for {pdf_file}.")
        print("-" * 50) # Small separator for ChromaDB processing

    print("\nAll embeddings stored in ChromaDB.")
    
def load_chroma_db_and_retriever(chroma_db_path: str):
    """
    Loads a persisted ChromaDB and creates a retriever from it.

    Args:
        chroma_db_path (str): The path to the persisted ChromaDB directory.

    Returns:
        langchain_core.vectorstores.VectorStoreRetriever: A retriever for the loaded ChromaDB,
                                                          or None if the database cannot be loaded.
    """
    if not os.path.exists(chroma_db_path):
        print(f"Error: ChromaDB not found at '{chroma_db_path}'. Please ensure it has been created and persisted.")
        return None

    print(f"\n--- Loading ChromaDB from '{chroma_db_path}' ---")
    try:
        # Load the persisted ChromaDB
        vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
        print("ChromaDB loaded successfully.")

        # Create a retriever to fetch relevant documents
        retriever = vectorstore.as_retriever()
        print("Retriever created.")
        return retriever
    except Exception as e:
        print(f"Error loading ChromaDB from '{chroma_db_path}': {e}")
        return None