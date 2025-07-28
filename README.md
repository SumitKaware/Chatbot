# Chatbot

![ChatbotFD](https://github.com/user-attachments/assets/5659891b-33be-4478-9693-c4bcd2cf05c2)

Step-by-step description of the system's operation:

Phase 1: Document Ingestion and Indexing

1. Data Storage: Raw data, including documents, is stored in a Data Storage system.

2. PDF: Specific PDF documents are retrieved from the Data Storage for processing.

3. Embedding/Parsing Pipeline: The content of the PDFs is fed into an Embedding/Parsing Pipeline. This pipeline is responsible for:

    Parsing the PDF content (extracting text, tables, images).

    Chunking the extracted content into smaller, manageable pieces.

    Generating numerical representations (embeddings) for each chunk using an embedding model.

4. Vector DB: The generated embeddings, along with associated metadata and original content, are stored in a Vector DB (Vector Database). This database allows for efficient similarity searches based on the embeddings.

Phase 2: Query Processing and Response Generation (Retrieval-Augmented Generation - RAG)

1. Client: A user interacts with the system through a Client application, typically by submitting a query or question.

2. Retriever (Query Initiation): The client sends its query to the Retriever component.

3. Retriever (Vector DB Interaction): The Retriever takes the user's query, converts it into an embedding (usually using the same embedding model as in Phase 1, though not explicitly shown as a separate box in this part of the diagram), and performs a similarity search against the Vector DB.

4. Vector DB (Retrieval): The Vector DB identifies and returns the most semantically relevant document chunks (and their associated original content/metadata) to the Retriever based on the query's embedding.

5. Retriever (Context Provision): The Retriever gathers the retrieved relevant information (context) and prepares it to be sent to the Language Model.

6. Generating LLM: The Retriever sends the user's original query along with the retrieved relevant context to the Generating LLM (Large Language Model). The LLM uses this provided context to formulate an informed and accurate response, reducing the likelihood of hallucinations.

7. Client (Response Display): The Generating LLM sends its generated response back to the Client, which then displays it to the user.
