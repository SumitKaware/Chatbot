import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Union
import operator
from model import llm, embeddings  # Importing the LLM and embeddings from model.py
from pdfloader import vector_db
from storing_embeddings import load_chroma_db_and_retriever  # Importing raw documents from pdfloader.py
#
# Create a vector database
#vector_db()  # Initialize the vector store with a sample query


# --- Demonstrate loading ChromaDB and using retriever ---
print("\n--- Demonstrating ChromaDB Loading and Retrieval ---")
current_script_dir = os.getcwd()
output_embedding_directory = os.path.join(current_script_dir, "embedded_content")
sample_chroma_path = os.path.join(output_embedding_directory, "chroma_db")

retriever = load_chroma_db_and_retriever(sample_chroma_path)
# query = "What is the profit for Q4?"
# print(f"\nQuerying ChromaDB: '{query}'")
# try:
#     retrieved_docs = retriever.invoke(query)
# except Exception as e:
#     print(f"Error querying ChromaDB: {e}")
#     retrieved_docs = []
# if retriever:
#     query = "What is the amount allocated for education?"
#     print(f"\nQuerying ChromaDB: '{query}'")
    
#     retrieved_docs = retriever.invoke(query)

#     print("\n--- Retrieved Documents ---")
#     for i, doc in enumerate(retrieved_docs):
#         print(f"Document {i+1} (Type: {doc.metadata.get('type')}, Page: {doc.metadata.get('page_num')}):")
#         print(f"  Content: {doc.page_content[:150]}...") # Print first 150 chars of content
#         print(f"  Metadata: {doc.metadata}\n")
# else:
#     print("Could not load ChromaDB for demonstration.")



# --- 3. LangGraph Agent Definition ---

# Define the state of our graph
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        chat_history: A list of messages forming the conversation history.
        documents: A list of retrieved documents.
        generation: The generated response from the LLM.
    """
    question: str
    chat_history: Annotated[List[BaseMessage], operator.add]
    documents: List[Document]
    generation: str

# Define the nodes (functions) in our graph

def retrieve(state: AgentState):
    """
    Retrieves documents based on the user's question.
    """
    print("---RETRIEVE NODE---")
    question = state["question"]
    if retriever:
        documents = retriever.invoke(question)
        print(f"Retrieved {documents} documents for the question: '{question}'")
        return {"documents": documents, "question": question, "chat_history": state["chat_history"]}
    else:
        return "Could not load ChromaDB for demonstration."
    

def generate(state: AgentState):
    """
    Generates a response using the LLM, incorporating retrieved documents.
    """
    print("---GENERATE NODE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    # Create a prompt for the LLM
    # We include chat history and retrieved context
    context = "\n".join([doc.page_content for doc in documents])
    prompt = f"""
    You are a helpful AI assistant. Answer the user's question based on the provided context and chat history.
    If the answer is not in the context, state that you don't know.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question: {question}
    Answer:
    """
    print(f"Prompting LLM with: \n{prompt[:200]}...") # Print a snippet of the prompt
    response = llm.invoke(prompt)
    return {"generation": response.content, "question": question, "documents": documents, "chat_history": chat_history}

# Build the LangGraph
workflow = StateGraph(AgentState)

# Add nodes to the workflow
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Set the entry point
workflow.set_entry_point("retrieve")

# Define the edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()
print("LangGraph agent compiled.")

# --- 4. Chat Loop ---

print("\n--- Chatbot Ready! Type 'exit' to quit. ---")
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Exiting chatbot. Goodbye!")
        break

    # Add user's message to chat history
    chat_history.append(HumanMessage(content=user_input))

    # Invoke the LangGraph agent
    try:
        # The agent expects the initial state.
        # We pass the current chat_history to ensure it's propagated.
        inputs = {"question": user_input, "chat_history": chat_history}
        result = app.invoke(inputs)

        # Get the AI's response from the 'generation' field in the final state
        ai_response = result["generation"]
        print(f"Bot: {ai_response}")

        # Update chat history with AI's response for the next turn
        chat_history.append(HumanMessage(content=ai_response)) # LangChain's HumanMessage is used here for simplicity, but AIMessage is more appropriate for bot responses. For this example, it still works.

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your GOOGLE_API_KEY is correctly set.")
        chat_history.pop() # Remove the last user message if an error occurred to avoid polluting history
