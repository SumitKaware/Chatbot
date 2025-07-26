import os
import pdfplumber
import pandas as pd
import fitz # PyMuPDF for image extraction
from PIL import Image # Pillow for image manipulation
import io
import base64
import json
import operator

# LangChain imports for embeddings and multimodal LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document as LangchainDocument # Alias to avoid conflict if needed
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma # Import ChromaDB
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import the text splitter

# --- 1. Configuration and Setup ---

# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google API Key or set it as an environment variable.
# You can get an API key from Google AI Studio: https://aistudio.google.com/app/apikey
# It's recommended to set it as an environment variable: export GOOGLE_API_KEY="your_api_key_here"
# api_key = os.getenv("GOOGLE_API_KEY", "")
# if not api_key:
#     print("Warning: GOOGLE_API_KEY environment variable not set. Attempting to use global __api_key__.")
#     api_key = globals().get('__api_key__', '')
#     if not api_key:
#         raise ValueError("Google API Key is not set. Please set GOOGLE_API_KEY environment variable or provide it.")
from google.colab import userdata
api_key = userdata.get('GOOGLE_API_KEY_1')
# Initialize embedding model (for text and table descriptions)
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
# Initialize multimodal LLM (for image descriptions)
vision_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def get_image_description_embedding(image_bytes: bytes) -> Dict[str, Any]:
    """
    Generates a description for an image using a multimodal LLM and then
    creates an embedding for that description.
    """
    try:
        # Convert image bytes to base64 for LLM input
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Prompt the multimodal LLM to describe the image
        print("    - Generating description for image...")
        response = vision_llm.invoke([
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": "Describe this image concisely and informatively. Focus on key objects, text, or concepts. If it's a diagram or chart, mention its type. Keep it under 100 words."}
                ]
            )
        ])
        description = response.content
        print(f"    - Image description generated (first 50 chars): {description[:50]}...")

        # Embed the generated description
        embedding = embeddings_model.embed_query(description)
        print("    - Image description embedded.")
        return {"description": description, "embedding": embedding}
    except Exception as e:
        print(f"    - Error generating image description/embedding: {e}")
        return {"description": "Error generating description.", "embedding": []}

def get_text_embedding(text_content: str) -> List[float]:
    """
    Generates an embedding for the given text content.
    """
    try:
        embedding = embeddings_model.embed_query(text_content)
        return embedding
    except Exception as e:
        print(f"    - Error generating text embedding: {e}")
        return []

def get_table_embedding(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Converts a DataFrame to a Markdown table string, generates a description,
    and then creates an embedding for that description.
    """
    try:
        # Convert DataFrame to Markdown string for LLM readability
        table_markdown = df.to_markdown(index=False)
        # Create a concise description of the table's content/structure
        description = f"Table with columns: {', '.join(df.columns.tolist())}. First few rows:\n{table_markdown.splitlines()[1:min(5, len(table_markdown.splitlines()))+1]}"
        description = f"A table with {len(df.columns)} columns and {len(df)} rows. Columns are: {', '.join(df.columns.tolist())}. Sample data:\n{df.head(2).to_string(index=False)}"

        embedding = embeddings_model.embed_query(description)
        print("    - Table description and embedding generated.")
        return {"description": description, "embedding": embedding}
    except Exception as e:
        print(f"    - Error generating table description/embedding: {e}")
        return {"description": "Error generating table description.", "embedding": []}


def extract_and_embed_pdf(pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Extracts text blocks, tables, and images from a PDF, orders them sequentially,
    generates embeddings, and saves the structured data.

    Args:
        pdf_path (str): The full path to the PDF file.
        output_dir (str): The base directory to save extracted content and embeddings.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing an extracted
                              element with its content, description, and embedding,
                              ordered by its appearance in the PDF.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_path = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_output_path, exist_ok=True)

    extracted_elements_with_embeddings = []

    print(f"\n--- Processing PDF: {pdf_name} ---")

    try:
        with pdfplumber.open(pdf_path) as pdf_plumber_doc, fitz.open(pdf_path) as fitz_doc:
            num_pages = len(pdf_plumber_doc.pages)
            print(f"  Total pages detected: {num_pages}")

            for page_num in range(num_pages):
                try: # Granular error handling for each page
                    print(f"  Processing Page {page_num + 1}...")
                    plumber_page = pdf_plumber_doc.pages[page_num]
                    fitz_page = fitz_doc[page_num]

                    page_elements = [] # To store elements for current page before sorting

                    # --- Extract Text Blocks and Split ---
                    page_text_content = plumber_page.extract_text(keep_blank_chars=False, layout=True)
                    if page_text_content:
                        # Create a LangchainDocument for the entire page text
                        # Use the full page bbox for all text chunks from this page for sorting
                        page_text_doc = LangchainDocument(
                            page_content=page_text_content,
                            metadata={"page_num": page_num + 1, "source_pdf": pdf_name, "bbox": (0, 0, plumber_page.width, plumber_page.height)}
                        )
                        # Split the page text into smaller chunks
                        text_chunks = text_splitter.split_documents([page_text_doc])
                        for chunk_idx, chunk in enumerate(text_chunks):
                            # For simplicity in ordering, we'll use the page's full bbox for all text chunks.
                            # For more precise spatial RAG, you'd need to derive chunk-specific bboxes,
                            # which is more complex and often requires a different parsing approach.
                            page_elements.append({
                                "type": "text",
                                "content": chunk.page_content,
                                "bbox": (0, 0, plumber_page.width, plumber_page.height), # Use page bbox for ordering
                                "page_num": page_num + 1,
                                "chunk_idx": chunk_idx # Add chunk index for unique identification
                            })

                    # --- Extract Tables ---
                    tables = plumber_page.extract_tables()
                    for table_idx, table_data in enumerate(tables):
                        if table_data:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            # Get table bounding box from pdfplumber's table object
                            table_bbox = plumber_page.find_tables()[table_idx].bbox
                            page_elements.append({
                                "type": "table",
                                "content": df,
                                "bbox": table_bbox,
                                "page_num": page_num + 1
                            })

                    # --- Extract Images ---
                    for img_idx, img_info in enumerate(fitz_page.get_images(full=True)):
                        xref = img_info[0]
                        base_image = fitz_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Get image bounding box (x0, y0, x1, y1)
                        plumber_images = plumber_page.images
                        img_bbox = None
                        # Try to find matching image bbox from pdfplumber's detected images
                        # This is an approximation as PyMuPDF and pdfplumber might detect images differently.
                        for p_img in plumber_images:
                            # Simple check: if image is on the same page and has similar dimensions/position
                            # A more robust check might involve comparing image hashes or exact coordinates.
                            if p_img['page_number'] == page_num + 1 and \
                               abs(p_img['width'] - img_info[2]) < 5 and \
                               abs(p_img['height'] - img_info[3]) < 5: # Compare width/height
                                img_bbox = (p_img['x0'], p_img['y0'], p_img['x1'], p_img['y1'])
                                break

                        if img_bbox:
                            page_elements.append({
                                "type": "image",
                                "content": image_bytes,
                                "bbox": img_bbox,
                                "page_num": page_num + 1,
                                "ext": image_ext
                            })
                        else:
                            print(f"    - Warning: Could not find precise bbox for image {img_idx+1} on page {page_num+1}. Skipping for ordering.")
                            continue # Skip if we can't get a bbox for ordering

                    # --- Sort elements by their vertical position (top-left y-coordinate) ---
                    # This ensures sequential order on the page
                    page_elements.sort(key=lambda x: x["bbox"][1]) # Sort by y0 (top of bounding box)

                    print(f"    - Found {len(page_elements)} elements on Page {page_num + 1} before embedding.")

                    # --- Generate Embeddings for Sorted Elements ---
                    for element_idx, element in enumerate(page_elements):
                        print(f"    - Processing {element['type']} element {element_idx + 1} on page {element['page_num']}...")
                        processed_element = {
                            "type": element["type"],
                            "page_num": element["page_num"],
                            "bbox": element["bbox"],
                            "original_content_summary": "" # A short summary of original content
                        }

                        if element["type"] == "text":
                            text_content = element["content"]
                            processed_element["original_content_summary"] = text_content[:200] + "..." if len(text_content) > 200 else text_content
                            processed_element["embedding"] = get_text_embedding(text_content)
                            processed_element["description"] = text_content # For text, description is the text itself
                            # Optionally save text to file
                            text_filename = os.path.join(pdf_output_path, f"page_{element['page_num']}_text_chunk_{element['chunk_idx']}.txt")
                            with open(text_filename, "w", encoding="utf-8") as f:
                                f.write(text_content)

                        elif element["type"] == "table":
                            df = element["content"]
                            table_result = get_table_embedding(df)
                            processed_element["description"] = table_result["description"]
                            processed_element["embedding"] = table_result["embedding"]
                            processed_element["original_content_summary"] = df.head(2).to_string(index=False) # First 2 rows summary
                            # Save table to CSV
                            table_filename = os.path.join(pdf_output_path, f"page_{element['page_num']}_table_{element_idx + 1}.csv")
                            df.to_csv(table_filename, index=False)

                        elif element["type"] == "image":
                            image_bytes = element["content"]
                            image_ext = element["ext"]
                            image_result = get_image_description_embedding(image_bytes)
                            processed_element["description"] = image_result["description"]
                            processed_element["embedding"] = image_result["embedding"]
                            processed_element["original_content_summary"] = f"Image (.{image_ext})"
                            # Save image to file
                            image_filename = os.path.join(pdf_output_path, f"page_{element['page_num']}_image_{element_idx + 1}.{image_ext}")
                            with open(image_filename, "wb") as img_file:
                                img_file.write(image_bytes)

                        extracted_elements_with_embeddings.append(processed_element)
                except Exception as page_e:
                    print(f"  - Error processing Page {page_num + 1} of {pdf_name}: {page_e}")
                    # Continue to the next page if an error occurs on the current one
                    continue

        print(f"--- Finished processing {pdf_name} --- Total elements extracted across all pages: {len(extracted_elements_with_embeddings)}")
        return extracted_elements_with_embeddings

    except Exception as e:
        print(f"Error processing {pdf_name}: {e}")
        return []

def process_multiple_pdfs_with_embeddings(input_dir: str, output_dir: str):
    """
    Processes all PDF files in a given input directory, extracts content,
    generates embeddings, and saves structured results.
    Also stores the embeddings in a persistent ChromaDB for each PDF.

    Args:
        input_dir (str): The directory containing the PDF files.
        output_dir (str): The directory where all extracted content and
                          embedding data will be saved.
    """
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting PDF extraction and embedding from '{input_dir}' to '{output_dir}'")

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    all_pdfs_processed_data = {}

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        extracted_data = extract_and_embed_pdf(pdf_path, output_dir)
        if extracted_data:
            all_pdfs_processed_data[pdf_file] = extracted_data
            # Save the structured data for this PDF
            output_json_path = os.path.join(output_dir, os.path.splitext(pdf_file)[0], "extracted_embeddings.json")
            with open(output_json_path, "w", encoding="utf-8") as f:
                # Convert embeddings (lists of floats) to JSON-serializable format
                serializable_data = []
                for item in extracted_data:
                    serializable_item = item.copy()
                    # Ensure content (DataFrame/bytes) is not directly in JSON, only description/embedding
                    if "content" in serializable_item:
                        del serializable_item["content"]
                    serializable_data.append(serializable_item)
                json.dump(serializable_data, f, indent=2)
            print(f"  - Saved structured data and embeddings to: {output_json_path}")

            # --- Store embeddings in ChromaDB ---
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

            # Add documents to ChromaDB. We pass the pre-computed embeddings.
            # If you don't pass embeddings, Chroma will compute them using the provided embeddings_model.
            # We explicitly pass them here to ensure consistency and control.
            vectorstore = Chroma.from_documents(
                documents=lc_documents_for_chroma,
                embedding=embeddings_model, # Pass the embedding function
                persist_directory=chroma_db_path
            )
            # Persist the database to disk
            vectorstore.persist()
            print(f"  - Stored {len(lc_documents_for_chroma)} elements in ChromaDB for {pdf_file}.")
            # You can optionally load it back later:
            # loaded_vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings_model)
            # results = loaded_vectorstore.similarity_search("your query")

        print("=" * 70) # Major separator

    print("\nAll PDF files processed, embeddings generated, and stored in ChromaDB.")
    print(f"Check the '{output_dir}' directory for results.")
    print("Each PDF will have its own subfolder containing extracted content (text, tables, images),")
    print("a JSON file with structured data and embeddings, and a 'chroma_db' folder.")


if __name__ == "__main__":
    # --- Configuration ---
    # Create a 'pdfs_to_embed' folder in the same directory as this script
    # and place your PDF files inside it.
    # Create an 'embedded_content' folder for the output.
    current_script_dir = os.getcwd()
    input_pdf_directory = os.path.join(current_script_dir, "pdfs_to_embed")
    output_embedding_directory = os.path.join(current_script_dir, "embedded_content")

    # --- Run the extraction and embedding process ---
    process_multiple_pdfs_with_embeddings(input_pdf_directory, output_embedding_directory)

    print(f"\nExtraction and Embedding complete! Check the '{output_embedding_directory}' directory for results.")
    print("Each PDF will have its own subfolder containing extracted content (text, tables, images) and a JSON file with structured data and embeddings.")







# Sample documents for our knowledge base
# In a real application, these would come from files, databases, etc.
raw_documents = [
    "The capital of France is Paris. Paris is known for the Eiffel Tower.",
    "The Amazon rainforest is the largest rainforest in the world.",
    "Python is a popular programming language for AI and web development.",
    "The highest mountain in the world is Mount Everest, located in the Himalayas.",
    "Artificial intelligence (AI) is a rapidly developing field.",
    "The primary colors are red, yellow, and blue.",
    "Water's chemical formula is H2O.",
    "The Earth revolves around the Sun."
]