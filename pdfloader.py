import io
import os
import pdfplumber
import pandas as pd
import fitz # PyMuPDF for image extraction
from PIL import Image # Pillow for image manipulation
import json
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document as LangchainDocument # Alias to avoid conflict if needed
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma # Import ChromaDB
from model import text_splitter # Import LLM and embeddings from model.py
from embedding_genration import get_text_embedding, get_table_embedding, get_image_description_embedding
from storing_embeddings import load_chroma_db_and_retriever, store_embeddings_in_chromadb

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

def read_and_embed_multiple_pdfs(input_dir: str, output_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads multiple PDF files, extracts their content (text, tables, images),
    generates embeddings, and saves structured results to JSON files.

    Args:
        input_dir (str): The directory containing the PDF files.
        output_dir (str): The base directory where all extracted content and
                          embedding data will be saved.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary where keys are PDF filenames
                                         and values are lists of extracted elements
                                         with their embeddings.
    """
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting PDF extraction and embedding from '{input_dir}' to '{output_dir}'")

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return {}

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
        print("=" * 70) # Major separator
    print("\nAll PDF files processed and embeddings generated.")
    return all_pdfs_processed_data


#if __name__ == "__main__":
    


def vector_db():
    # --- Configuration ---
    # Create a 'pdfs_to_embed' folder in the same directory as this script
    # and place your PDF files inside it.
    # Create an 'embedded_content' folder for the output.
    current_script_dir = os.getcwd()
    input_pdf_directory = os.path.join(current_script_dir, "pdfs_to_embed")
    output_embedding_directory = os.path.join(current_script_dir, "embedded_content")

    # --- Create dummy PDF files for testing (optional) ---
    # This part helps you test the script without manually creating PDFs.
    # You would typically comment this out or remove it in a real scenario.
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image as ReportLabImage
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        print("Creating dummy PDF files for testing...")
        os.makedirs(input_pdf_directory, exist_ok=True)

        # PDF 1: Text, Table, Text
        doc1 = SimpleDocTemplate(os.path.join(input_pdf_directory, "doc_with_text_table.pdf"), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("Introduction text for the document.", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Below is a key performance indicator table:", styles['Normal']))
        data = [['Metric', 'Q1', 'Q2', 'Q3'],
                ['Revenue', '$100K', '$120K', '$130K'],
                ['Profit', '$20K', '$25K', '$28K']]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#E8F5E9')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#81C784'))
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("This concludes the first section. More details follow.", styles['Normal']))
        doc1.build(story)
        print("  - Created doc_with_text_table.pdf")

        # PDF 2: Text, Image, Text
        # For a real image, you'd need a small image file (e.g., 'sample_image.png')
        # Here, we'll create a simple placeholder image in memory
        img_width, img_height = 200, 150
        img_buffer = io.BytesIO()
        img = Image.new('RGB', (img_width, img_height), color = 'red')
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0) # Rewind buffer for reading

        doc2 = SimpleDocTemplate(os.path.join(input_pdf_directory, "doc_with_image.pdf"), pagesize=letter)
        story2 = []
        story2.append(Paragraph("Here's some information about our product.", styles['Normal']))
        story2.append(Spacer(1, 0.2 * inch))
        # Embed the image from buffer
        reportlab_image = ReportLabImage(img_buffer, img_width, img_height)
        story2.append(reportlab_image)
        story2.append(Spacer(1, 0.2 * inch))
        story2.append(Paragraph("The image above illustrates the product's design.", styles['Normal']))
        doc2.build(story2)
        print("  - Created doc_with_image.pdf (with embedded PNG)")

        # PDF 3: Just Text (multi-page)
        doc3 = SimpleDocTemplate(os.path.join(input_pdf_directory, "long_text_doc.pdf"), pagesize=letter)
        story3 = []
        for i in range(5):
            story3.append(Paragraph(f"This is paragraph {i+1} of a longer text document. It aims to demonstrate how text across multiple pages is handled and chunked. LangChain's text splitters are essential for breaking down large documents into manageable pieces for LLMs. This helps in overcoming token limits and improving the relevance of retrieved information in RAG applications.", styles['Normal']))
            story3.append(Spacer(1, 0.1 * inch))
            if i == 2: # Add a page break after 3 paragraphs
                story3.append(Paragraph("--- Page Break ---", styles['h2']))
                story3.append(Spacer(1, 0.2 * inch))
                story3.append(Paragraph("This is the start of a new page.", styles['Normal']))
                story3.append(Spacer(1, 0.1 * inch))
        doc3.build(story3)
        print("  - Created long_text_doc.pdf")

    except ImportError:
        print("\nNote: reportlab or Pillow library not found. Dummy PDF files will not be created.")
        print("Please manually place your PDF files in the 'pdfs_to_embed' folder.")
        print("Install with: pip install reportlab Pillow\n")
    # --- End of dummy PDF creation ---

    # --- Run the extraction and embedding process ---
    # Step 1: Read PDFs, extract content, generate embeddings, and save JSON
    processed_data = read_and_embed_multiple_pdfs(input_pdf_directory, output_embedding_directory)

    # Step 2: Store the generated embeddings into ChromaDB
    store_embeddings_in_chromadb(processed_data, output_embedding_directory)

    # --- Demonstrate loading ChromaDB and using retriever ---
    print("\n--- Demonstrating ChromaDB Loading and Retrieval ---")
    # Assuming 'doc_with_text_table.pdf' was processed and its ChromaDB created
    sample_pdf_name = "doc_with_text_table" # Adjust if you want to test another dummy PDF
    sample_chroma_path = os.path.join(output_embedding_directory, sample_pdf_name, "chroma_db")

    retriever = load_chroma_db_and_retriever(sample_chroma_path)

    if retriever:
        query = "What is the revenue for Q2?"
        print(f"\nQuerying ChromaDB: '{query}'")
        retrieved_docs = retriever.invoke(query)

        print("\n--- Retrieved Documents ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"Document {i+1} (Type: {doc.metadata.get('type')}, Page: {doc.metadata.get('page_num')}):")
            print(f"  Content: {doc.page_content[:150]}...") # Print first 150 chars of content
            print(f"  Metadata: {doc.metadata}\n")
    else:
        print("Could not load ChromaDB for demonstration.")

    print(f"\nOverall process complete! Check the '{output_embedding_directory}' directory for results.")
    print("Each PDF will have its own subfolder containing extracted content (text, tables, images),")
    print("a JSON file with structured data and embeddings, and a 'chroma_db' folder.")
    return retriever

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