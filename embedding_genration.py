from model import llm, embeddings
from langchain_core.messages import HumanMessage
from typing import List, Dict, Any
import base64
import pandas as pd

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
        response = llm.invoke([
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
        embedding = embeddings.embed_query(description)
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
        embedding = embeddings.embed_query(text_content)
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

        embedding = embeddings.embed_query(description)
        print("    - Table description and embedding generated.")
        return {"description": description, "embedding": embedding}
    except Exception as e:
        print(f"    - Error generating table description/embedding: {e}")
        return {"description": "Error generating table description.", "embedding": []}
