"""Extract structured data from PDFs using PyMuPDF (fitz)."""

import fitz  # PyMuPDF
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class PDFExtractor:
    """Extracts text, tables, and images from PDFs."""

    def __init__(self, output_dir: str = "./data/staged"):
        """Initialize extractor.
        
        Args:
            output_dir: Directory to save extracted JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured data from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted data structure
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        
        extracted_data = {
            "metadata": {
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "total_pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
            },
            "pages": []
        }
        
        print(f"Extracting structure from {pdf_path.name}...")
        
        for page_num, page in enumerate(doc, 1):
            page_data = {
                "page_number": page_num,
                "text": "",
                "tables": [],
                "images": [],
                "sections": [] # Placeholder for now, can implement heuristic detection later
            }
            
            # 1. Extract Text
            text = page.get_text()
            page_data["text"] = text
            
            # 2. Extract Tables (using fitz's built-in table finder)
            # Find tables with default settings
            tabs = page.find_tables()
            if tabs.tables:
                for i, tab in enumerate(tabs):
                    # Extract content as list of lists
                    content = tab.extract()
                    # Convert to markdown-like string for better RAG ingestion
                    md_table = self._table_to_markdown(content)
                    
                    page_data["tables"].append({
                        "id": f"table_{page_num}_{i+1}",
                        "content": md_table,
                        "raw_data": content,
                        "bbox": list(tab.bbox),
                        "caption": f"Table on page {page_num}" # Placeholder, could use text analysis to find caption
                    })
            
            # 3. Extract Images
            # Logic: Iterate over image blocks
            image_list = page.get_images()
            if image_list:
                for i, img in enumerate(image_list):
                    xref = img[0]
                    # We could save images to disk here if needed for multimodal RAG
                    # For now, we just note their existence and location
                    # Optionally extract text from image using OCR (tesseract) if needed
                    # For MVP, we skip OCR to avoid heavy dependencies unless requested
                    page_data["images"].append({
                        "id": f"image_{page_num}_{i+1}",
                        "xref": xref,
                        "description": f"Image on page {page_num} (xref {xref})",
                        "type": "image"
                    })

            extracted_data["pages"].append(page_data)
            
        doc.close()
        
        # Save to JSON
        output_file = self.output_dir / f"{pdf_path.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            
        print(f"✓ Saved extraction to {output_file}")
        return extracted_data

    def _table_to_markdown(self, content: List[List[str]]) -> str:
        """Convert list of lists to markdown table."""
        if not content:
            return ""
            
        # Identify headers (heuristically first row)
        headers = content[0]
        rows = content[1:]
        
        # Clean data (remove None, replace newlines)
        headers = [str(h).replace("\n", " ") if h is not None else "" for h in headers]
        
        # Markdown table construction
        # | Header 1 | Header 2 |
        # | --- | --- |
        # | Row 1 | Row 1 |
        
        md_lines = []
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in rows:
            clean_row = [str(c).replace("\n", " ") if c is not None else "" for c in row]
            md_lines.append("| " + " | ".join(clean_row) + " |")
            
        return "\n".join(md_lines)
