import os
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
from sentence_transformers import SentenceTransformer
import torch
import clip
from typing import List, Dict, Tuple, Any
import pandas as pd
import camelot
import re
from pathlib import Path
import hashlib
import base64
from io import BytesIO

class PDFProcessor:
    def __init__(self, pdf_dir: str = "pdfs", output_dir: str = "processed_data"):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.text_dir = os.path.join(output_dir, "text")
        self.images_dir = os.path.join(output_dir, "images")
        self.tables_dir = os.path.join(output_dir, "tables")
        
        # Create directories
        for dir_path in [self.text_dir, self.images_dir, self.tables_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
    def extract_text_with_positions(self, doc, page_num: int) -> List[Dict]:
        """Extract text with bounding box positions"""
        page = doc[page_num]
        text_instances = []
        
        # Get text blocks with positions
        blocks = page.get_text("dict")
        
        for block in blocks["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text and len(text) > 2:  # Filter out very short text
                            bbox = span["bbox"]
                            text_instances.append({
                                "text": text,
                                "bbox": bbox,
                                "page": page_num,
                                "font_size": span["size"],
                                "font": span["font"]
                            })
        
        return text_instances
    
    def is_header_footer_or_logo(self, text_instance: Dict, page_height: float) -> bool:
        """Identify headers, footers, or logo text to exclude"""
        bbox = text_instance["bbox"]
        text = text_instance["text"].lower()
        
        # Check position (top 10% or bottom 10% of page)
        if bbox[1] < page_height * 0.1 or bbox[3] > page_height * 0.9:
            return True
            
        # Check for common header/footer patterns
        header_footer_patterns = [
            r'page\s+\d+', r'confidential', r'proprietary', 
            r'copyright', r'©', r'all rights reserved',
            r'doc\.\s*ref', r'rev\.\s*no', r'date\s*:', r'page\s*\d+\s*of\s*\d+'
        ]
        
        for pattern in header_footer_patterns:
            if re.search(pattern, text):
                return True
                
        return False
    
    def extract_images_with_context(self, doc, page_num: int) -> List[Dict]:
        """Extract images with surrounding text context"""
        page = doc[page_num]
        page_rect = page.rect
        
        # Get all images on the page
        image_list = page.get_images()
        extracted_images = []
        
        # Get text instances for context
        text_instances = self.extract_text_with_positions(doc, page_num)
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to PIL Image
                pil_image = Image.open(BytesIO(image_bytes))
                
                # Get image position on page
                img_rects = page.get_image_rects(xref)
                
                for rect in img_rects:
                    # Filter out very small images (likely logos or decorative elements)
                    if rect.width < 50 or rect.height < 50:
                        continue
                    
                    # Find surrounding text context
                    context_text = self.find_text_context(text_instances, rect, page_rect)
                    
                    # Generate unique filename
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    filename = f"page_{page_num}_{img_index}_{img_hash}.{image_ext}"
                    
                    # Crop image intelligently if needed
                    cropped_image = self.intelligent_crop(pil_image, context_text)
                    
                    # Save image
                    image_path = os.path.join(self.images_dir, filename)
                    cropped_image.save(image_path)
                    
                    # Extract text from image using OCR
                    ocr_text = pytesseract.image_to_string(cropped_image)
                    
                    extracted_images.append({
                        "filename": filename,
                        "path": image_path,
                        "page": page_num,
                        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "context_text": context_text,
                        "ocr_text": ocr_text.strip(),
                        "width": rect.width,
                        "height": rect.height
                    })
                    
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num}: {e}")
                continue
        
        return extracted_images
    
    def find_text_context(self, text_instances: List[Dict], img_rect, page_rect, 
                         context_radius: float = 100) -> str:
        """Find text near an image for context"""
        context_texts = []
        
        for text_instance in text_instances:
            if self.is_header_footer_or_logo(text_instance, page_rect.height):
                continue
                
            text_bbox = text_instance["bbox"]
            
            # Check if text is near the image
            if self.is_near(text_bbox, [img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1], 
                           context_radius):
                context_texts.append(text_instance["text"])
        
        return " ".join(context_texts)
    
    def is_near(self, bbox1: List[float], bbox2: List[float], radius: float) -> bool:
        """Check if two bounding boxes are within a certain distance"""
        # Calculate center points
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        # Calculate distance
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        
        return distance <= radius
    
    def intelligent_crop(self, image: Image.Image, context_text: str) -> Image.Image:
        """Intelligently crop image to remove unnecessary parts"""
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Simple edge detection to find content boundaries
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main content)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add some padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.width - x, w + 2 * padding)
            h = min(image.height - y, h + 2 * padding)
            
            # Crop the image
            cropped = image.crop((x, y, x + w, y + h))
            return cropped
        
        return image
    
    def extract_tables(self, pdf_path: str, page_num: int) -> List[Dict]:
        """Extract tables from PDF page"""
        tables = []
        
        try:
            # Use camelot to extract tables
            camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1))
            
            for i, table in enumerate(camelot_tables):
                if table.df.empty:
                    continue
                    
                # Convert table to dict format
                table_dict = {
                    "page": page_num,
                    "table_index": i,
                    "data": table.df.to_dict('records'),
                    "headers": table.df.columns.tolist(),
                    "bbox": table._bbox if hasattr(table, '_bbox') else None
                }
                
                # Generate table description
                table_text = self.table_to_text(table.df)
                table_dict["text_representation"] = table_text
                
                tables.append(table_dict)
                
        except Exception as e:
            print(f"Error extracting tables from page {page_num}: {e}")
        
        return tables
    
    def table_to_text(self, df: pd.DataFrame) -> str:
        """Convert table to readable text format"""
        text_parts = []
        
        # Add headers
        headers = " | ".join(df.columns)
        text_parts.append(f"Table Headers: {headers}")
        
        # Add rows
        for index, row in df.iterrows():
            row_text = " | ".join([str(val) for val in row.values])
            text_parts.append(row_text)
        
        return "\n".join(text_parts)
    
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """Process a single PDF file"""
        print(f"Processing PDF: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        pdf_data = {
            "pdf_name": pdf_name,
            "pdf_path": pdf_path,
            "pages": [],
            "total_pages": len(doc)
        }
        
        for page_num in range(len(doc)):
            print(f"Processing page {page_num + 1}/{len(doc)}")
            
            page_data = {
                "page_number": page_num,
                "text_content": [],
                "images": [],
                "tables": []
            }
            
            # Extract text with positions
            text_instances = self.extract_text_with_positions(doc, page_num)
            
            # Filter out headers, footers, and logos
            filtered_text = []
            page_rect = doc[page_num].rect
            
            for text_instance in text_instances:
                if not self.is_header_footer_or_logo(text_instance, page_rect.height):
                    filtered_text.append(text_instance)
            
            page_data["text_content"] = filtered_text
            
            # Extract images with context
            images = self.extract_images_with_context(doc, page_num)
            page_data["images"] = images
            
            # Extract tables
            tables = self.extract_tables(pdf_path, page_num)
            page_data["tables"] = tables
            
            pdf_data["pages"].append(page_data)
        
        doc.close()
        
        # Save processed data
        output_file = os.path.join(self.text_dir, f"{pdf_name}_processed.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pdf_data, f, indent=2, ensure_ascii=False)
        
        return pdf_data
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDFs in the PDF directory"""
        all_pdf_data = []
        
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            try:
                pdf_data = self.process_single_pdf(pdf_path)
                all_pdf_data.append(pdf_data)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue
        
        return all_pdf_data

if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor()
    
    # Process all PDFs
    print("Starting PDF processing...")
    processed_data = processor.process_all_pdfs()
    print(f"Processed {len(processed_data)} PDFs successfully!")