import os
import json
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import torch
import clip
from PIL import Image
import faiss
import sqlite3
from datetime import datetime
import base64
from io import BytesIO

class MultimodalKnowledgeBase:
    def __init__(self, data_dir: str = "processed_data", db_path: str = "knowledge_base.db"):
        self.data_dir = data_dir
        self.db_path = db_path
        self.text_dir = os.path.join(data_dir, "text")
        self.images_dir = os.path.join(data_dir, "images")
        
        # Initialize models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Initialize database
        self.init_database()
        
        # Initialize FAISS indices
        self.text_index = None
        self.image_index = None
        self.combined_index = None
        
    def init_database(self):
        """Initialize SQLite database for storing metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Text content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_name TEXT,
                page_number INTEGER,
                text TEXT,
                bbox TEXT,
                font_size REAL,
                font TEXT,
                embedding_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Image content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_name TEXT,
                page_number INTEGER,
                filename TEXT,
                path TEXT,
                bbox TEXT,
                context_text TEXT,
                ocr_text TEXT,
                width INTEGER,
                height INTEGER,
                embedding_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS table_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_name TEXT,
                page_number INTEGER,
                table_index INTEGER,
                headers TEXT,
                data TEXT,
                text_representation TEXT,
                bbox TEXT,
                embedding_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Combined embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_type TEXT,  -- 'text', 'image', 'table', 'combined'
                content_id INTEGER,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_text_embeddings(self, text_chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        embeddings = self.sentence_model.encode(text_chunks, convert_to_tensor=False)
        return np.array(embeddings)
    
    def create_image_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Create embeddings for images using CLIP"""
        embeddings = []
        
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    embeddings.append(image_features.cpu().numpy().flatten())
                    
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                # Add zero embedding as placeholder
                embeddings.append(np.zeros(512))
        
        return np.array(embeddings)
    
    def create_combined_embeddings(self, text: str, image_path: str = None) -> np.ndarray:
        """Create combined text-image embeddings"""
        # Text embedding
        text_embedding = self.sentence_model.encode([text])[0]
        
        if image_path and os.path.exists(image_path):
            # Image embedding
            try:
                image = Image.open(image_path).convert('RGB')
                image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    image_embedding = image_features.cpu().numpy().flatten()
                
                # Combine embeddings (concatenate and normalize)
                combined = np.concatenate([text_embedding, image_embedding])
                combined = combined / np.linalg.norm(combined)
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                # Use only text embedding, padded to match combined size
                combined = np.concatenate([text_embedding, np.zeros(512)])
                combined = combined / np.linalg.norm(combined)
        else:
            # Use only text embedding, padded to match combined size
            combined = np.concatenate([text_embedding, np.zeros(512)])
            combined = combined / np.linalg.norm(combined)
        
        return combined
    
    def store_text_content(self, pdf_data: Dict):
        """Store text content in database with embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        text_embeddings = []
        text_data = []
        
        for page in pdf_data["pages"]:
            for text_instance in page["text_content"]:
                # Prepare text data
                text_data.append({
                    "pdf_name": pdf_data["pdf_name"],
                    "page_number": page["page_number"],
                    "text": text_instance["text"],
                    "bbox": json.dumps(text_instance["bbox"]),
                    "font_size": text_instance["font_size"],
                    "font": text_instance["font"]
                })
                
                # Create embedding
                text_embeddings.append(text_instance["text"])
        
        if text_embeddings:
            # Create embeddings
            embeddings = self.create_text_embeddings(text_embeddings)
            
            # Store in database
            for i, (data, embedding) in enumerate(zip(text_data, embeddings)):
                # Store text content
                cursor.execute('''
                    INSERT INTO text_content 
                    (pdf_name, page_number, text, bbox, font_size, font, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data["pdf_name"], data["page_number"], data["text"],
                    data["bbox"], data["font_size"], data["font"], i
                ))
                
                text_id = cursor.lastrowid
                
                # Store embedding
                embedding_blob = pickle.dumps(embedding)
                cursor.execute('''
                    INSERT INTO embeddings (content_type, content_id, embedding)
                    VALUES (?, ?, ?)
                ''', ("text", text_id, embedding_blob))
        
        conn.commit()
        conn.close()
    
    def store_image_content(self, pdf_data: Dict):
        """Store image content in database with embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for page in pdf_data["pages"]:
            for image_data in page["images"]:
                # Store image content
                cursor.execute('''
                    INSERT INTO image_content 
                    (pdf_name, page_number, filename, path, bbox, context_text, 
                     ocr_text, width, height, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pdf_data["pdf_name"], page["page_number"], image_data["filename"],
                    image_data["path"], json.dumps(image_data["bbox"]),
                    image_data["context_text"], image_data["ocr_text"],
                    image_data["width"], image_data["height"], 0  # Will update with actual ID
                ))
                
                image_id = cursor.lastrowid
                
                # Create combined embedding (text context + image)
                combined_text = f"{image_data['context_text']} {image_data['ocr_text']}".strip()
                combined_embedding = self.create_combined_embeddings(
                    combined_text, image_data["path"]
                )
                
                # Store embedding
                embedding_blob = pickle.dumps(combined_embedding)
                cursor.execute('''
                    INSERT INTO embeddings (content_type, content_id, embedding)
                    VALUES (?, ?, ?)
                ''', ("image", image_id, embedding_blob))
                
                embedding_id = cursor.lastrowid
                
                # Update image record with embedding ID
                cursor.execute('''
                    UPDATE image_content SET embedding_id = ? WHERE id = ?
                ''', (embedding_id, image_id))
        
        conn.commit()
        conn.close()
    
    def store_table_content(self, pdf_data: Dict):
        """Store table content in database with embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for page in pdf_data["pages"]:
            for table_data in page["tables"]:
                # Store table content
                cursor.execute('''
                    INSERT INTO table_content 
                    (pdf_name, page_number, table_index, headers, data, 
                     text_representation, bbox, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pdf_data["pdf_name"], page["page_number"], table_data["table_index"],
                    json.dumps(table_data["headers"]), json.dumps(table_data["data"]),
                    table_data["text_representation"], 
                    json.dumps(table_data["bbox"]) if table_data["bbox"] else None, 0
                ))
                
                table_id = cursor.lastrowid
                
                # Create embedding for table text representation
                table_embedding = self.create_text_embeddings([table_data["text_representation"]])[0]
                
                # Store embedding
                embedding_blob = pickle.dumps(table_embedding)
                cursor.execute('''
                    INSERT INTO embeddings (content_type, content_id, embedding)
                    VALUES (?, ?, ?)
                ''', ("table", table_id, embedding_blob))
                
                embedding_id = cursor.lastrowid
                
                # Update table record with embedding ID
                cursor.execute('''
                    UPDATE table_content SET embedding_id = ? WHERE id = ?
                ''', (embedding_id, table_id))
        
        conn.commit()
        conn.close()
    
    def build_faiss_indices(self):
        """Build FAISS indices for fast similarity search"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all embeddings
        cursor.execute('''
            SELECT id, content_type, embedding FROM embeddings
        ''')
        
        embeddings_data = cursor.fetchall()
        
        if not embeddings_data:
            print("No embeddings found in database")
            return
        
        # Separate embeddings by type
        text_embeddings = []
        image_embeddings = []
        table_embeddings = []
        combined_embeddings = []
        
        text_ids = []
        image_ids = []
        table_ids = []
        combined_ids = []
        
        for embedding_id, content_type, embedding_blob in embeddings_data:
            embedding = pickle.loads(embedding_blob)
            
            if content_type == "text":
                text_embeddings.append(embedding)
                text_ids.append(embedding_id)
            elif content_type == "image":
                image_embeddings.append(embedding)
                image_ids.append(embedding_id)
            elif content_type == "table":
                table_embeddings.append(embedding)
                table_ids.append(embedding_id)
        
        # Build FAISS indices
        if text_embeddings:
            text_embeddings = np.array(text_embeddings).astype('float32')
            self.text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
            self.text_index.add(text_embeddings)
            self.text_ids = text_ids
        
        if image_embeddings:
            image_embeddings = np.array(image_embeddings).astype('float32')
            self.image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
            self.image_index.add(image_embeddings)
            self.image_ids = image_ids
        
        if table_embeddings:
            table_embeddings = np.array(table_embeddings).astype('float32')
            self.table_index = faiss.IndexFlatIP(table_embeddings.shape[1])
            self.table_index.add(table_embeddings)
            self.table_ids = table_ids
        
        # Create combined index with all embeddings
        all_embeddings = []
        all_ids = []
        all_types = []
        
        for emb, ids, content_type in [
            (text_embeddings if text_embeddings != [] else None, text_ids, "text"),
            (image_embeddings if image_embeddings != [] else None, image_ids, "image"),
            (table_embeddings if table_embeddings != [] else None, table_ids, "table")
        ]:
            if emb is not None:
                all_embeddings.extend(emb)
                all_ids.extend(ids)
                all_types.extend([content_type] * len(ids))
        
        if all_embeddings:
            # Normalize dimensions for combined index
            max_dim = max(emb.shape[0] for emb in all_embeddings)
            normalized_embeddings = []
            
            for emb in all_embeddings:
                if len(emb) < max_dim:
                    # Pad with zeros
                    padded = np.zeros(max_dim)
                    padded[:len(emb)] = emb
                    normalized_embeddings.append(padded)
                else:
                    normalized_embeddings.append(emb[:max_dim])
            
            combined_embeddings = np.array(normalized_embeddings).astype('float32')
            self.combined_index = faiss.IndexFlatIP(combined_embeddings.shape[1])
            self.combined_index.add(combined_embeddings)
            self.combined_ids = all_ids
            self.combined_types = all_types
        
        conn.close()
        print("FAISS indices built successfully!")
    
    def build_knowledge_base(self, processed_data_files: List[str] = None):
        """Build complete knowledge base from processed PDF data"""
        if processed_data_files is None:
            # Get all processed JSON files
            processed_data_files = [
                os.path.join(self.text_dir, f) 
                for f in os.listdir(self.text_dir) 
                if f.endswith('_processed.json')
            ]
        
        print(f"Building knowledge base from {len(processed_data_files)} files...")
        
        for json_file in processed_data_files:
            print(f"Processing {json_file}...")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                pdf_data = json.load(f)
            
            # Store different types of content
            self.store_text_content(pdf_data)
            self.store_image_content(pdf_data)
            self.store_table_content(pdf_data)
        
        # Build FAISS indices
        self.build_faiss_indices()
        
        print("Knowledge base built successfully!")
    
    def search(self, query: str, top_k: int = 10, content_types: List[str] = None) -> List[Dict]:
        """Search the knowledge base"""
        if content_types is None:
            content_types = ["text", "image", "table"]
        
        # Create query embedding
        query_embedding = self.sentence_model.encode([query])[0]
        
        results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Search in each content type
        for content_type in content_types:
            if content_type == "text" and self.text_index:
                # Search text index
                query_emb = query_embedding.reshape(1, -1).astype('float32')
                scores, indices = self.text_index.search(query_emb, min(top_k, self.text_index.ntotal))
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:
                        embedding_id = self.text_ids[idx]
                        
                        # Get text content
                        cursor.execute('''
                            SELECT * FROM text_content 
                            WHERE embedding_id = (
                                SELECT content_id FROM embeddings WHERE id = ?
                            )
                        ''', (embedding_id,))
                        
                        content = cursor.fetchone()
                        if content:
                            results.append({
                                "type": "text",
                                "score": float(score),
                                "content": {
                                    "id": content[0],
                                    "pdf_name": content[1],
                                    "page_number": content[2],
                                    "text": content[3],
                                    "bbox": json.loads(content[4]) if content[4] else None,
                                    "font_size": content[5],
                                    "font": content[6]
                                }
                            })
            
            elif content_type == "image" and self.image_index:
                # Create combined query embedding for images
                combined_query = self.create_combined_embeddings(query)
                query_emb = combined_query.reshape(1, -1).astype('float32')
                scores, indices = self.image_index.search(query_emb, min(top_k, self.image_index.ntotal))
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:
                        embedding_id = self.image_ids[idx]
                        
                        # Get image content
                        cursor.execute('''
                            SELECT * FROM image_content 
                            WHERE embedding_id = ?
                        ''', (embedding_id,))
                        
                        content = cursor.fetchone()
                        if content:
                            results.append({
                                "type": "image",
                                "score": float(score),
                                "content": {
                                    "id": content[0],
                                    "pdf_name": content[1],
                                    "page_number": content[2],
                                    "filename": content[3],
                                    "path": content[4],
                                    "bbox": json.loads(content[5]) if content[5] else None,
                                    "context_text": content[6],
                                    "ocr_text": content[7],
                                    "width": content[8],
                                    "height": content[9]
                                }
                            })
            
            elif content_type == "table" and hasattr(self, 'table_index') and self.table_index:
                # Search table index
                query_emb = query_embedding.reshape(1, -1).astype('float32')
                scores, indices = self.table_index.search(query_emb, min(top_k, self.table_index.ntotal))
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:
                        embedding_id = self.table_ids[idx]
                        
                        # Get table content
                        cursor.execute('''
                            SELECT * FROM table_content 
                            WHERE embedding_id = ?
                        ''', (embedding_id,))
                        
                        content = cursor.fetchone()
                        if content:
                            results.append({
                                "type": "table",
                                "score": float(score),
                                "content": {
                                    "id": content[0],
                                    "pdf_name": content[1],
                                    "page_number": content[2],
                                    "table_index": content[3],
                                    "headers": json.loads(content[4]) if content[4] else None,
                                    "data": json.loads(content[5]) if content[5] else None,
                                    "text_representation": content[6],
                                    "bbox": json.loads(content[7]) if content[7] else None
                                }
                            })
        
        conn.close()
        
        # Sort results by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

if __name__ == "__main__":
    # Example usage
    kb = MultimodalKnowledgeBase()
    kb.build_knowledge_base()
    
    # Test search
    results = kb.search("ring size chart", top_k=5)
    for result in results:
        print(f"Type: {result['type']}, Score: {result['score']:.4f}")
        print(f"Content: {result['content']}")
        print("-" * 50)