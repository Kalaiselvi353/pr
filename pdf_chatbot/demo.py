#!/usr/bin/env python3
"""
Demo script for PDF Chat Bot

This script demonstrates how to use the PDF Chat Bot system programmatically.
It shows the complete workflow from PDF upload to querying the knowledge base.
"""

import os
import sys
import json
import time
from pathlib import Path
import requests
import base64
from PIL import Image
import io

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_processor import PDFProcessor
from knowledge_base import MultimodalKnowledgeBase

class PDFChatBotDemo:
    def __init__(self, pdf_dir="pdfs", output_dir="processed_data"):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.pdf_processor = None
        self.knowledge_base = None
        
    def initialize(self):
        """Initialize the PDF processor and knowledge base"""
        print("🔧 Initializing PDF Chat Bot...")
        
        # Create directories
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.pdf_processor = PDFProcessor(self.pdf_dir, self.output_dir)
        self.knowledge_base = MultimodalKnowledgeBase(self.output_dir)
        
        print("✅ Initialization complete!")
        
    def add_sample_pdfs(self):
        """Add sample PDFs to demonstrate the system"""
        print("\n📄 Sample PDF Management")
        print("=" * 40)
        
        # Check if PDFs already exist
        existing_pdfs = list(Path(self.pdf_dir).glob("*.pdf"))
        
        if existing_pdfs:
            print(f"📁 Found {len(existing_pdfs)} existing PDF files:")
            for pdf in existing_pdfs:
                file_size = pdf.stat().st_size / (1024 * 1024)  # MB
                print(f"   • {pdf.name} ({file_size:.1f} MB)")
            return existing_pdfs
        else:
            print(f"📂 No PDFs found in {self.pdf_dir}")
            print(f"💡 Please add your PDF files to: {os.path.abspath(self.pdf_dir)}")
            print("   You can:")
            print("   1. Copy PDF files manually to the pdfs/ folder")
            print("   2. Use the web interface to upload files")
            print("   3. Continue with this demo if you have PDFs ready")
            return []
    
    def process_pdfs(self, max_files=None):
        """Process PDFs and build knowledge base"""
        print("\n🔄 Processing PDFs...")
        print("=" * 40)
        
        # Get PDF files
        pdf_files = list(Path(self.pdf_dir).glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found to process")
            return False
        
        if max_files:
            pdf_files = pdf_files[:max_files]
            print(f"🔢 Processing first {max_files} files only")
        
        processed_data = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n📄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            start_time = time.time()
            
            try:
                # Process single PDF
                pdf_data = self.pdf_processor.process_single_pdf(str(pdf_file))
                processed_data.append(pdf_data)
                
                # Show statistics
                processing_time = time.time() - start_time
                total_text = sum(len(page['text_content']) for page in pdf_data['pages'])
                total_images = sum(len(page['images']) for page in pdf_data['pages'])
                total_tables = sum(len(page['tables']) for page in pdf_data['pages'])
                
                print(f"✅ Completed in {processing_time:.1f}s")
                print(f"   • Pages: {pdf_data['total_pages']}")
                print(f"   • Text segments: {total_text}")
                print(f"   • Images: {total_images}")
                print(f"   • Tables: {total_tables}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")
                continue
        
        if not processed_data:
            print("❌ No PDFs processed successfully")
            return False
        
        # Build knowledge base
        print(f"\n🧠 Building knowledge base from {len(processed_data)} PDFs...")
        try:
            self.knowledge_base.build_knowledge_base()
            print("✅ Knowledge base built successfully!")
            return True
        except Exception as e:
            print(f"❌ Error building knowledge base: {e}")
            return False
    
    def demonstrate_queries(self):
        """Demonstrate various types of queries"""
        print("\n🔍 Query Demonstrations")
        print("=" * 40)
        
        # Sample queries based on the jewelry PDF content you showed
        sample_queries = [
            {
                "query": "ring size chart",
                "description": "Search for ring size information and charts"
            },
            {
                "query": "gold purity marking",
                "description": "Find information about gold purity and marking standards"
            },
            {
                "query": "jewelry specifications table",
                "description": "Look for tabular data with specifications"
            },
            {
                "query": "metal types and standards",
                "description": "Search for different metals and their standards"
            },
            {
                "query": "size measurements",
                "description": "Find measurement data and sizing information"
            }
        ]
        
        for i, query_info in enumerate(sample_queries, 1):
            query = query_info["query"]
            description = query_info["description"]
            
            print(f"\n🔍 Query {i}: '{query}'")
            print(f"📝 Description: {description}")
            
            try:
                results = self.knowledge_base.search(query, top_k=5)
                
                if results:
                    print(f"✅ Found {len(results)} results")
                    
                    for j, result in enumerate(results[:3], 1):
                        score = result['score']
                        content_type = result['type']
                        page = result['content'].get('page_number', 0) + 1
                        
                        print(f"   {j}. {content_type.upper()} (Page {page}, Score: {score:.3f})")
                        
                        if content_type == 'text':
                            text = result['content']['text']
                            preview = text[:150] + "..." if len(text) > 150 else text
                            print(f"      Preview: {preview}")
                            
                        elif content_type == 'image':
                            context = result['content']['context_text']
                            ocr_text = result['content']['ocr_text']
                            filename = result['content']['filename']
                            
                            print(f"      Image: {filename}")
                            if context:
                                preview = context[:100] + "..." if len(context) > 100 else context
                                print(f"      Context: {preview}")
                            if ocr_text:
                                preview = ocr_text[:100] + "..." if len(ocr_text) > 100 else ocr_text
                                print(f"      OCR Text: {preview}")
                                
                        elif content_type == 'table':
                            headers = result['content']['headers']
                            print(f"      Table Headers: {headers}")
                            
                else:
                    print("❌ No results found")
                    
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            
            print("-" * 40)
    
    def interactive_mode(self):
        """Interactive query mode"""
        print("\n💬 Interactive Mode")
        print("=" * 40)
        print("Enter your queries (type 'quit' to exit, 'help' for commands)")
        
        while True:
            try:
                query = input("\n🔍 Your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help     - Show this help message")
                    print("  stats    - Show knowledge base statistics")
                    print("  quit/q   - Exit interactive mode")
                    print("\nOr enter any question about your PDFs!")
                    continue
                elif query.lower() == 'stats':
                    self.show_statistics()
                    continue
                elif not query:
                    continue
                
                # Process query
                print(f"🔍 Searching for: '{query}'")
                results = self.knowledge_base.search(query, top_k=8)
                
                if results:
                    print(f"\n📊 Found {len(results)} results:")
                    
                    # Group results by type
                    text_results = [r for r in results if r['type'] == 'text']
                    image_results = [r for r in results if r['type'] == 'image']
                    table_results = [r for r in results if r['type'] == 'table']
                    
                    # Show text results
                    if text_results:
                        print(f"\n📄 Text Results ({len(text_results)}):")
                        for i, result in enumerate(text_results[:3], 1):
                            content = result['content']
                            print(f"  {i}. Page {content['page_number'] + 1} (Score: {result['score']:.3f})")
                            print(f"     {content['text']}")
                            print()
                    
                    # Show image results
                    if image_results:
                        print(f"\n🖼️  Image Results ({len(image_results)}):")
                        for i, result in enumerate(image_results[:2], 1):
                            content = result['content']
                            print(f"  {i}. {content['filename']} - Page {content['page_number'] + 1} (Score: {result['score']:.3f})")
                            if content['context_text']:
                                print(f"     Context: {content['context_text']}")
                            if content['ocr_text']:
                                print(f"     OCR: {content['ocr_text']}")
                            print()
                    
                    # Show table results
                    if table_results:
                        print(f"\n📊 Table Results ({len(table_results)}):")
                        for i, result in enumerate(table_results[:2], 1):
                            content = result['content']
                            print(f"  {i}. Page {content['page_number'] + 1} (Score: {result['score']:.3f})")
                            print(f"     Headers: {content['headers']}")
                            print(f"     Data preview: {str(content['data'][:2])}...")
                            print()
                else:
                    print("❌ No relevant results found. Try rephrasing your query.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_statistics(self):
        """Show knowledge base statistics"""
        print("\n📊 Knowledge Base Statistics")
        print("=" * 40)
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_base.db_path)
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute('SELECT COUNT(*) FROM text_content')
            text_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM image_content')
            image_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM table_content')
            table_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT pdf_name) FROM text_content')
            pdf_count = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"📚 PDFs processed: {pdf_count}")
            print(f"📄 Text segments: {text_count}")
            print(f"🖼️  Images: {image_count}")
            print(f"📊 Tables: {table_count}")
            
            # Index information
            if hasattr(self.knowledge_base, 'combined_index') and self.knowledge_base.combined_index:
                print(f"🔍 Search index size: {self.knowledge_base.combined_index.ntotal}")
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
    
    def run_demo(self):
        """Run the complete demo"""
        print("🎯 PDF Chat Bot - Complete Demo")
        print("=" * 50)
        
        # Initialize
        self.initialize()
        
        # Check for PDFs
        pdf_files = self.add_sample_pdfs()
        
        if not pdf_files:
            print("\n⚠️  Demo cannot continue without PDF files.")
            print("Please add PDF files and run the demo again.")
            return
        
        # Process PDFs
        if not self.process_pdfs():
            print("\n❌ Demo failed - could not process PDFs")
            return
        
        # Show statistics
        self.show_statistics()
        
        # Demonstrate queries
        self.demonstrate_queries()
        
        # Interactive mode
        print("\n🎉 Demo completed successfully!")
        response = input("\nWould you like to try interactive mode? (y/N): ").strip().lower()
        if response == 'y':
            self.interactive_mode()
        
        print("\n👋 Demo finished! You can now:")
        print("1. Run 'python app.py' to start the web interface")
        print("2. Open http://localhost:5000 in your browser")
        print("3. Upload more PDFs and chat with your documents!")

def main():
    """Main function"""
    demo = PDFChatBotDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()