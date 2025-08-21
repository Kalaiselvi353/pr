#!/usr/bin/env python3
"""
Build Corpus Script for PDF Chat Bot

This script demonstrates the complete workflow:
1. Process PDFs from the pdfs/ directory
2. Extract text, images, and tables
3. Create multimodal embeddings
4. Build the knowledge base
5. Test the system

Usage:
    python build_corpus.py [--pdf-dir PATH] [--output-dir PATH] [--rebuild]
"""

import os
import sys
import argparse
import time
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_processor import PDFProcessor
from knowledge_base import MultimodalKnowledgeBase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_corpus.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Build corpus for PDF Chat Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build_corpus.py                          # Use default directories
    python build_corpus.py --pdf-dir my_pdfs       # Custom PDF directory
    python build_corpus.py --rebuild               # Force rebuild existing corpus
    python build_corpus.py --test-query "ring size"  # Test with specific query
        """
    )
    
    parser.add_argument(
        '--pdf-dir',
        default='pdfs',
        help='Directory containing PDF files (default: pdfs)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='processed_data',
        help='Output directory for processed data (default: processed_data)'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild of existing knowledge base'
    )
    
    parser.add_argument(
        '--test-query',
        help='Test query to run after building corpus'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of PDF files to process (for testing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("🔍 Checking dependencies...")
    
    required_modules = [
        'fitz',
        'cv2', 
        'PIL',
        'pytesseract',
        'sentence_transformers',
        'torch',
        'clip',
        'faiss',
        'camelot',
        'numpy',
        'pandas'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError:
            missing_modules.append(module)
            logger.error(f"❌ {module}")
    
    if missing_modules:
        logger.error(f"Missing modules: {', '.join(missing_modules)}")
        logger.error("Please run: pip install -r requirements.txt")
        return False
    
    # Test Tesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        logger.info(f"✅ Tesseract OCR version: {version}")
    except Exception as e:
        logger.error(f"❌ Tesseract OCR not available: {e}")
        return False
    
    logger.info("✅ All dependencies are available")
    return True

def scan_pdf_directory(pdf_dir):
    """Scan PDF directory and return list of PDF files"""
    pdf_dir = Path(pdf_dir)
    
    if not pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {pdf_dir}")
        return []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return []
    
    logger.info(f"📁 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        file_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"   • {pdf_file.name} ({file_size:.1f} MB)")
    
    return pdf_files

def process_pdfs(pdf_processor, pdf_files, max_files=None):
    """Process PDF files and return processed data"""
    if max_files:
        pdf_files = pdf_files[:max_files]
        logger.info(f"🔢 Processing only first {max_files} files (test mode)")
    
    processed_data = []
    total_files = len(pdf_files)
    
    logger.info(f"🔄 Starting to process {total_files} PDF files...")
    
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"\n📄 Processing file {i}/{total_files}: {pdf_file.name}")
        start_time = time.time()
        
        try:
            pdf_data = pdf_processor.process_single_pdf(str(pdf_file))
            processed_data.append(pdf_data)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Processed {pdf_file.name} in {processing_time:.1f}s")
            logger.info(f"   • Pages: {pdf_data['total_pages']}")
            
            # Count extracted content
            total_text = sum(len(page['text_content']) for page in pdf_data['pages'])
            total_images = sum(len(page['images']) for page in pdf_data['pages'])
            total_tables = sum(len(page['tables']) for page in pdf_data['pages'])
            
            logger.info(f"   • Text segments: {total_text}")
            logger.info(f"   • Images: {total_images}")
            logger.info(f"   • Tables: {total_tables}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file.name}: {e}")
            continue
    
    logger.info(f"\n🎉 Successfully processed {len(processed_data)}/{total_files} PDF files")
    return processed_data

def build_knowledge_base(kb, processed_data):
    """Build the knowledge base from processed data"""
    logger.info("🧠 Building knowledge base...")
    start_time = time.time()
    
    try:
        # Store processed data in knowledge base
        for pdf_data in processed_data:
            logger.info(f"📚 Storing data for: {pdf_data['pdf_name']}")
            kb.store_text_content(pdf_data)
            kb.store_image_content(pdf_data)
            kb.store_table_content(pdf_data)
        
        # Build FAISS indices
        logger.info("🔍 Building search indices...")
        kb.build_faiss_indices()
        
        build_time = time.time() - start_time
        logger.info(f"✅ Knowledge base built successfully in {build_time:.1f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error building knowledge base: {e}")
        return False

def test_system(kb, test_queries=None):
    """Test the system with sample queries"""
    if test_queries is None:
        test_queries = [
            "ring size chart",
            "jewelry specifications",
            "purity marking",
            "table data",
            "gold measurements"
        ]
    
    logger.info("🧪 Testing system with sample queries...")
    
    for query in test_queries:
        logger.info(f"\n🔍 Testing query: '{query}'")
        
        try:
            results = kb.search(query, top_k=5)
            
            if results:
                logger.info(f"✅ Found {len(results)} results")
                for i, result in enumerate(results[:3], 1):
                    score = result['score']
                    content_type = result['type']
                    logger.info(f"   {i}. Type: {content_type}, Score: {score:.3f}")
                    
                    if content_type == 'text':
                        text = result['content']['text'][:100] + "..." if len(result['content']['text']) > 100 else result['content']['text']
                        logger.info(f"      Text: {text}")
                    elif content_type == 'image':
                        context = result['content']['context_text'][:100] + "..." if len(result['content']['context_text']) > 100 else result['content']['context_text']
                        logger.info(f"      Context: {context}")
                    elif content_type == 'table':
                        headers = result['content']['headers']
                        logger.info(f"      Headers: {headers}")
            else:
                logger.warning(f"❌ No results found for query: '{query}'")
                
        except Exception as e:
            logger.error(f"❌ Error testing query '{query}': {e}")

def print_statistics(kb):
    """Print knowledge base statistics"""
    logger.info("📊 Knowledge Base Statistics:")
    
    try:
        import sqlite3
        conn = sqlite3.connect(kb.db_path)
        cursor = conn.cursor()
        
        # Get content counts
        cursor.execute('SELECT COUNT(*) FROM text_content')
        text_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM image_content')
        image_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM table_content')
        table_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        embedding_count = cursor.fetchone()[0]
        
        # Get unique PDFs
        cursor.execute('SELECT COUNT(DISTINCT pdf_name) FROM text_content')
        pdf_count = cursor.fetchone()[0]
        
        conn.close()
        
        logger.info(f"   • PDFs processed: {pdf_count}")
        logger.info(f"   • Text segments: {text_count}")
        logger.info(f"   • Images: {image_count}")
        logger.info(f"   • Tables: {table_count}")
        logger.info(f"   • Total embeddings: {embedding_count}")
        
        # Index statistics
        if hasattr(kb, 'text_index') and kb.text_index:
            logger.info(f"   • Text index size: {kb.text_index.ntotal}")
        if hasattr(kb, 'image_index') and kb.image_index:
            logger.info(f"   • Image index size: {kb.image_index.ntotal}")
        if hasattr(kb, 'combined_index') and kb.combined_index:
            logger.info(f"   • Combined index size: {kb.combined_index.ntotal}")
            
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")

def main():
    """Main function"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 PDF Chat Bot - Corpus Builder")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Create directories
    os.makedirs(args.pdf_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Scan for PDF files
    pdf_files = scan_pdf_directory(args.pdf_dir)
    if not pdf_files:
        logger.error("❌ No PDF files found. Please add PDF files to the directory.")
        logger.info(f"💡 Add your PDF files to: {os.path.abspath(args.pdf_dir)}")
        sys.exit(1)
    
    # Check if knowledge base exists
    kb_path = os.path.join(args.output_dir, "..", "knowledge_base.db")
    if os.path.exists(kb_path) and not args.rebuild:
        logger.info("📚 Existing knowledge base found.")
        response = input("Do you want to rebuild it? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Using existing knowledge base.")
            
            # Load existing knowledge base for testing
            kb = MultimodalKnowledgeBase(args.output_dir)
            kb.build_faiss_indices()
            
            if args.test_query:
                test_system(kb, [args.test_query])
            else:
                test_system(kb)
            
            print_statistics(kb)
            return
    
    # Initialize processors
    logger.info("🔧 Initializing processors...")
    pdf_processor = PDFProcessor(args.pdf_dir, args.output_dir)
    kb = MultimodalKnowledgeBase(args.output_dir)
    
    # Process PDFs
    processed_data = process_pdfs(pdf_processor, pdf_files, args.max_files)
    
    if not processed_data:
        logger.error("❌ No PDFs were processed successfully.")
        sys.exit(1)
    
    # Build knowledge base
    if not build_knowledge_base(kb, processed_data):
        logger.error("❌ Failed to build knowledge base.")
        sys.exit(1)
    
    # Test system
    if args.test_query:
        test_system(kb, [args.test_query])
    else:
        test_system(kb)
    
    # Print statistics
    print_statistics(kb)
    
    logger.info("\n🎉 Corpus building completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Run: python app.py")
    logger.info("2. Open: http://localhost:5000")
    logger.info("3. Start chatting with your PDFs!")

if __name__ == "__main__":
    main()