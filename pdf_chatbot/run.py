#!/usr/bin/env python3
"""
PDF Chat Bot - Easy Startup Script

This script provides an easy way to start the PDF Chat Bot system.
It handles initialization, checks dependencies, and starts the Flask app.

Usage:
    python run.py                    # Start with default settings
    python run.py --demo             # Run demo mode first
    python run.py --build-corpus     # Build corpus from PDFs first
    python run.py --port 8000        # Use custom port
    python run.py --debug            # Enable debug mode
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='PDF Chat Bot - Easy Startup',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run the Flask app on (default: 5000)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to run the Flask app on (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode first'
    )
    
    parser.add_argument(
        '--build-corpus',
        action='store_true',
        help='Build corpus from PDFs before starting'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run setup script first'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check system status, don\'t start the app'
    )
    
    return parser.parse_args()

def print_banner():
    """Print startup banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║               📄 PDF Chat Bot System 🤖                   ║
    ║                                                           ║
    ║          Multimodal Document Assistant                    ║
    ║          with Text, Image & Table Understanding           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if all dependencies are installed"""
    print("🔍 Checking system dependencies...")
    
    required_modules = [
        'flask', 'fitz', 'cv2', 'PIL', 'pytesseract', 
        'sentence_transformers', 'torch', 'clip', 'faiss', 
        'camelot', 'numpy', 'pandas', 'sqlite3'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'fitz':
                import fitz
            else:
                __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"  ❌ {module}")
    
    if missing_modules:
        print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Check Tesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"  ✅ Tesseract OCR: {version}")
    except Exception as e:
        print(f"  ❌ Tesseract OCR: {e}")
        print("Please install Tesseract OCR:")
        print("  - Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  - macOS: brew install tesseract")
        print("  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False
    
    print("✅ All dependencies are available!")
    return True

def check_directories():
    """Check and create necessary directories"""
    print("\n📁 Checking directories...")
    
    directories = [
        'pdfs',
        'processed_data',
        'processed_data/text',
        'processed_data/images',
        'processed_data/tables',
        'templates',
        'static'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"  ✅ Created: {directory}")
        else:
            print(f"  ✅ Exists: {directory}")

def check_system_status():
    """Check overall system status"""
    print("\n📊 System Status:")
    
    # Check PDFs
    pdf_dir = Path('pdfs')
    pdf_files = list(pdf_dir.glob('*.pdf')) if pdf_dir.exists() else []
    print(f"  📄 PDF files: {len(pdf_files)}")
    
    # Check knowledge base
    kb_path = 'knowledge_base.db'
    if os.path.exists(kb_path):
        print(f"  🧠 Knowledge base: ✅ Ready")
        
        # Get KB statistics
        try:
            import sqlite3
            conn = sqlite3.connect(kb_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM text_content')
            text_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM image_content')
            image_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM table_content')
            table_count = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"    • Text segments: {text_count}")
            print(f"    • Images: {image_count}")
            print(f"    • Tables: {table_count}")
            
        except Exception as e:
            print(f"    ⚠️  Error reading KB: {e}")
    else:
        print(f"  🧠 Knowledge base: ❌ Not found")
        if pdf_files:
            print(f"    💡 Run with --build-corpus to create it")
    
    # Check processed data
    processed_dir = Path('processed_data/text')
    processed_files = list(processed_dir.glob('*_processed.json')) if processed_dir.exists() else []
    print(f"  📝 Processed files: {len(processed_files)}")
    
    return len(pdf_files) > 0 or os.path.exists(kb_path)

def run_setup():
    """Run the setup script"""
    print("\n🔧 Running setup script...")
    try:
        result = subprocess.run([sys.executable, 'setup.py'], check=True)
        print("✅ Setup completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ setup.py not found")
        return False

def run_demo():
    """Run the demo script"""
    print("\n🎯 Running demo...")
    try:
        result = subprocess.run([sys.executable, 'demo.py'], check=True)
        print("✅ Demo completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ demo.py not found")
        return False

def build_corpus():
    """Build the corpus from PDFs"""
    print("\n🏗️  Building corpus...")
    try:
        result = subprocess.run([sys.executable, 'build_corpus.py'], check=True)
        print("✅ Corpus building completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Corpus building failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ build_corpus.py not found")
        return False

def start_flask_app(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask application"""
    print(f"\n🚀 Starting Flask app on {host}:{port}")
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    if debug:
        os.environ['FLASK_ENV'] = 'development'
        os.environ['FLASK_DEBUG'] = '1'
    
    try:
        # Import and run the Flask app
        from app import app
        
        print(f"\n🌐 Server starting...")
        print(f"   • URL: http://localhost:{port}")
        print(f"   • Debug mode: {'ON' if debug else 'OFF'}")
        print(f"   • Press Ctrl+C to stop")
        print("\n" + "="*50)
        
        app.run(host=host, port=port, debug=debug)
        
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        return False
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    print_banner()
    
    # Run setup if requested
    if args.setup:
        if not run_setup():
            print("❌ Setup failed. Please fix errors and try again.")
            sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed.")
        print("Please run: python setup.py")
        sys.exit(1)
    
    # Check directories
    check_directories()
    
    # Check system status
    system_ready = check_system_status()
    
    if args.check_only:
        print("\n✅ System check completed!")
        sys.exit(0)
    
    # Run demo if requested
    if args.demo:
        if not run_demo():
            print("❌ Demo failed.")
            sys.exit(1)
    
    # Build corpus if requested
    if args.build_corpus:
        if not build_corpus():
            print("❌ Corpus building failed.")
            sys.exit(1)
        system_ready = True
    
    # Warn if no content available
    if not system_ready:
        print("\n⚠️  Warning: No PDFs or knowledge base found!")
        print("The system will start, but you'll need to upload PDFs first.")
        print("\nOptions:")
        print("1. Add PDF files to the 'pdfs' folder and restart with --build-corpus")
        print("2. Upload PDFs through the web interface")
        print("3. Run the demo with --demo flag")
        
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("👋 Goodbye!")
            sys.exit(0)
    
    # Start Flask app
    try:
        start_flask_app(args.host, args.port, args.debug)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()