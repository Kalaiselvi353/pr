#!/usr/bin/env python3
"""
Setup script for PDF Chat Bot
This script handles the installation and setup of the PDF chat bot system.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}")
        print(f"Error: {e.stderr}")
        return False

def check_system_requirements():
    """Check if system requirements are met"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python 3.8+ required. Current version: {python_version.major}.{python_version.minor}")
        return False
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("✅ pip is available")
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False
    
    return True

def install_system_dependencies():
    """Install system-level dependencies"""
    system = platform.system().lower()
    
    if system == "linux":
        print("🐧 Detected Linux system")
        # Install system dependencies for Linux
        commands = [
            "apt-get update",
            "apt-get install -y tesseract-ocr tesseract-ocr-eng",
            "apt-get install -y poppler-utils",
            "apt-get install -y libgl1-mesa-glx libglib2.0-0"
        ]
        
        for cmd in commands:
            if not run_command(f"sudo {cmd}", f"Installing system dependencies: {cmd}"):
                print("⚠️  Failed to install some system dependencies. You may need to install them manually.")
                
    elif system == "darwin":  # macOS
        print("🍎 Detected macOS system")
        commands = [
            "brew install tesseract",
            "brew install poppler"
        ]
        
        for cmd in commands:
            if not run_command(cmd, f"Installing system dependencies: {cmd}"):
                print("⚠️  Failed to install some system dependencies. Make sure Homebrew is installed.")
                
    elif system == "windows":
        print("🪟 Detected Windows system")
        print("⚠️  Please install the following manually:")
        print("   1. Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Poppler: https://github.com/oschwartz10612/poppler-windows/releases/")
        print("   3. Add both to your system PATH")
        input("Press Enter after installing the above dependencies...")

def create_directories():
    """Create necessary directories"""
    directories = [
        "pdfs",
        "processed_data/text",
        "processed_data/images", 
        "processed_data/tables",
        "models",
        "static/css",
        "static/js"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def install_python_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch with CPU support (you can modify for GPU)
    torch_command = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    if not run_command(torch_command, "Installing PyTorch"):
        print("❌ Failed to install PyTorch. Please install manually.")
        return False
    
    # Install other dependencies
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing other dependencies"):
        print("❌ Failed to install dependencies. Please check requirements.txt")
        return False
    
    return True

def download_models():
    """Download required models"""
    print("🤖 Downloading AI models...")
    
    # This will download models on first use
    try:
        import sentence_transformers
        import clip
        
        print("📥 Downloading sentence transformer model...")
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model ready")
        
        print("📥 Downloading CLIP model...")
        device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        print("✅ CLIP model ready")
        
    except Exception as e:
        print(f"⚠️  Error downloading models: {e}")
        print("Models will be downloaded on first use.")

def create_sample_config():
    """Create sample configuration files"""
    print("⚙️ Creating configuration files...")
    
    # Create .env file
    env_content = """# PDF Chat Bot Configuration
DEBUG=True
SECRET_KEY=your-secret-key-change-this-in-production
MAX_FILE_SIZE=104857600  # 100MB
UPLOAD_FOLDER=pdfs
PROCESSED_DATA_FOLDER=processed_data
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✅ Created .env file")

def run_initial_test():
    """Run initial system test"""
    print("🧪 Running initial system test...")
    
    try:
        # Test imports
        import fitz
        import cv2
        import pytesseract
        import sentence_transformers
        import torch
        import clip
        import faiss
        import camelot
        
        print("✅ All required modules can be imported")
        
        # Test Tesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 PDF Chat Bot Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please fix and try again.")
        sys.exit(1)
    
    # Install system dependencies
    install_system_dependencies()
    
    # Create directories
    create_directories()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies.")
        sys.exit(1)
    
    # Download models
    download_models()
    
    # Create configuration
    create_sample_config()
    
    # Run tests
    if run_initial_test():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Place your PDF files in the 'pdfs' folder")
        print("2. Run: python app.py")
        print("3. Open http://localhost:5000 in your browser")
        print("4. Upload PDFs and start chatting!")
        print("\n📖 For more information, check the README.md file")
    else:
        print("\n❌ Setup completed with warnings. Some features may not work correctly.")
        print("Please check the error messages above and install missing dependencies.")

if __name__ == "__main__":
    main()