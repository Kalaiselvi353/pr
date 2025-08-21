# PDF Chat Bot - Multimodal Document Assistant

A sophisticated AI-powered chat bot that can understand and interact with PDF documents containing text, images, tables, and mixed content. Built with Flask, OpenAI CLIP, and advanced document processing techniques.

## 🌟 Features

### Multimodal Understanding
- **Text Processing**: Extracts and understands textual content with context awareness
- **Image Analysis**: Processes images using CLIP embeddings and OCR for comprehensive understanding
- **Table Extraction**: Intelligently extracts and processes tabular data
- **Smart Cropping**: Automatically removes headers, footers, and irrelevant content
- **Combined Search**: Searches across text, images, and tables simultaneously

### Advanced Document Processing
- **Intelligent Content Filtering**: Removes headers, footers, logos, and other non-essential content
- **Context-Aware Image Extraction**: Links images with surrounding text context
- **OCR Integration**: Extracts text from images and handwritten content
- **Table Understanding**: Converts tables to searchable text format
- **Multi-page Processing**: Handles complex multi-page documents

### User Experience
- **WhatsApp-like Interface**: Familiar chat interface for easy interaction
- **Real-time Processing**: Live status updates during document processing
- **Image Viewing**: Click to expand images in full-screen modal
- **Drag & Drop Upload**: Easy PDF upload with progress tracking
- **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Architecture

### Core Components

1. **PDF Processor** (`pdf_processor.py`)
   - Extracts text with positional information
   - Identifies and crops relevant images
   - Extracts tables using Camelot
   - Filters out headers, footers, and logos
   - Performs OCR on images

2. **Knowledge Base** (`knowledge_base.py`)
   - Creates multimodal embeddings using Sentence Transformers and CLIP
   - Stores processed content in SQLite database
   - Builds FAISS indices for fast similarity search
   - Handles combined text-image search queries

3. **Flask Backend** (`app.py`)
   - RESTful API for file upload and chat interactions
   - Real-time status monitoring
   - Error handling and logging
   - Image serving and management

4. **Frontend** (`templates/index.html`)
   - Modern WhatsApp-like chat interface
   - File upload with drag & drop
   - Real-time status updates
   - Image modal viewing
   - Responsive design

## 🚀 Quick Start

### 1. Automatic Setup (Recommended)

```bash
# Clone or download the project
cd pdf_chatbot

# Run the setup script
python setup.py
```

The setup script will:
- Check system requirements
- Install system dependencies (Tesseract, Poppler)
- Install Python packages
- Download AI models
- Create necessary directories
- Run initial tests

### 2. Manual Setup

If the automatic setup fails, follow these steps:

#### System Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
sudo apt-get install poppler-utils
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

**macOS:**
```bash
brew install tesseract
brew install poppler
```

**Windows:**
1. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/)
3. Add both to your system PATH

#### Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Running the Application

```bash
# Start the application
python app.py

# Open your browser and go to:
# http://localhost:5000
```

## 📁 Project Structure

```
pdf_chatbot/
├── app.py                      # Flask application
├── pdf_processor.py            # PDF processing pipeline
├── knowledge_base.py           # Multimodal knowledge base
├── setup.py                    # Automatic setup script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env                        # Configuration (created by setup)
├── pdfs/                       # PDF upload directory
├── processed_data/             # Processed content storage
│   ├── text/                   # Extracted text data
│   ├── images/                 # Processed images
│   └── tables/                 # Extracted tables
├── templates/                  # HTML templates
│   └── index.html              # Main chat interface
├── static/                     # Static assets
│   ├── css/
│   └── js/
└── knowledge_base.db           # SQLite database (created automatically)
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
DEBUG=True                      # Enable debug mode
SECRET_KEY=your-secret-key      # Flask secret key
MAX_FILE_SIZE=104857600         # Max upload size (100MB)
UPLOAD_FOLDER=pdfs              # PDF upload directory
PROCESSED_DATA_FOLDER=processed_data  # Processing output directory
```

### Model Configuration
The system uses these AI models:
- **Text Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Image Embeddings**: `ViT-B/32` (OpenAI CLIP)
- **OCR**: Tesseract OCR Engine

## 📖 Usage Guide

### 1. Upload PDFs
- Click "Choose Files" or drag & drop PDF files
- System automatically processes documents
- Monitor progress in the sidebar status panel

### 2. Ask Questions
- Type questions in natural language
- System searches across text, images, and tables
- Results include relevant content with source information

### 3. View Results
- Text content displayed with page references
- Images shown inline with context
- Tables displayed in formatted view
- Click images to view in full-screen modal

### Example Queries
- "What is the ring size chart?"
- "Show me images of jewelry specifications"
- "What are the purity marking guidelines?"
- "Find tables with size measurements"
- "What metals are mentioned in the document?"

## 🔍 How It Works

### Document Processing Pipeline

1. **PDF Parsing**: Extract raw content using PyMuPDF
2. **Content Classification**: Identify text, images, and tables
3. **Smart Filtering**: Remove headers, footers, and logos
4. **Image Processing**: Crop and enhance relevant images
5. **OCR Processing**: Extract text from images
6. **Table Extraction**: Process tabular data with Camelot
7. **Context Linking**: Associate images with surrounding text

### Search and Retrieval

1. **Query Processing**: Analyze user query intent
2. **Embedding Generation**: Create query embeddings
3. **Multimodal Search**: Search across text and image embeddings
4. **Result Ranking**: Score and rank results by relevance
5. **Response Generation**: Format and present results

### Embedding Strategy

- **Text Embeddings**: Sentence Transformers for semantic understanding
- **Image Embeddings**: CLIP for visual-semantic understanding
- **Combined Embeddings**: Concatenated text-image vectors for multimodal search
- **FAISS Indexing**: Fast approximate nearest neighbor search

## 🎯 Key Features Explained

### Smart Content Filtering
The system automatically identifies and excludes:
- Document headers and footers
- Company logos and watermarks
- Page numbers and metadata
- Navigation elements

### Intelligent Image Cropping
- Detects content boundaries using edge detection
- Removes whitespace and irrelevant borders
- Preserves important visual information
- Links images with contextual text

### Multimodal Search
- Searches text content semantically
- Matches images based on visual similarity
- Finds tables with relevant data
- Combines results for comprehensive answers

### Context Preservation
- Maintains document structure relationships
- Links images with surrounding text
- Preserves table relationships
- Provides source page references

## 🔧 Troubleshooting

### Common Issues

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check system dependencies (Tesseract, Poppler)

**Tesseract Not Found:**
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: Install from official site and add to PATH

**Memory Issues:**
- Large PDFs may require more RAM
- Process PDFs individually for better performance
- Consider using GPU for faster processing

**Upload Failures:**
- Check file size limits (default 100MB)
- Ensure PDF files are not corrupted
- Verify sufficient disk space

### Performance Optimization

**For Better Speed:**
- Use GPU if available (modify requirements for CUDA)
- Process smaller batches of PDFs
- Increase system RAM

**For Better Accuracy:**
- Use higher quality PDF scans
- Ensure good image resolution
- Provide clear, well-formatted documents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for CLIP model
- Sentence Transformers for text embeddings
- PyMuPDF for PDF processing
- Tesseract for OCR capabilities
- Camelot for table extraction

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review system requirements
3. Check the logs for error messages
4. Create an issue with detailed information

---

**Happy chatting with your PDFs! 🎉**