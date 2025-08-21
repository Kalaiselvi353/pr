from flask import Flask, request, jsonify, render_template, send_file
import os
import json
import base64
from datetime import datetime
from pdf_processor import PDFProcessor
from knowledge_base import MultimodalKnowledgeBase
import logging
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize components
pdf_processor = None
knowledge_base = None

# Configuration
PDF_UPLOAD_FOLDER = 'pdfs'
PROCESSED_DATA_FOLDER = 'processed_data'
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_system():
    """Initialize the PDF processing and knowledge base system"""
    global pdf_processor, knowledge_base
    
    try:
        # Create directories if they don't exist
        os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)
        
        # Initialize components
        pdf_processor = PDFProcessor(PDF_UPLOAD_FOLDER, PROCESSED_DATA_FOLDER)
        knowledge_base = MultimodalKnowledgeBase(PROCESSED_DATA_FOLDER)
        
        # Check if knowledge base exists and load it
        if os.path.exists(knowledge_base.db_path):
            knowledge_base.build_faiss_indices()
            logger.info("Knowledge base loaded successfully")
        else:
            logger.info("No existing knowledge base found. Upload PDFs to build one.")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return False

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF files"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        processed_files = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(PDF_UPLOAD_FOLDER, filename)
                
                # Save uploaded file
                file.save(filepath)
                
                # Process the PDF
                try:
                    pdf_data = pdf_processor.process_single_pdf(filepath)
                    processed_files.append({
                        'filename': filename,
                        'pages': pdf_data['total_pages'],
                        'status': 'processed'
                    })
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    processed_files.append({
                        'filename': filename,
                        'status': 'error',
                        'error': str(e)
                    })
        
        # Rebuild knowledge base with new files
        if processed_files:
            try:
                knowledge_base.build_knowledge_base()
                return jsonify({
                    'message': 'Files uploaded and processed successfully',
                    'files': processed_files
                })
            except Exception as e:
                logger.error(f"Error building knowledge base: {e}")
                return jsonify({
                    'message': 'Files processed but error building knowledge base',
                    'files': processed_files,
                    'error': str(e)
                }), 500
        
        return jsonify({'error': 'No valid PDF files provided'}), 400
        
    except Exception as e:
        logger.error(f"Error in upload_pdf: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat queries"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if not knowledge_base or not hasattr(knowledge_base, 'combined_index'):
            return jsonify({
                'response': 'Knowledge base is not ready. Please upload some PDF files first.',
                'results': []
            })
        
        # Search the knowledge base
        results = knowledge_base.search(query, top_k=10)
        
        if not results:
            return jsonify({
                'response': 'I couldn\'t find any relevant information for your query. Please try rephrasing your question.',
                'results': []
            })
        
        # Format response
        response_parts = []
        formatted_results = []
        
        # Group results by type and relevance
        text_results = [r for r in results if r['type'] == 'text' and r['score'] > 0.3]
        image_results = [r for r in results if r['type'] == 'image' and r['score'] > 0.2]
        table_results = [r for r in results if r['type'] == 'table' and r['score'] > 0.3]
        
        # Build response from text results
        if text_results:
            for result in text_results[:3]:  # Top 3 text results
                content = result['content']
                response_parts.append(content['text'])
                
                formatted_results.append({
                    'type': 'text',
                    'content': content['text'],
                    'page': content['page_number'] + 1,
                    'score': result['score']
                })
        
        # Add table results
        if table_results:
            for result in table_results[:2]:  # Top 2 table results
                content = result['content']
                response_parts.append(f"Table data:\n{content['text_representation']}")
                
                formatted_results.append({
                    'type': 'table',
                    'content': content['text_representation'],
                    'headers': content['headers'],
                    'data': content['data'],
                    'page': content['page_number'] + 1,
                    'score': result['score']
                })
        
        # Add image results
        if image_results:
            for result in image_results[:3]:  # Top 3 image results
                content = result['content']
                
                # Convert image to base64 for display
                try:
                    with open(content['path'], 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        
                    formatted_results.append({
                        'type': 'image',
                        'content': content['context_text'] + ' ' + content['ocr_text'],
                        'image_data': img_data,
                        'filename': content['filename'],
                        'page': content['page_number'] + 1,
                        'score': result['score']
                    })
                    
                    # Add context to response
                    if content['context_text'] or content['ocr_text']:
                        context = (content['context_text'] + ' ' + content['ocr_text']).strip()
                        if context:
                            response_parts.append(f"Related image content: {context}")
                            
                except Exception as e:
                    logger.error(f"Error processing image {content['path']}: {e}")
        
        # Create final response
        if response_parts:
            response_text = '\n\n'.join(response_parts)
        else:
            response_text = 'I found some relevant content but couldn\'t extract readable text. Please check the images and tables below.'
        
        return jsonify({
            'response': response_text,
            'results': formatted_results,
            'total_results': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<path:filename>')
def serve_image(filename):
    """Serve processed images"""
    try:
        image_path = os.path.join(PROCESSED_DATA_FOLDER, 'images', filename)
        if os.path.exists(image_path):
            return send_file(image_path)
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Get system status"""
    try:
        # Count PDFs
        pdf_count = len([f for f in os.listdir(PDF_UPLOAD_FOLDER) if f.endswith('.pdf')]) if os.path.exists(PDF_UPLOAD_FOLDER) else 0
        
        # Check knowledge base status
        kb_ready = knowledge_base and hasattr(knowledge_base, 'combined_index') and knowledge_base.combined_index
        
        # Count processed content
        content_counts = {'text': 0, 'images': 0, 'tables': 0}
        if knowledge_base and os.path.exists(knowledge_base.db_path):
            import sqlite3
            conn = sqlite3.connect(knowledge_base.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM text_content')
            content_counts['text'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM image_content')
            content_counts['images'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM table_content')
            content_counts['tables'] = cursor.fetchone()[0]
            
            conn.close()
        
        return jsonify({
            'status': 'ready' if kb_ready else 'not_ready',
            'pdf_count': pdf_count,
            'content_counts': content_counts,
            'knowledge_base_ready': kb_ready
        })
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rebuild', methods=['POST'])
def rebuild_knowledge_base():
    """Rebuild the knowledge base from existing PDFs"""
    try:
        if not pdf_processor:
            return jsonify({'error': 'PDF processor not initialized'}), 500
        
        # Process all PDFs in the directory
        processed_data = pdf_processor.process_all_pdfs()
        
        # Rebuild knowledge base
        knowledge_base.build_knowledge_base()
        
        return jsonify({
            'message': f'Knowledge base rebuilt successfully with {len(processed_data)} PDFs',
            'processed_files': len(processed_data)
        })
        
    except Exception as e:
        logger.error(f"Error rebuilding knowledge base: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize the system
    if initialize_system():
        logger.info("System initialized successfully")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize system")
        exit(1)