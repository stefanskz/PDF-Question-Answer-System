import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time
from main import initialize_qa_system
import shutil

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

qa_system = None

@app.route('/')
def index():
    return render_template('index.html')

def cleanup_old_files():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.getmtime(filepath) < time.time() - 3600:
            os.remove(filepath)
    
    chroma_dir = 'docs/chroma/'
    if os.path.exists(chroma_dir):
        try:
            shutil.rmtree(chroma_dir)
        except Exception as e:
            print(f"Error cleaning up Chroma directory: {e}")

@app.route('/upload', methods=['POST'])
def upload_file():
    cleanup_old_files()
    global qa_system
    
    print("Files in request:", request.files)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            qa_system = initialize_qa_system(filepath)
            return jsonify({'success': 'File uploaded successfully'})
        except Exception as e:
            print(f"Upload error: {str(e)}")
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global qa_system
    if qa_system is None:
        return jsonify({'error': 'Please upload a PDF first'})
    
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'})
    
    result = qa_system.invoke({"question": question})
    return jsonify({'answer': result['answer']})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)