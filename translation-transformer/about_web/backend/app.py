from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import threading
import time

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 前端文件路径
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

# 图片文件路径
image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'image'))

# 模型和分词器（初始化为None）
device = None
zh_tokenizer = None
en_tokenizer = None
model = None
model_loaded = False
load_error = None

# 模型加载函数
def load_model_in_background():
    global device, zh_tokenizer, en_tokenizer, model, model_loaded, load_error
    print("Starting model loading in background...")
    try:
        import torch
        from predict import ready
        device, zh_tokenizer, en_tokenizer, model = ready()
        model_loaded = True
        print("Model loaded successfully!")
    except Exception as e:
        load_error = str(e)
        print(f"Error loading model: {load_error}")
        import traceback
        traceback.print_exc()

# 启动后台线程加载模型
load_thread = threading.Thread(target=load_model_in_background)
load_thread.daemon = True
load_thread.start()

# 健康检查路由
@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'load_error': load_error
    })

# 前端路由
@app.route('/')
def index():
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(frontend_dir, path)

# 图片路由
@app.route('/image/<path:path>')
def serve_images(path):
    return send_from_directory(image_dir, path)

# 翻译路由
@app.route('/translate', methods=['POST'])
def translate():
    if not model_loaded:
        if load_error:
            return jsonify({'error': f'Model loading failed: {load_error}'}), 500
        return jsonify({'error': 'Model not loaded yet, please try again later'}), 503
    
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Missing text parameter'}), 400
    
    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Text cannot be empty'}), 400
    
    try:
        from predict import predict
        result = predict(text, device, zh_tokenizer, en_tokenizer, model)
        return jsonify({'translation': result})
    except Exception as e:
        print(f"Error during translation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on port 3001...")
    app.run(host='0.0.0.0', port=3001, debug=True)