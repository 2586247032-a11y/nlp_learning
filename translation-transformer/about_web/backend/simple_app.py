from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 前端文件路径
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

# 图片文件路径
image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'image'))

# 健康检查路由
@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

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

# 简单的翻译路由（模拟）
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Missing text parameter'}), 400
    
    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Text cannot be empty'}), 400
    
    # 模拟翻译
    return jsonify({'translation': f"Translated: {text}"})

if __name__ == '__main__':
    print("Starting simple Flask server on port 3001...")
    app.run(host='0.0.0.0', port=3001, debug=True)