from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Flask server is running!'})

if __name__ == '__main__':
    print("Starting Flask server on port 3000...")
    app.run(host='0.0.0.0', port=3000, debug=True)