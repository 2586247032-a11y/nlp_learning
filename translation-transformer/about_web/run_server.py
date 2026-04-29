import subprocess
import time
import os

# 启动Flask服务器
print("Starting translation server...")
server_process = subprocess.Popen(
    ['python', 'src/backend/app.py'],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# 等待服务器启动
time.sleep(3)

# 检查服务器状态
if server_process.poll() is not None:
    print("Server failed to start:")
    print(server_process.stderr.read())
    exit(1)

print("Server started successfully!")
print("Access the translation app at: http://localhost:3001")
print("Press Ctrl+C to stop the server")

# 保持服务器运行
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping server...")
    server_process.terminate()
    server_process.wait()
    print("Server stopped.")