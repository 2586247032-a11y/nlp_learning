import sys
import os

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# 测试基本导入
try:
    import torch
    print("torch imported successfully")
except Exception as e:
    print(f"Error importing torch: {e}")

try:
    import nltk
    print("nltk imported successfully")
except Exception as e:
    print(f"Error importing nltk: {e}")

try:
    import jieba
    print("jieba imported successfully")
except Exception as e:
    print(f"Error importing jieba: {e}")