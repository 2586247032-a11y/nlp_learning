import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict import ready, predict

print("Testing model loading...")
try:
    device, zh_tokenizer, en_tokenizer, model = ready()
    print("Model loaded successfully!")
    
    # Test translation
    test_text = "你好"
    result = predict(test_text, device, zh_tokenizer, en_tokenizer, model)
    print(f"Test translation: {test_text} -> {result}")
    print("Test passed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()