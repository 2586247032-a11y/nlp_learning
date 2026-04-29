import sys
import os
import traceback

# 添加src目录到Python路径
sys.path.append(os.path.abspath('src'))

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# 测试基本导入
try:
    import torch
    print("✓ torch imported successfully")
except Exception as e:
    print(f"✗ Error importing torch: {e}")
    traceback.print_exc()

try:
    import nltk
    print("✓ nltk imported successfully")
except Exception as e:
    print(f"✗ Error importing nltk: {e}")
    traceback.print_exc()

try:
    import jieba
    print("✓ jieba imported successfully")
except Exception as e:
    print(f"✗ Error importing jieba: {e}")
    traceback.print_exc()

# 测试模型和分词器加载
try:
    from model import TranslationModel
    print("✓ model imported successfully")
except Exception as e:
    print(f"✗ Error importing model: {e}")
    traceback.print_exc()

try:
    import config
    print("✓ config imported successfully")
    print(f"  Models directory: {config.MODELS_DIR}")
    print(f"  Models directory exists: {os.path.exists(config.MODELS_DIR)}")
except Exception as e:
    print(f"✗ Error importing config: {e}")
    traceback.print_exc()

try:
    from tokenizer import ChineseTokenizer, EnglishTokenizer
    print("✓ tokenizer imported successfully")
except Exception as e:
    print(f"✗ Error importing tokenizer: {e}")
    traceback.print_exc()

# 测试词表文件
try:
    zh_vocab_path = config.MODELS_DIR / 'zh_vocab.txt'
    en_vocab_path = config.MODELS_DIR / 'en_vocab.txt'
    model_path = config.MODELS_DIR / 'best.pt'
    
    print(f"  zh_vocab.txt exists: {os.path.exists(zh_vocab_path)}")
    print(f"  en_vocab.txt exists: {os.path.exists(en_vocab_path)}")
    print(f"  best.pt exists: {os.path.exists(model_path)}")
    
    if os.path.exists(zh_vocab_path):
        with open(zh_vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"  zh_vocab.txt has {len(lines)} lines")
    
    if os.path.exists(en_vocab_path):
        with open(en_vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"  en_vocab.txt has {len(lines)} lines")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  best.pt size: {file_size:.2f} MB")
        
except Exception as e:
    print(f"✗ Error checking files: {e}")
    traceback.print_exc()

# 测试模型加载
try:
    zh_tokenizer = ChineseTokenizer.from_vocab(zh_vocab_path)
    print("✓ ChineseTokenizer loaded successfully")
    print(f"  Chinese vocab size: {zh_tokenizer.vocab_size}")
except Exception as e:
    print(f"✗ Error loading ChineseTokenizer: {e}")
    traceback.print_exc()

try:
    en_tokenizer = EnglishTokenizer.from_vocab(en_vocab_path)
    print("✓ EnglishTokenizer loaded successfully")
    print(f"  English vocab size: {en_tokenizer.vocab_size}")
except Exception as e:
    print(f"✗ Error loading EnglishTokenizer: {e}")
    traceback.print_exc()

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device set to: {device}")
    
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_token_index,
        en_tokenizer.pad_token_index
    ).to(device)
    print("✓ Model created successfully")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✓ Model weights loaded successfully")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    traceback.print_exc()

print("\nDebug completed!")