import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.abspath('src'))

print("Testing tokenizer loading...")

try:
    import config
    from tokenizer import ChineseTokenizer, EnglishTokenizer
    
    print("Imports successful")
    
    # 测试中文分词器
    print("Loading Chinese tokenizer...")
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    print(f"Chinese tokenizer loaded, vocab size: {zh_tokenizer.vocab_size}")
    
    # 测试英文分词器
    print("Loading English tokenizer...")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print(f"English tokenizer loaded, vocab size: {en_tokenizer.vocab_size}")
    
    # 测试分词功能
    test_text = "你好世界"
    tokens = zh_tokenizer.tokenize(test_text)
    print(f"Chinese tokenization: {test_text} -> {tokens}")
    
    test_text_en = "Hello world"
    tokens_en = en_tokenizer.tokenize(test_text_en)
    print(f"English tokenization: {test_text_en} -> {tokens_en}")
    
    print("Tokenizer test successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Tokenizer test completed!")