import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.abspath('src'))

print("Testing model loading...")
try:
    import torch
    from model import TranslationModel
    import config
    from tokenizer import ChineseTokenizer,EnglishTokenizer
    
    print("All modules imported successfully!")
    
    # 1.确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 2.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print('分词器加载成功')
    
    # 3.定义模型
    model = TranslationModel(zh_tokenizer.vocab_size,
                             en_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index).to(device)
    # 加载参数
    model.load_state_dict(torch.load(config.MODELS_DIR / "best.pt"))
    print('模型加载成功')
    
    print("All tests passed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()