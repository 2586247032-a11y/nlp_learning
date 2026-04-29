import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.abspath('src'))

print("Testing minimal model loading...")

try:
    import torch
    from model import TranslationModel
    import config
    from tokenizer import ChineseTokenizer, EnglishTokenizer
    
    print("All imports successful")
    
    # 加载分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print("Tokenizers loaded")
    
    # 加载模型
    device = torch.device("cpu")
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_token_index,
        en_tokenizer.pad_token_index
    ).to(device)
    
    model.load_state_dict(torch.load(config.MODELS_DIR / "best.pt", map_location=device))
    print("Model loaded")
    
    # 测试预测
    def predict(text):
        indexes = zh_tokenizer.encode(text)
        input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)
        
        model.eval()
        with torch.no_grad():
            src_pad_mask = (input_tensor == model.zh_embedding.padding_idx)
            memory = model.encode(input_tensor, src_pad_mask)
            
            batch_size = input_tensor.shape[0]
            decoder_input = torch.full([batch_size, 1], en_tokenizer.sos_token_index, device=device)
            
            generated = []
            is_finished = torch.full([batch_size], False, device=device)
            
            for i in range(config.MAX_SEQ_LENGTH):
                tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1])
                decoder_output = model.decode(memory, decoder_input, tgt_mask, src_pad_mask)
                next_token_indexes = torch.argmax(decoder_output[:, -1, :], dim=-1, keepdim=True)
                generated.append(next_token_indexes)
                decoder_input = torch.cat([decoder_input, next_token_indexes], dim=-1)
                is_finished = is_finished | (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
                if is_finished.all():
                    break
            
            generated_tensor = torch.cat(generated, dim=1)
            generated_list = generated_tensor.tolist()
            
            for index, sentence in enumerate(generated_list):
                if en_tokenizer.eos_token_index in sentence:
                    eos_pos = sentence.index(en_tokenizer.eos_token_index)
                    generated_list[index] = sentence[:eos_pos]
            
            return en_tokenizer.decode(generated_list[0])
    
    # 测试翻译
    test_text = "你好"
    result = predict(test_text)
    print(f"Translation: {test_text} -> {result}")
    print("Test successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Minimal test completed!")