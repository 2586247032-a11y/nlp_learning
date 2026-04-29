import torch
import config
from model import ReviewAnalyzeModel
from src.dataset import get_dataloader
from predict import predict_1batch
from tokenizer import JiebaTokenizer

def evaluate(model, test_dataloader, device):
    total_count = 0
    correct_count = 0
    # 推理结果
    for inputs,targets in test_dataloader:
        inputs = inputs.to(device)
        # inputs.shape:[batch_size, seq_len]
        targets = targets.tolist()
        # targets.shape:[batch_size]  e.g.[1,3,5]
        batch_reasult = predict_1batch(model,inputs)
        # batch_reasult.shape:[batch_size]  e.g.[0.1,0.2,0.3]
        for result, target in zip(batch_reasult, targets):
            result = 1 if result>0.5 else 0
            if result==target:
                correct_count += 1
            total_count += 1
    return correct_count / total_count
def run_evaluate():
    # 1.确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载完成")
    # 3.定义模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size,padding_index=tokenizer.pad_token_index).to(device)
    # 加载参数
    model.load_state_dict(torch.load(config.MODELS_DIR / "best.pt"))
    print("模型加载完成")
    # 4.数据集
    test_dataloader = get_dataloader(train=False)

    # 5.评估逻辑
    acc = evaluate(model,test_dataloader,device)
    print("评估结果")
    print(f"top1_acc:{acc}")


if __name__ == "__main__":
    run_evaluate()