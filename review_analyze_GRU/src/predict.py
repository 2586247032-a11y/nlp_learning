import jieba
import torch
from model import ReviewAnalyzeModel
import config
from tokenizer import JiebaTokenizer
def ready():
    # 1.确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')

    # 3.定义模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_token_index).to(device)
    # 加载参数
    model.load_state_dict(torch.load(config.MODELS_DIR / "best.pt"))
    return device,tokenizer,model

def predict_1batch(model,inputs):
    """
    批量预测
    :param model: 模型
    :param inputs:输入.shape:[batchsize,sql_len]
    :return: 预测结果.shape:[batch_size]
    """
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        # output.shape:[batch_size]  这里的batch_size其实是1
    batch_reasult = torch.sigmoid(output)
    return batch_reasult.tolist()

def predict(text,device,tokenizer,model):

    # 4.处理输入
    indexes = tokenizer.encode(text,seq_len=config.SEQ_LEN)
    input_tensor = torch.tensor([indexes],dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [1(batch_size), sql_len]
    # 5.预测逻辑
    batch_reasult = predict_1batch(model,input_tensor)
    return batch_reasult[0]


def run_predict():
    device, tokenizer, model = ready()
    print("欢迎情感分析模型，按q退出")
    while True:
        user_input = input("> ")
        if user_input == "q":
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue

        result = predict(user_input, device, tokenizer, model)
        if result>0.5:
            print(f'正面评价<置信度{result}>')
        elif result<0.5:
            print(f'负面评价<置信度{1-result}>')
        else:
            print('中性评价')
if __name__ == '__main__':
    run_predict()