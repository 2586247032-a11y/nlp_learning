import jieba
import torch
from model import InputMethodModel
import config
from tokenizer import JiebaTokenizer
def ready():
    # 1.确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / 'vocab.txt')

    # 3.定义模型
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    # 加载参数
    model.load_state_dict(torch.load(config.MODEL_DIR / "model.pth"))
    return device,tokenizer,model

def predict_1batch(model,inputs):
    """
    批量预测
    :param model: 模型
    :param inputs:输入.shape:[batchsize,sql_len]
    :return: 预测结果.shape:[batch_size, 5]
    """
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        # output.shape:[batch_size, vocab_size]  这里的batch_size其实是1
    # 把output里面最高的前五个取出来
    top5_indexes = torch.topk(output, 5).indices
    # top5_indexes.shape: [batch_size，5]   ------  [[1,2,3,4,5]]
    top5_indexes_list = top5_indexes.tolist()  # 从张量形式转换为列表-------- [[1,2,3,4,5]]
    return top5_indexes_list

def predict(text,device,tokenizer,model):

    # 4.处理输入
    indexes = tokenizer.encode(text)
    input_tensor = torch.tensor([indexes],dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # 5.预测逻辑
    top5_indexes_list = predict_1batch(model,input_tensor)
    top5_tokens = [tokenizer.index2word[index] for index in top5_indexes_list[0]]
    return top5_tokens


def run_predict():
    device, tokenizer, model = ready()
    print("欢迎使用输入法预测系统，按q退出")
    input_history = ''
    while True:
        user_input = input("> ")
        if user_input == "q":
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue
        input_history += user_input
        print(f"输入历史：{input_history}")
        top5_tokens = predict(input_history, device, tokenizer, model)
        print(f"预测结果：{top5_tokens}")
if __name__ == '__main__':
    run_predict()