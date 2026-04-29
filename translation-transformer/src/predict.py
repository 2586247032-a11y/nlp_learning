import torch
from model import TranslationModel
import config
from tokenizer import ChineseTokenizer,EnglishTokenizer

def ready():
    # 1.确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model.load_state_dict(torch.load(config.MODELS_DIR / "best.pt", map_location=device))
    print('模型加载成功')
    return device,zh_tokenizer,en_tokenizer,model

def predict_1batch(model,inputs,en_tokenizer):
    """
    批量预测
    :param model: 模型
    :param inputs:输入.shape:[batchsize,seq_len]
    :return: 预测结果.shape:[[*,*,*,*],[*,*,*,*,*,*,*],[*,*]]
    """
    model.eval()
    with torch.no_grad():
        # 编码阶段
        src_pad_mask = (inputs == model.zh_embedding.padding_idx)
        memory = model.encode(inputs, src_pad_mask)
        # memory.shape = [batch_size, src_seq_len, d_modle]

        # 解码
        batch_size = inputs.shape[0]
        device = inputs.device

        # decoder_hidden.shape = [1, batch_size, hidden_size]
        decoder_input = torch.full([batch_size,1],en_tokenizer.sos_token_index,device=device) # <sos>作为解码器第一步的输入
        # decoder_input.shape = [batch_size, tgt_seq_len]  tgt_seq_len初始是1，（sos）

        # 预测结果缓存
        generated = []

        # 记录每个样本是否已经生成eos
        is_finished = torch.full([batch_size],False,device=device)

        # 自回归生成
        for i in range(config.MAX_SEQ_LENGTH):
            # 解码
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1])  # 换成只有0和-inf（负无穷）的下三角矩阵
            decoder_output = model.decode(memory, decoder_input, tgt_mask, src_pad_mask)  # memory.shape = src_pad_mask
            # decoder_output.shape = [batch_size, tgt_seq_len, en_vocab_size]
            # 保存预测结果
            next_token_indexes = torch.argmax(decoder_output[:,-1,:],dim=-1,keepdim=True)
            # 取到上一步输出的最后一个token，把他拼到已经出来的所有token后面，基于拼好的预测下一个token
            # torch.argmax(input, dim) 的作用是返回指定维度上最大值的索引,在分类任务中，torch.argmax 用来获取预测类别
            # next_token_indexes.shape = [batch_size, 1]

            # 拼接预测结果
            generated.append(next_token_indexes)

            # 更新输入(decoder_input)
            decoder_input = torch.cat([decoder_input,next_token_indexes],dim = -1)

            # 判断是否应该结束,判断所有的样本是否都已经生成eos了
            is_finished = is_finished | (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            if is_finished.all():
                break
        # 处理预测结果
        # 整理预测结果的形状
        # 生成的结果generated是一个列表，列表的元素是一个张量，张量的shape为[tensor([batch_size,1])]
        generated_tensor = torch.cat(generated,dim = 1)
        # generated_tensor.shape = [batch_size, seq_len]
        generated_list = generated_tensor.tolist()
        # generated_list:[[*,*,*,eos,*],[*,*,*,eos,*,*,*,*],[*,*,eos,*,*,*,*]]
        # 删除eos之后的token id
        for index,sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]
        # generated_list:[[*,*,*],[*,*,*],[*,*]]
        return generated_list



def predict(text,device,zh_tokenizer,en_tokenizer,model):

    # 4.处理输入
    indexes = zh_tokenizer.encode(text)
    input_tensor = torch.tensor([indexes],dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # input_tensor.shape: [1(batch_size), seq_len]
    # 5.预测逻辑
    batch_reasult = predict_1batch(model,input_tensor,en_tokenizer)
    return en_tokenizer.decode(batch_reasult[0])


def run_predict():
    device, zh_tokenizer, en_tokenizer, model = ready()
    print("欢迎使用中英翻译v1.0模型，按q退出")
    while True:
        user_input = input("中文：")
        if user_input == "q":
            print("欢迎下次再来")
            break
        if user_input.strip() == '':
            print("请输入内容")
            continue

        result = predict(user_input, device, zh_tokenizer, en_tokenizer, model)
        print("英文：", result)
if __name__ == '__main__':
    run_predict()