import torch
from torch import nn
import config

class ReviewAnalyzeModel(nn.Module):
    def __init__(self,vocab_size,padding_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM,padding_idx = padding_index)
        self.lstm = nn.LSTM(input_size = config.EMBEDDING_DIM,
                            hidden_size = config.HIDDEN_SIZE,
                            batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_SIZE,1)
    def forward(self,x:torch.Tensor):   # 声明一下x的形状，防止下面的sum不认识
        # x.shape:[batch_size,seq_len]
        embed = self.embedding(x)
        # embed.shape:[batch_size,seq_len,embedding_dim]
        output,(h_n,c_n) = self.lstm(embed)   # h_0和c_0不传，默认全零初始化
        # output.shape:[batch_size,seq_len,hidden_size]

        # 获取每个样本真实的最后一个token的隐藏状态
        batch_indexes = torch.arange(0,output.shape[0])
        #  x != self.embedding.padding_idx  把x中所有不是pad的元素全变成1，把所有pad的元素全变成0
        lengths = (x != self.embedding.padding_idx).sum(dim = 1)
        # 取最后一个时刻的输出作为句子的表示
        last_hidden = output[batch_indexes,lengths-1,:]
        # lsat_hidden.shape:[batch_size,hidden_size]相较于前面的output把seq_len维度取消了，
                                                  # 因为每个句子只取真实的最后一个token的隐藏状态（hidden_size维）
        output = self.linear(last_hidden).squeeze(1)
        # output.shape:[batch_size,1]    squeeze(1)：把1维度去掉   output.shape:[batch_size]
        return output

























