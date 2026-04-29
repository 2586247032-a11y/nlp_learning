import torch
from torch import nn
from src import config


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx = padding_idx)
        self.gru = nn.GRU(input_size = config.EMBEDDING_DIM,
                          hidden_size = config.HIDDEN_SIZE,
                          batch_first = True)
    def forward(self,x):
        # x.shape = [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape = [batch_size, seq_len, embedding_dim]
        output, hidden = self.gru(embed)
        # 算出每个batch的序列长度
        lengths = (x != self.embedding.padding_idx).sum(dim = 1)
        # 获取序列最后一个时间步的输出，作为上下文向量
        lsat_hidden_step = output[torch.arange(output.shape[0]),lengths - 1]
        # lsat_hidden_step.shape = [batch_size, hidden_size]
        return lsat_hidden_step

class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        # vocab_size, padding_idx都是英文词表
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,
                               out_features=vocab_size)

    def forward(self,x,hidden_0):
        # forward方法用来处理一个时间步，每次处理一个token
        # x.shaoe = [batch_size, 1]    hidden.shape = [1, batch_size, hidden_size]
        embed = self.embedding(x)
        # embed.shape = [batch_size, 1, embedding_dim]
        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape = [batch_size, 1, hidden_size]
        output = self.linear(output)
        # output.shape = [batch_size, 1, vocab_size]
        return output, hidden_n

class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_idx, en_padding_idx):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size = zh_vocab_size, padding_idx = zh_padding_idx)
        self.decoder = TranslationDecoder(vocab_size = en_vocab_size, padding_idx = en_padding_idx)

















