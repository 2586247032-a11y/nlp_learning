import torch
from torch import nn
from src import config

class Attention(nn.Module):
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden.shape = [batch_size, 1, hidden_size]
        attention_scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1,2))
        # attention_scores.shape = [batch_size, 1, seq_len]
        attention_weights = torch.softmax(attention_scores, dim = -1)
        return torch.bmm(attention_weights, encoder_outputs)


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
        # output.shape = [batch_size, seq_len, hidden_size]

        # 算出每个batch的序列长度
        lengths = (x != self.embedding.padding_idx).sum(dim = 1)
        # 获取序列最后一个时间步的输出，作为上下文向量
        lsat_hidden_step = output[torch.arange(output.shape[0]),lengths - 1]
        # lsat_hidden_step.shape = [batch_size, hidden_size]
        return output, lsat_hidden_step

class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        # vocab_size, padding_idx都是英文词表
        super().__init__()
        self.attention = Attention()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)
        self.linear = nn.Linear(in_features=2 * config.HIDDEN_SIZE,
                               out_features=vocab_size)

    def forward(self,x,hidden_0,encoder_outputs):
        # forward方法用来处理一个时间步，每次处理一个token
        # x.shaoe = [batch_size, 1]    hidden.shape = [1, batch_size, hidden_size]
        embed = self.embedding(x)
        # embed.shape = [batch_size, 1, embedding_dim]
        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape = [batch_size, 1, hidden_size]

        # 应用注意力机制(output,encoder_outputs) 解码器当前时间步的隐藏状态和编码器每一个时间步的隐藏状态
        context_vector = self.attention(output, encoder_outputs)
        # context_vector.shape = [batch_size, 1, hidden_size]

        # 拼接
        combined = torch.cat([output, context_vector], dim = -1)
        # combined.shape = [batch_size, 1, hidden_size * 2]

        output = self.linear(combined)
        # output.shape = [batch_size, 1, vocab_size]
        return output, hidden_n

class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_idx, en_padding_idx):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size = zh_vocab_size, padding_idx = zh_padding_idx)
        self.decoder = TranslationDecoder(vocab_size = en_vocab_size, padding_idx = en_padding_idx)

















