import math

import torch
from torch import nn
import config

class PositionEncoding(nn.Module):
    def __init__(self, max_seq_len, dim_model):
        super().__init__()
        # 把要创建的二维矩阵定义成一个属性
        self.pe = torch.zeros([max_seq_len, dim_model], dtype=torch.float)
        for pos in range(max_seq_len):
            for i in range(0, dim_model, 2):
                self.pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / dim_model)))
                self.pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / dim_model)))
        # pe.shape = [max_seq_len, dim_model]
    def forward(self,x):
        # x.shape = [batch_size,seq_len,d_model]
        seq_len = x.shape[1]
        part_pe = self.pe[0:seq_len].to(x.device)
        # part_pe.shape = [seq_len, dim_model]   从所有的位置编码中取出x所对应需要的位置编码
        return x + part_pe  # 广播


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_idx, en_padding_idx):
        super().__init__()
        # 词嵌入
        self.zh_embedding = nn.Embedding(num_embeddings = zh_vocab_size,
                                         embedding_dim = config.DIM_MODEL,
                                         padding_idx=zh_padding_idx)
        self.en_embedding = nn.Embedding(num_embeddings = en_vocab_size,
                                         embedding_dim = config.DIM_MODEL ,
                                         padding_idx=en_padding_idx)
        # 位置编码
        self.position_encoding = PositionEncoding(max_seq_len=config.MAX_SEQ_LENGTH,
                                                 dim_model=config.DIM_MODEL)


        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                         nhead=config.NUM_HEAD,
                                         num_encoder_layers=config.NUM_ENCODER_LAYERS,
                                         num_decoder_layers=config.NUM_DECODER_LAYERS,
                                         batch_first=True)
        self.linear = nn.Linear(in_features = config.DIM_MODEL, out_features = en_vocab_size)

    # forward用来训练
    def forward(self, src, tgt, src_pad_mask, tgt_mask):
        memory = self.encode(src, src_pad_mask)
        return self.decode(memory, tgt, tgt_mask, src_pad_mask)  # memory_pad_mask和src_pad_mask一样，都是对编码器的pad进行掩盖

    # encode和decode用来推理
    def encode(self, src,src_pad_mask):
        # src.shape = [batch_size, src_len]
        # src_pad_mask.shape = [batch_size, src_len]
        # 先把token的id转成词向量
        embed = self.zh_embedding(src)
        # embed.shape = [batch_size, src_len, dim_model]

        #位置编码
        embed = self.position_encoding(embed)

        memory = self.transformer.encoder(src=embed, src_key_padding_mask=src_pad_mask)
        # memory.shape = [batch_size, src_len, dim_model]

        return memory


    def decode(self, memory, tgt, tgt_mask, memory_pad_mask):
        # tgt.shape = [batch_size, tgt_len]
        # memory.shape = [batch_size, src_len, dim_model]
        embed = self.en_embedding(tgt)
        embed = self.position_encoding(embed)
        # embed.shape = [batch_size, tgt_len, dim_model]

        output = self.transformer.decoder(tgt=embed,
                                          memory=memory,
                                          tgt_mask=tgt_mask,   #只在解码器前面用一次，推理时掩盖后面的序列
                                          memory_key_padding_mask=memory_pad_mask)    #每次交叉注意力都要用，来掩盖来自编码器的pad)
        # output.shape = [batch_size, tgt_len, dim_model]

        outputs = self.linear(output)
        # outputs.shape = [batch_size, tgt_len, en_vocab_size]
        return outputs




