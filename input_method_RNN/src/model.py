from torch import nn
import config

class InputMethodModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM)
        self.nn = nn.RNN(input_size = config.EMBEDDING_DIM,
                         hidden_size = config.HIDDEN_SIZE,
                         batch_first = True,)
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,
                                out_features=vocab_size)    # 因为最终要映射到词表，所以out_features=vocab_size

    def forward(self,x):
        # x.shape = [batch_size, seq_len]

        embed = self.embedding(x)
        # embed.shape = [batch_size, seq_len, embedding_dim]  # 经过embedding层之后，每个词的向量维度变为config.EMBEDDING_DIM

        output, hn = self.nn(embed)   # 把embed当作input,  H0不给的话默认全零
        # output.shape = [batch_size, seq_len, hidden_size]

        # 取出最后一个时间步的输出
        last_hidden_state = output[:,-1,:]
        # last_hidden_state.shape = [batch_size, hidden_size]

        output = self.linear(last_hidden_state)
        # output.shape = [batch_size, vocab_size]
        return output



