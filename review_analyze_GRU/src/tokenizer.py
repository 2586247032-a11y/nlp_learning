import jieba
from tqdm import tqdm
import config

class JiebaTokenizer:
    # 把所有和词表相关的内容和操作都封装到这个类里面
    unk_token = '<unk>'   # 类属性，不需要创建实例
    pad_token = '<pad>'
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index,word in enumerate(vocab_list)}
        self.index2word = {index: word for index,word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    # 把文本转换成索引序列
    def encode(self, text,seq_len):
        tokens = self.tokenize(text)

        # 截取或填充到指定的长度
        if len(tokens)>seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens)<seq_len:
            tokens += [self.pad_token] * (seq_len - len(tokens))

        indices = [self.word2index.get(token, self.unk_token_index) for token in tokens]
        return indices   # 一个列表

    # 构建词表的逻辑
    @classmethod
    def build_vocab(cls,sentences,vocab_path):
        vocab_set = set()
        for sentence in tqdm(sentences, desc='构建词表'):
            vocab_set.update([token for token in jieba.lcut(sentence) if token.strip != ''])  # 将句子中的词语添加到集合中，自动去重
                                                                                              # 去掉空格
        vocab_list = [cls.pad_token,cls.unk_token] + list(vocab_set)
        # print(vocab_list[0:10])

        # 保存词表(文本文件，一行一个词)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):   # 创建类
        with open(vocab_path, 'r',encoding="utf-8") as f:
            vocab_list = [readline.strip() for readline in f.readlines()]
        return cls(vocab_list)

if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")
    print(f"词表大小：{tokenizer.vocab_size}")
    print(f"特殊符号：{tokenizer.unk_token}")
    print(tokenizer.encode("今天天气不错", config.SEQ_LEN))