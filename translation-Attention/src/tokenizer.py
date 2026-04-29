import jieba
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer

from tqdm import tqdm
import config

class BaseTokenizer:
    # 把所有和词表相关的内容和操作都封装到这个类里面
    unk_token = '<unk>'   # 类属性，不需要创建实例
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index,word in enumerate(vocab_list)}
        self.index2word = {index: word for index,word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]
    @classmethod
    def tokenize(cla,text) -> list:
        pass

    # 把文本转换成索引序列
    def encode(self, text,add_sos_eos=False):
        tokens = self.tokenize(text)
        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        indices = [self.word2index.get(token, self.unk_token_index) for token in tokens]
        return indices   # 一个列表

    # 构建词表的逻辑
    @classmethod
    def build_vocab(cls,sentences,vocab_path):
        vocab_set = set()
        for sentence in tqdm(sentences, desc='构建词表'):
            vocab_set.update(cls.tokenize(sentence))
        vocab_list = [cls.pad_token,cls.unk_token,cls.sos_token,cls.eos_token] + list(vocab_set)
        # print(vocab_list[0:10])

        # 保存词表(文本文件，一行一个词)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):   # 创建类
        with open(vocab_path, 'r',encoding="utf-8") as f:
            vocab_list = [readline.strip() for readline in f.readlines()]
        return cls(vocab_list)
class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(csl,text)-> list:
        return list(text)
class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    @classmethod
    def tokenize(cls,text)-> list[str]:
        return cls.tokenizer.tokenize(text)
    def decode(self,indexes):
        # 把id索引转换成单词
        tokens = [self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)

if __name__ == '__main__':
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    wordlist = tokenizer.tokenize("Hello my name is Leo and I want to fuck you!")
    print(wordlist)
    print(detokenizer.detokenize(wordlist))