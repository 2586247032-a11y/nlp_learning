import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import jieba
from tqdm import tqdm
import config
from tokenizer import JiebaTokenizer
# __file__ 是当前这个文件的绝对路径

def built_dataset(sentences, tokenizer):
    indexed_sentences = [tokenizer.encode(sentence) for sentence in sentences]
    # 二维列表，每一个元素是一个列表，列表中是这个句子的分好的词语的索引

    # 滑动窗口处理
    dataset = []
    # [{'input':[1,2,3,4,5],'target':6}, {'input':[2,3,4,5,6],'target':7}]
    for sentence in tqdm(indexed_sentences, desc="构建数据集"):
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset

def process():
    # 1.读取文件
    df = pd.read_json(config.RAW_DATA_DIR / 'synthesized_.jsonl' , lines = True,orient='records',).sample(frac = 0.2)
    # print(df.head())
    # 2.提取所有句子
    sentences = []
    for dialog in df['dialog']:   # dialog是一个列表，包含了所有的对话内容
        for sentence in dialog:   # sentence就是一句话"user2：是的，虽然工作忙，我还是会抽空参加一些。"
            sentences.append(sentence.split('：')[1])

    # print(sentences[0:20])
    # print(len(sentences))  句子总数

    # 3.划分数据集
    train_sentences , test_sentences = train_test_split(sentences, test_size = 0.2)

    # 4.构建词表（基于训练集构建）
    JiebaTokenizer.build_vocab(train_sentences, config.MODEL_DIR / "vocab.txt")


    # 6.构建训练集
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / "vocab.txt")
    train_dataset = built_dataset(train_sentences, tokenizer)
    # print(train_dataset[0:5])

    # 7.保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR/'train.jsonl',orient='records',lines=True)

    # 8.构建测试集
    test_dataset = built_dataset(test_sentences, tokenizer)
    # 9.保存测试集
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)
if __name__ == '__main__':
    process()