import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer
import config



def process():
    print("开始处理数据")

    # 1.读取文件
    df = pd.read_csv(config.RAW_DATA_DIR/"online_shopping_10_cats.csv",usecols= ['label','review'],
                     encoding="utf-8").dropna()
    # print(df.head())

    # 2.划分数据集
    train_df,test_df = train_test_split(df, test_size=0.2,stratify=df["label"])   # stratify 分层抽样

    # 构建词表
    JiebaTokenizer.build_vocab(train_df['review'].tolist(),config.MODELS_DIR/'vocab.txt')  # 前面传的是列表，里面是一系列句子，后面传一个路径

    # 创建Tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR/'vocab.txt')

    # 计算序列长度
    # print(train_df['review'].apply(lambda x:len(tokenizer.tokenize(x))).quantile(0.95) ) # 计算每个句子的的token序列长度的百分之九十五中位数
    # 计算一次，记住就行

    # 构建训练集
    train_df['review'] = train_df['review'].apply(lambda x:tokenizer.encode(x,config.SEQ_LEN))# apply：接收的参数是一个函数，把这个函数应用到前面的每一个元素当中
    # 保存训练集
    train_df.to_json(config.PROCESSED_DATA_DIR/'train.jsonl',orient='records',lines = True)

    # 构建测试集
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x,config.SEQ_LEN))  # apply：接收的参数是一个函数，把这个函数应用到前面的每一个元素当中
    # 保存测试集
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)

    print("处理完成")



if __name__ == '__main__':
    process()












