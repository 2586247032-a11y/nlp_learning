import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src import config

# 1. 定义dataset类
# 作用是导入数据并转换成模型可以读取的张量格式---数据格式转换器
class TranslationDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')  # 将数据转换成成一个列表
        # 通过 pandas 读取 JSONL 格式的文件，并将其转换为字典列表（每个字典包含 review 和 label 键值对），存储在 self.data 中
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)  # 将输入转换成tensor张量
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input_tensor, target_tensor

def collate_fn(batch):
    # batch: 二元组列表：[(input_tensor, target_tensor)]
    # collate_fn(batch) 的作用是自定义如何从 batch 中构造数据张量，这里是对每一批进行填充
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]
    input_tensor = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensor = torch.nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=0)
    return input_tensor, target_tensor
    # pad_sequence 的作用是对不等长的序列进行填充（padding），使其达到相同长度。

# 2. 提供一个获取dataloader的方法
# 这个的作用是一批一批封装数据在DataLoader里面---批次数据打包器
def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = TranslationDataset(path)
    return DataLoader(dataset,batch_size = config.BATCH_SIZE, shuffle = True, collate_fn = collate_fn)   # 返回一个dataloader对象  shuffle是否打乱


if __name__ == '__main__':
    get_dataloader()
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    print(len(train_loader))  # 一共有多少批
    print(len(test_loader))

    for input_tensor, target_tensor in train_loader:
        print(input_tensor)   # [batch_size, seq_length]
        print(target_tensor)  # [batch_size]
        break