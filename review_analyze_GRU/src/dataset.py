import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src import config

# 1. 定义dataset类
# 作用是导入数据并转换成模型可以读取的张量格式---数据格式转换器
class ReviewAnalyseDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')  # 将数据转换成成一个列表
        # 通过 pandas 读取 JSONL 格式的文件，并将其转换为字典列表（每个字典包含 review 和 label 键值对），存储在 self.data 中
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['review'], dtype=torch.long)  # 将输入转换成tensor张量
        target_tensor = torch.tensor(self.data[index]['label'], dtype=torch.float)
        return input_tensor, target_tensor


# 2. 提供一个获取dataloader的方法
# 这个的作用是一批一批封装数据在DataLoader里面---批次数据打包器
def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = ReviewAnalyseDataset(path)
    return DataLoader(dataset,batch_size = config.BATCH_SIZE, shuffle = True)   # 返回一个dataloader对象  shuffle是否打乱


if __name__ == '__main__':
    get_dataloader()
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    print(len(train_loader))  # 一共有多少批
    print(len(test_loader))

    for input_tensor, target_tensor in train_loader:
        print(input_tensor.shape)   # [batch_size, seq_length]
        print(target_tensor.shape)  # [batch_size]
        break