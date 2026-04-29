import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from dataset import get_dataloader
from model import InputMethodModel
import config
from tokenizer import JiebaTokenizer

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个epoch的逻辑
    :param model: 模型
    :param dataloader: 数据集
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return:
    """
    total_loss = 0
    model.train()
    for inputs,targets in tqdm(dataloader,desc='训练'):
        inputs, targets = inputs.to(device), targets.to(device)
        # inputs.shape: [batch_size, seq_length]
        # targets.shape: [batch_size]

        # 前向传播
        outputs = model(inputs)
        # output.shape: [batch_size, vocab_size]
        # 计算损失
        loss = loss_fn(outputs,targets)

        # 反向传播
        loss.backward()         # 反向传播
        optimizer.step()        # 更新参数
        optimizer.zero_grad()   # 清空梯度缓存

        total_loss += loss.item()  # 累加损失
    return total_loss / len(dataloader)


def train():
    # 1.确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2.加载数据集
    dataloader = get_dataloader()
    # 3.词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / 'vocab.txt')
    # 4.定义模型
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    # 5.损失函数
    loss_fn = torch.nn.CrossEntropyLoss()   # 交叉熵损失函数
    # 6.优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    # 7.tensorboard writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR/time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 8.训练
    best_loss = float('inf')   # 初始化为无穷大
    for epoch in range(1,1+config.EPOCHS):
        # 训练一个epoch的逻辑
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"epoch:{epoch},loss:{loss}")

        # 记录训练结果
        writer.add_scalar("loss",loss,epoch)

        # 8.保存模型
        # 一般情况下是有一个指标，来判断什么时候早停     这里暂时先不写
        if best_loss > loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODEL_DIR / "model.pth")    # model.state_dict() 返回模型的状态字典 模型的所有参数
            print("保存模型成功")

    writer.close()

if __name__ == '__main__':
    train()









