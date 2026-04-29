import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel
import config


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader,desc = '训练'):
        inputs = inputs.to(device)   # inputs.shape = [batch_size, seq_len]
        targets = targets.to(device) # targets.shape = [batch_size]
        outputs = model(inputs)      # outputs.shape = [batch_size, 1]
        loss = loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)   # 返回平均损失


def train():
    # 1.设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 2.数据集
    dataloader = get_dataloader()
    # 3.分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")
    # 4.模型
    model = ReviewAnalyzeModel(tokenizer.vocab_size,tokenizer.pad_token_index).to(device)
    # 5.损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # 6.优化器
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.LEARNING_RATE)
    # 7.Tensorboard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S"))

    # 训练
    best_loss = float('inf')
    for epoch in range(1,config.EPOCHS+1):
        print(f'=========== Epoch{epoch} ===========')
        loss = train_one_epoch(model,dataloader,loss_fn,optimizer,device)
        print(f'Loss:{loss:.4f}')
        # 把损失记录到Tensorboard
        writer.add_scalar('Loss', loss, epoch)
        # 保存最优模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(),config.MODELS_DIR / "best.pt")
    writer.close()

if __name__ == '__main__':
    train()
