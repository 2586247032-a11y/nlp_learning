import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import ChineseTokenizer,EnglishTokenizer
from model import TranslationModel
import config


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader,desc = '训练'):
        encoder_inputs = inputs.to(device)   # inout.shape = [batch_size, src_seq_len]
        targets = targets.to(device)   # targets.shape = [batch_size, tgt_seq_len]
        decoder_inputs = targets[:,:-1].long()  # decoder_inputs.shape = [batch_size, seq_len]
        decoder_targets = targets[:,1:].long()   # targets_output.shape = [batch_size, seq_len]
        # 前向传播
        src_pad_mask = (encoder_inputs == model.zh_embedding.padding_idx)
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1])
        decoder_outputs = model(encoder_inputs,decoder_inputs,src_pad_mask,tgt_mask)
        # decoder_outputs.shape = [batch_size, seq_len, en_vocab_size]

        # 损失函数有他的输入格式要求，需要reshape一下
        loss = loss_fn(decoder_outputs.reshape(-1,decoder_outputs.shape[-1]),decoder_targets.reshape(-1))

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    # 1.设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 2.数据集
    dataloader = get_dataloader()
    # 3.分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / "zh_vocab.txt")
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / "en_vocab.txt")
    # 4.模型
    model = TranslationModel(zh_tokenizer.vocab_size,
                             en_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,
                             en_tokenizer.pad_token_index).to(device)
    # 5.损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
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
