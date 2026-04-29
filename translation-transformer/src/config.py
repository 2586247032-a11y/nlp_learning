from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent

#
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
LOGS_DIR = ROOT_DIR / 'logs'

# 训练参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 30
MAX_SEQ_LENGTH = 128


# transformer模型结构
DIM_MODEL = 128     # 模型维度原始的用的512
NUM_HEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2