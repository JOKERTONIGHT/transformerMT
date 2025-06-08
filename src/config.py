import os

# 基础路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向MT根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')

# 配置参数
config = {
    # 数据文件
    'train_file': os.path.join(DATA_DIR, 'training.txt'),
    'val_file': os.path.join(DATA_DIR, 'validation.txt'),
    'test_file': os.path.join(DATA_DIR, 'testing.txt'),
    
    # 词汇表文件
    'src_word2int_path': os.path.join(DATA_DIR, 'word2int_en.json'),
    'tgt_word2int_path': os.path.join(DATA_DIR, 'word2int_cn.json'),
    'src_int2word_path': os.path.join(DATA_DIR, 'int2word_en.json'),
    'tgt_int2word_path': os.path.join(DATA_DIR, 'int2word_cn.json'),
    
    # 模型保存目录
    'model_save_dir': MODEL_SAVE_DIR,
    
    # 模型参数
    'd_model': 512,          # 特征维度
    'nhead': 8,              # 注意力头数
    'num_encoder_layers': 6, # 编码器层数
    'num_decoder_layers': 6, # 解码器层数
    'dim_feedforward': 2048, # 前馈网络维度
    'dropout': 0.05,          # dropout比例
    'max_length': 100,       # 最大序列长度
    
    # 特殊token索引
    'pad_idx': 0,
    'bos_idx': 1,
    'eos_idx': 2,
    'unk_idx': 3,
    
    # 训练参数
    'batch_size': 64,        # 批大小
    'learning_rate': 0.0001, # 学习率
    'epochs': 30,            # 训练轮数
    'clip': 1.0,             # 梯度裁剪
    'patience': 5,           # 早停耐心值
}
