import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    def __init__(self, file_path, src_word2int_path, tgt_word2int_path, max_length=100):
        self.max_length = max_length
        
        # 加载词汇表
        with open(src_word2int_path, 'r', encoding='utf-8') as f:
            self.src_word2int = json.load(f)
        
        with open(tgt_word2int_path, 'r', encoding='utf-8') as f:
            self.tgt_word2int = json.load(f)
        
        # 特殊标记
        self.PAD_IDX = 0
        self.BOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        
        # 读取数据
        self.data_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    src, tgt = parts
                    self.data_pairs.append((src, tgt))
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data_pairs[idx]
        
        # 转换为token ids
        src_tokens = [self.src_word2int.get(word, self.UNK_IDX) for word in src_text.split()]
        tgt_tokens = [self.BOS_IDX] + [self.tgt_word2int.get(word, self.UNK_IDX) for word in tgt_text.split()] + [self.EOS_IDX]
        
        # 截断长度
        src_tokens = src_tokens[:self.max_length]
        tgt_tokens = tgt_tokens[:self.max_length]
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def collate_fn(batch):
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    src_text = [item['src_text'] for item in batch]
    tgt_text = [item['tgt_text'] for item in batch]
    
    # 填充序列
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return {
        'src': src_batch,
        'tgt': tgt_batch,
        'src_text': src_text,
        'tgt_text': tgt_text
    }


def get_dataloader(file_path, src_word2int_path, tgt_word2int_path, batch_size, 
                  shuffle=True, max_length=100):
    dataset = TranslationDataset(
        file_path=file_path,
        src_word2int_path=src_word2int_path,
        tgt_word2int_path=tgt_word2int_path,
        max_length=max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def get_vocab_size(word2int_path):
    with open(word2int_path, 'r', encoding='utf-8') as f:
        word2int = json.load(f)
    return len(word2int)
