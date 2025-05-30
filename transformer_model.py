import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [max_seq_length, d_model] -> [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为buffer (不作为模型参数)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        return x + self.pe[:, :x.size(1), :]


# Transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # 多头注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = F.relu
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: [seq_length, batch_size, d_model]
        
        # 自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


# Transformer decoder layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 多头交叉注意力
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = F.relu
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt: [tgt_seq_length, batch_size, d_model]
        # memory: [src_seq_length, batch_size, d_model]
        
        # 自注意力
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 交叉注意力
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


# 完整的Transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_seq_length=5000, pad_idx=0):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer 编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Transformer 解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._reset_parameters()
        
    def _reset_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def make_src_mask(self, src):
        # src: [batch_size, src_seq_length]
        # 返回 padding mask
        src_mask = (src == self.pad_idx)  # [batch_size, src_seq_length]
        return src_mask  # [batch_size, src_seq_length]
    
    def make_tgt_mask(self, tgt):
        # tgt: [batch_size, tgt_seq_length]
        
        # 创建padding mask
        tgt_pad_mask = (tgt == self.pad_idx)  # [batch_size, tgt_seq_length]
        
        # 创建后续位置mask (让解码器不能看到未来位置)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        
        # 组合两个mask
        tgt_mask = tgt_pad_mask.unsqueeze(1).expand(-1, tgt_len, -1) | tgt_sub_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)
        
        return tgt_mask  # [batch_size, tgt_seq_length, tgt_seq_length]
    
    def encode(self, src, src_mask=None):
        # src: [batch_size, src_seq_length]
        
        # 生成源语言mask
        if src_mask is None:
            src_mask = self.make_src_mask(src)  # [batch_size, src_seq_length]
            
        # 词嵌入和位置编码
        src = self.src_embedding(src) * math.sqrt(self.d_model)  # [batch_size, src_seq_length, d_model]
        src = self.positional_encoding(src)  # [batch_size, src_seq_length, d_model]
        src = self.dropout(src)
        
        # 调整维度顺序，以适应PyTorch Transformer的输入
        src = src.permute(1, 0, 2)  # [src_seq_length, batch_size, d_model]
        
        # 过编码器层
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_key_padding_mask=src_mask)
            
        return src, src_mask
    
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        # tgt: [batch_size, tgt_seq_length]
        # memory: [src_seq_length, batch_size, d_model]
        
        # 生成目标语言mask
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)  # [batch_size, tgt_seq_length, tgt_seq_length]
        
        # 词嵌入和位置编码
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  # [batch_size, tgt_seq_length, d_model]
        tgt = self.positional_encoding(tgt)  # [batch_size, tgt_seq_length, d_model]
        tgt = self.dropout(tgt)
        
        # 调整维度顺序
        tgt = tgt.permute(1, 0, 2)  # [tgt_seq_length, batch_size, d_model]
        
        # 提取tgt的自注意力mask和padding mask
        tgt_pad_mask = tgt_mask[:, 0, :] if tgt_mask is not None else None  # [batch_size, tgt_seq_length]
        tgt_attn_mask = tgt_mask[0, :, :] if tgt_mask is not None else None  # [tgt_seq_length, tgt_seq_length]
        
        # 过解码器层
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                tgt, memory,
                tgt_mask=tgt_attn_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_mask
            )
            
        # 调整回原始维度顺序
        tgt = tgt.permute(1, 0, 2)  # [batch_size, tgt_seq_length, d_model]
        
        # 通过输出层得到预测
        output = self.output_layer(tgt)  # [batch_size, tgt_seq_length, tgt_vocab_size]
        
        return output
    
    def forward(self, src, tgt):
        # src: [batch_size, src_seq_length]
        # tgt: [batch_size, tgt_seq_length]
        
        # 编码过程
        memory, src_mask = self.encode(src)
        
        # 解码过程 (训练时使用teacher forcing)
        # 去掉目标序列的最后一个token
        tgt_input = tgt[:, :-1] if tgt.size(1) > 1 else tgt
        
        # 解码并预测
        output = self.decode(tgt_input, memory, src_mask)
        
        return output  # [batch_size, tgt_seq_length-1, tgt_vocab_size]
