"""
Translator module, for loading trained models and performing translations
"""
import torch
import json
from src.transformer_model import Transformer


class Translator:
    def __init__(self, config):
        """Initialize translator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载词汇表
        with open(config['src_word2int_path'], 'r', encoding='utf-8') as f:
            self.src_word2int = json.load(f)
        
        with open(config['tgt_int2word_path'], 'r', encoding='utf-8') as f:
            self.tgt_int2word = json.load(f)
        
        # 获取词汇表大小
        self.src_vocab_size = len(self.src_word2int)
        self.tgt_vocab_size = len(self.tgt_int2word)
        
        # 创建模型
        self.model = Transformer(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            pad_idx=config['pad_idx']
        ).to(self.device)
        
        # 特殊token
        self.pad_idx = config['pad_idx']
        self.bos_idx = config['bos_idx']
        self.eos_idx = config['eos_idx']
        self.unk_idx = config['unk_idx']
    
    def load_model(self, model_path):
        """Load pre-trained model

        Args:
            model_path: Path to model weights file
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded to {self.device} device")
    
    def translate(self, input_text, max_len=100):
        """Translate a single sentence

        Args:
            input_text: English sentence to translate (string)
            max_len: Maximum translation length

        Returns:
            Translated Chinese sentence (string)
        """
        self.model.eval()
        
        with torch.no_grad():
            # 将输入文本转换为token id
            tokens = []
            for word in input_text.strip().split():
                if word in self.src_word2int:
                    tokens.append(self.src_word2int[word])
                else:
                    tokens.append(self.unk_idx)
            
            # 转换为tensor
            src_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            
            # 编码
            memory, src_mask = self.model.encode(src_tensor)
            
            # 开始解码
            ys = torch.ones(1, 1).fill_(self.bos_idx).type(torch.long).to(self.device)
            
            for i in range(max_len - 1):
                # 解码一步
                out = self.model.decode(ys, memory, src_mask)
                
                # 获取最后一步的预测
                prob = out[:, -1]
                
                # 取概率最大的单词
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.unsqueeze(1)
                
                # 将预测添加到输出序列中
                ys = torch.cat([ys, next_word], dim=1)
                
                # 如果遇到EOS则停止
                if next_word.item() == self.eos_idx:
                    break
            
            # 将预测结果转换为文本
            pred_text = []
            for idx in ys[0].tolist():
                if idx == self.eos_idx:
                    break
                if idx != self.bos_idx and idx != self.pad_idx:
                    pred_text.append(self.tgt_int2word.get(str(idx), "UNK"))
            
            # 返回翻译结果
            return ' '.join(pred_text)
    
    def batch_translate(self, input_file, output_file):
        """Batch translate a file

        Args:
            input_file: Input file path
            output_file: Output file path
        """
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for i, line in enumerate(f_in):
                line = line.strip()
                if line:
                    translation = self.translate(line)
                    f_out.write(translation + '\n')
                    
                    # 显示进度
                    if (i + 1) % 10 == 0:
                        print(f"Translated {i+1} sentences...")
        
        print(f"Translation completed! Results saved to {output_file}")
