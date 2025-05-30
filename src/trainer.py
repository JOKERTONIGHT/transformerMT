"""
Model Training and Evaluation Module
"""
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu

# Configure matplotlib for non-interactive use
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.transformer_model import Transformer
from src.dataset import get_dataloader, get_vocab_size


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # 数据加载
        self.train_loader = get_dataloader(
            file_path=config['train_file'],
            src_word2int_path=config['src_word2int_path'],
            tgt_word2int_path=config['tgt_word2int_path'],
            batch_size=config['batch_size'],
            shuffle=True,
            max_length=config['max_length']
        )
        
        self.val_loader = get_dataloader(
            file_path=config['val_file'],
            src_word2int_path=config['src_word2int_path'],
            tgt_word2int_path=config['tgt_word2int_path'],
            batch_size=config['batch_size'],
            shuffle=False,
            max_length=config['max_length']
        )
        
        # 获取词汇表大小
        self.src_vocab_size = get_vocab_size(config['src_word2int_path'])
        self.tgt_vocab_size = get_vocab_size(config['tgt_word2int_path'])
        
        print(f"Source vocabulary size: {self.src_vocab_size}")
        print(f"Target vocabulary size: {self.tgt_vocab_size}")
        
        # 加载词汇对应表（用于BLEU评估和生成）
        with open(config['tgt_int2word_path'], 'r', encoding='utf-8') as f:
            self.int2word = json.load(f)
        
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
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 学习率调整
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=2
        )
        
        # 特殊token索引
        self.BOS_IDX = config['bos_idx']
        self.EOS_IDX = config['eos_idx']
        self.PAD_IDX = config['pad_idx']
        
        # 创建保存模型的目录
        if not os.path.exists(config['model_save_dir']):
            os.makedirs(config['model_save_dir'])
        
        # 创建保存图表的目录
        self.plots_dir = os.path.join(os.path.dirname(config['model_save_dir']), 'plots')
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'bleu': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        steps = len(self.train_loader)
        
        # 添加进度条
        print(f"Epoch {epoch}/{self.config['epochs']} - Training...")
        
        for i, batch in enumerate(self.train_loader):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # 前向传播
            output = self.model(src, tgt)
            
            # 计算损失（忽略BOS token)
            output = output.contiguous().view(-1, self.tgt_vocab_size)
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(output, tgt)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip'])
            
            # 更新参数
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if (i + 1) % 50 == 0 or (i + 1) == steps:
                print(f"Epoch {epoch}, Step {i+1}/{steps}, Loss: {loss.item():.4f}, Progress: {(i+1)/steps*100:.1f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch}, Average training loss: {avg_loss:.4f}")
        
        # 记录当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history['learning_rate'].append(current_lr)
        
        return avg_loss
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # 前向传播
                output = self.model(src, tgt)
                
                # 计算损失
                output = output.contiguous().view(-1, self.tgt_vocab_size)
                tgt = tgt[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(output, tgt)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def translate_sentence(self, src_tensor, max_len=100):
        self.model.eval()
        
        with torch.no_grad():
            src_tensor = src_tensor.to(self.device)
            
            # 编码
            memory, src_mask = self.model.encode(src_tensor)
            
            # 开始解码
            ys = torch.ones(src_tensor.size(0), 1).fill_(self.BOS_IDX).type(torch.long).to(self.device)
            
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
                if (next_word == self.EOS_IDX).sum() == src_tensor.size(0):
                    break
        
        return ys
    
    def calculate_bleu(self, loader):
        self.model.eval()
        references = []
        candidates = []
        
        with torch.no_grad():
            for batch in loader:
                src = batch['src'].to(self.device)
                tgt = batch['tgt']
                
                # 获取预测翻译
                pred = self.translate_sentence(src)
                
                # 转换为词列表 (移除BOS, EOS, PAD)
                pred_text = []
                for sent in pred.tolist():
                    sent_tokens = []
                    for idx in sent:
                        if idx == self.EOS_IDX:
                            break
                        if idx != self.BOS_IDX and idx != self.PAD_IDX:
                            sent_tokens.append(self.int2word.get(str(idx), "UNK"))
                    pred_text.append(sent_tokens)
                
                # 准备参考翻译
                tgt_text = []
                for sent in tgt.tolist():
                    sent_tokens = []
                    for idx in sent:
                        if idx == self.EOS_IDX:
                            break
                        if idx != self.BOS_IDX and idx != self.PAD_IDX:
                            sent_tokens.append(self.int2word.get(str(idx), "UNK"))
                    tgt_text.append([sent_tokens])  # corpus_bleu需要二维列表
                
                references.extend(tgt_text)
                candidates.extend(pred_text)
        
        # 计算BLEU分数
        bleu_score = corpus_bleu(references, candidates) * 100
        print(f"BLEU score: {bleu_score:.2f}")
        
        return bleu_score
    
    def train(self, run_name='default'):
        """Train the model

        Args:
            run_name: Run name, used for saving charts and records
        """
        best_val_loss = float('inf')
        best_bleu = 0
        
        print(f"Starting training task '{run_name}'...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            
            # 在验证集上评估
            val_loss = self.evaluate()
            
            # 计算BLEU分数
            bleu = self.calculate_bleu(self.val_loader)
            
            # 记录指标
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['bleu'].append(bleu)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.config['model_save_dir'], f'{run_name}_best_model_loss.pt'))
                print("Saved model with lowest loss!")
            
            if bleu > best_bleu:
                best_bleu = bleu
                torch.save(self.model.state_dict(), os.path.join(self.config['model_save_dir'], f'{run_name}_best_model_bleu.pt'))
                print("Saved model with highest BLEU!")
            
            # 每个epoch结束绘制一次图表
            self.plot_history(run_name)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            print(f"Epoch {epoch} completed in {epoch_mins}m {epoch_secs:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu:.2f}")
            print("=" * 60)
        
        # 保存最终模型
        torch.save(self.model.state_dict(), os.path.join(self.config['model_save_dir'], f'{run_name}_final_model.pt'))
        print(f"Training completed! Final model saved. Best BLEU: {best_bleu:.2f}, Best Val Loss: {best_val_loss:.4f}")
        
    def test(self, test_loader=None, model_path=None):
        """Evaluate model on test set
        
        Args:
            test_loader: Test data loader, created if None
            model_path: Model path, uses best BLEU model if None
            
        Returns:
            bleu_score: BLEU score on test set
        """
        if test_loader is None:
            test_loader = get_dataloader(
                file_path=self.config['test_file'],
                src_word2int_path=self.config['src_word2int_path'],
                tgt_word2int_path=self.config['tgt_word2int_path'],
                batch_size=self.config['batch_size'],
                shuffle=False,
                max_length=self.config['max_length']
            )
        
        # 加载模型
        if model_path is None:
            model_path = os.path.join(self.config['model_save_dir'], 'best_model_bleu.pt')
            
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using current model...")
        
        # 计算测试集上的BLEU分数
        bleu_score = self.calculate_bleu(test_loader)
        
        return bleu_score
    
    def plot_history(self, run_name='default'):
        """Plot training history charts
        
        Args:
            run_name: Run name, used for saving charts
        """
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        # 创建图表目录
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'{run_name}_loss_curve.png'))
        plt.close()  # 关闭图表释放资源
        
        # 绘制BLEU分数曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['bleu'], 'g-', label='BLEU Score')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU')
        plt.title('Validation BLEU Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'{run_name}_bleu_curve.png'))
        plt.close()  # 关闭图表释放资源
        
        # 绘制学习率变化曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['learning_rate'], 'm-', label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Changes')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(os.path.join(self.plots_dir, f'{run_name}_lr_curve.png'))
        plt.close()  # 关闭图表释放资源
        
        # 保存训练历史数据
        history_path = os.path.join(self.plots_dir, f'{run_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
            
        print(f"Training history plots saved to {self.plots_dir}")
