import argparse
import os
import torch
import json
from trainer import Trainer
from dataset import get_dataloader
from config import config
from evaluate import evaluate_translations
import nltk


def download_nltk_resources():
    """下载NLTK资源"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


def create_directories():
    """创建必要的目录"""
    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs('outputs', exist_ok=True)


def train_model():
    """训练模型"""
    print("Starting model training...")
    trainer = Trainer(config)
    trainer.train()
    print("Training completed!")


def test_model():
    """测试模型"""
    print("Testing model...")
    trainer = Trainer(config)
    
    # 加载最佳模型
    best_model_path = os.path.join(config['model_save_dir'], 'best_model_bleu.pt')
    
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model not found, using current model.")
    
    # 加载测试数据集
    test_loader = get_dataloader(
        file_path=config['test_file'],
        src_word2int_path=config['src_word2int_path'],
        tgt_word2int_path=config['tgt_word2int_path'],
        batch_size=config['batch_size'],
        shuffle=False,
        max_length=config['max_length']
    )
    
    # 计算BLEU分数
    bleu_score = trainer.calculate_bleu(test_loader)
    print(f"Test BLEU score: {bleu_score:.2f}")
    
    return bleu_score


def translate(input_text=None, input_file=None, output_file=None):
    """使用模型进行翻译"""
    
    # 加载词汇表
    with open(config['src_word2int_path'], 'r', encoding='utf-8') as f:
        src_word2int = json.load(f)
    
    with open(config['tgt_int2word_path'], 'r', encoding='utf-8') as f:
        tgt_int2word = json.load(f)
    
    # 加载模型
    trainer = Trainer(config)
    best_model_path = os.path.join(config['model_save_dir'], 'best_model_bleu.pt')
    
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model not found, using current model.")
    
    trainer.model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if input_text:
        # 将输入文本转换为token id
        tokens = []
        for word in input_text.strip().split():
            if word in src_word2int:
                tokens.append(src_word2int[word])
            else:
                tokens.append(config['unk_idx'])
        
        # 转换为tensor并预测
        src_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        pred_tensor = trainer.translate_sentence(src_tensor)
        
        # 将预测结果转换为文本
        pred_text = []
        for idx in pred_tensor[0].tolist():
            if idx == config['eos_idx']:
                break
            if idx != config['bos_idx'] and idx != config['pad_idx']:
                pred_text.append(tgt_int2word.get(str(idx), "UNK"))
        
        # 返回翻译结果
        result = ' '.join(pred_text)
        print(f"Source: {input_text}")
        print(f"Translation: {result}")
        
        return result
    
    elif input_file and output_file:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if line.strip():
                    # 将输入文本转换为token id
                    tokens = []
                    for word in line.strip().split():
                        if word in src_word2int:
                            tokens.append(src_word2int[word])
                        else:
                            tokens.append(config['unk_idx'])
                    
                    # 转换为tensor并预测
                    src_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
                    pred_tensor = trainer.translate_sentence(src_tensor)
                    
                    # 将预测结果转换为文本
                    pred_text = []
                    for idx in pred_tensor[0].tolist():
                        if idx == config['eos_idx']:
                            break
                        if idx != config['bos_idx'] and idx != config['pad_idx']:
                            pred_text.append(tgt_int2word.get(str(idx), "UNK"))
                    
                    # 写入翻译结果
                    result = ' '.join(pred_text)
                    f_out.write(result + '\n')
        
        print(f"Translated {input_file} to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Neural Machine Translation with Transformer')
    parser.add_argument('--mode', choices=['train', 'test', 'translate'], default='train',
                       help='运行模式 (train, test, translate)')
    parser.add_argument('--input_text', type=str, help='待翻译的文本')
    parser.add_argument('--input_file', type=str, help='待翻译的文件')
    parser.add_argument('--output_file', type=str, help='翻译结果输出文件')
    
    args = parser.parse_args()
    
    # 下载NLTK资源
    download_nltk_resources()
    
    # 创建必要的目录
    create_directories()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'test':
        test_model()
    elif args.mode == 'translate':
        if args.input_text:
            translate(input_text=args.input_text)
        elif args.input_file and args.output_file:
            translate(input_file=args.input_file, output_file=args.output_file)
        else:
            print("请提供待翻译的文本或文件路径")


if __name__ == '__main__':
    main()
