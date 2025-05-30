"""
English-Chinese Machine Translation System Entry Point
For training models, evaluating performance and batch translation
"""
import argparse
import os
import sys
import torch
import json
import nltk

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trainer import Trainer
from src.translator import Translator
from src.dataset import get_dataloader
from src.config import config
from src.evaluate import evaluate_translations


def download_nltk_resources():
    """Download NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


def create_directories():
    """Create necessary directories"""
    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs('outputs', exist_ok=True)


def train_model(run_name='default'):
    """Train the model
    
    Args:
        run_name: The name of the training run, used for saving models and charts
    """
    print(f"Starting model training '{run_name}'...")
    trainer = Trainer(config)
    trainer.train(run_name)
    print("Training completed!")


def test_model(model_path=None):
    """Test the model
    
    Args:
        model_path: Path to the model to test, uses the best BLEU model if None
    
    Returns:
        bleu_score: BLEU score on the test set
    """
    print("Testing model...")
    trainer = Trainer(config)
    
    # 如果未指定模型路径，使用默认最佳模型
    if model_path is None:
        model_path = os.path.join(config['model_save_dir'], 'best_model_bleu.pt')
    
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
    bleu_score = trainer.test(test_loader, model_path)
    print(f"Test set BLEU score: {bleu_score:.2f}")
    
    return bleu_score


def translate(input_text=None, input_file=None, output_file=None, model_path=None):
    """Translate text using the model
    
    Args:
        input_text: Single text to translate
        input_file: Path to file containing texts to translate
        output_file: Path for output translation results
        model_path: Model path, defaults to best BLEU model
    
    Returns:
        Translation result string if translating a single sentence
    """
    # 加载翻译器
    translator = Translator(config)
    
    # 如果未指定模型路径，使用默认最佳模型
    if model_path is None:
        model_path = os.path.join(config['model_save_dir'], 'best_model_bleu.pt')
    
    # 加载模型
    if os.path.exists(model_path):
        translator.load_model(model_path)
    else:
        print(f"Model not found: {model_path}")
        return None
    
    if input_text:
        # 单句翻译
        result = translator.translate(input_text)
        print(f"Source text: {input_text}")
        print(f"Translation result: {result}")
        return result
    
    elif input_file and output_file:
        # 批量翻译
        translator.batch_translate(input_file, output_file)
        print(f"Batch translation completed: {input_file} -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Transformer-based English-Chinese Machine Translation System')
    parser.add_argument('--mode', choices=['train', 'test', 'translate'], default='train',
                       help='Running mode (train, test, translate)')
    parser.add_argument('--run_name', type=str, default='default',
                       help='Training run name, used to distinguish results with different parameters')
    parser.add_argument('--input_text', type=str, help='Text to translate')
    parser.add_argument('--input_file', type=str, help='File to translate')
    parser.add_argument('--output_file', type=str, help='Output file for translation results')
    parser.add_argument('--model_path', type=str, help='Specify model path to load')
    
    args = parser.parse_args()
    
    # 下载NLTK资源
    download_nltk_resources()
    
    # 创建必要的目录
    create_directories()
    
    if args.mode == 'train':
        train_model(args.run_name)
    elif args.mode == 'test':
        test_model(args.model_path)
    elif args.mode == 'translate':
        if args.input_text:
            translate(input_text=args.input_text, model_path=args.model_path)
        elif args.input_file and args.output_file:
            translate(input_file=args.input_file, output_file=args.output_file, 
                     model_path=args.model_path)
        else:
            print("Please provide text or file path to translate")


if __name__ == '__main__':
    main()
