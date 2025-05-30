"""
Interactive English-Chinese Translation Demo
"""
import os
import sys
import torch
import json
import argparse

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.translator import Translator
from src.config import config


def main():
    """Run interactive translation demo"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Interactive English-Chinese Translation Demo')
    parser.add_argument('--model', type=str, default=None, 
                       help='Model path to use, defaults to best BLEU model')
    args = parser.parse_args()
    
    # 如果未指定模型路径，使用默认最佳模型
    model_path = args.model
    if model_path is None:
        model_path = os.path.join(config['model_save_dir'], 'best_model_bleu.pt')
    
    # 创建翻译器
    translator = Translator(config)
    
    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        translator.load_model(model_path)
    else:
        print("Model not found. Please train a model first or check configuration.")
        return
    
    # 交互式翻译
    print("=" * 60)
    print("Welcome to the Transformer-based English-Chinese Translation System!")
    print("Enter English sentences to translate, type 'q' or 'quit' to exit.")
    print("=" * 60)
    
    while True:
        # 获取输入
        input_text = input("\nEnter English text: ")
        
        # 检查是否退出
        if input_text.lower() in ['q', 'quit', 'exit']:
            print("Thank you for using the system. Goodbye!")
            break
        
        if not input_text.strip():
            continue
        
        # 翻译
        try:
            translation = translator.translate(input_text)
            print(f"Translation result: {translation}")
        except Exception as e:
            print(f"Error during translation: {e}")


if __name__ == '__main__':
    main()
