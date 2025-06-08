"""
Tool script for comparing training effects of different model parameters
Automatically train models with different parameter configurations and generate comparison charts
"""
import os
import sys
import copy
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from main import train_model, test_model


def load_history(run_name):
    """Load training history for a specific run
    
    Args:
        run_name: Run name
    
    Returns:
        history: Training history data dictionary
    """
    plots_dir = os.path.join(os.path.dirname(config['model_save_dir']), 'plots')
    history_path = os.path.join(plots_dir, f'{run_name}_history.json')
    
    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def compare_train_histories(run_names, labels, metric='bleu'):
    """Compare historical data from multiple training runs
    
    Args:
        run_names: List of run names
        labels: List of labels for each run
        metric: Metric to compare ('bleu', 'bleu4', 'train_loss', 'val_loss', 'learning_rate')
    """
    plt.figure(figsize=(12, 6))
    
    # Track the actual metric used for consistent labeling
    display_metric = metric
    
    for run_name, label in zip(run_names, labels):
        history = load_history(run_name)
        if history is None:
            continue
        
        # Handle different metric names
        actual_metric = metric
        if metric == 'bleu' and 'bleu' not in history:
            # If 'bleu' is requested but not available, try 'bleu4' as default
            if 'bleu4' in history:
                actual_metric = 'bleu4'
                display_metric = 'bleu4'  # Update display metric for consistent labeling
            else:
                print(f"Warning: Neither 'bleu' nor 'bleu4' found in history for {run_name}. Available metrics: {list(history.keys())}")
                continue
        elif actual_metric not in history:
            print(f"Warning: Metric '{actual_metric}' not found in history for {run_name}. Available metrics: {list(history.keys())}")
            continue
        
        epochs = list(range(1, len(history[actual_metric]) + 1))
        plt.plot(epochs, history[actual_metric], '-o', label=label)
    
    metric_labels = {
        'bleu': 'BLEU Score',
        'bleu1': 'BLEU-1 Score',
        'bleu2': 'BLEU-2 Score', 
        'bleu3': 'BLEU-3 Score',
        'bleu4': 'BLEU-4 Score',
        'train_loss': 'Training Loss',
        'val_loss': 'Validation Loss',
        'learning_rate': 'Learning Rate'
    }
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_labels.get(display_metric, display_metric))
    plt.title(f'Comparison of {metric_labels.get(display_metric, display_metric)} with Different Parameters')
    plt.grid(True)
    plt.legend()
    
    plots_dir = os.path.join(os.path.dirname(config['model_save_dir']), 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    comparison_name = '_'.join(run_names)
    plt.savefig(os.path.join(plots_dir, f'compare_{display_metric}_{comparison_name}.png'))
    plt.close()  # Close figure to release resources
    print(f"Saved comparison chart: compare_{display_metric}_{comparison_name}.png")


def compare_multiple_metrics(run_names, labels, metrics=None):
    """Compare multiple metrics across multiple training runs
    
    Args:
        run_names: List of run names
        labels: List of labels for each run
        metrics: List of metrics to compare
    """
    if metrics is None:
        metrics = ['bleu4', 'train_loss', 'val_loss']
    
    for metric in metrics:
        compare_train_histories(run_names, labels, metric)


def train_with_different_params(param_configs):
    """Train models with different parameter configurations
    
    Args:
        param_configs: List of parameter configurations, each element is a tuple (run_name, param_dict)
    
    Returns:
        run_names: List of run names
        labels: List of labels
    """
    run_names = []
    labels = []
    
    for run_name, params in param_configs:
        print(f"="*80)
        print(f"Starting training for '{run_name}' configuration:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        # 创建配置的深拷贝并更新
        current_config = copy.deepcopy(config)
        for k, v in params.items():
            current_config[k] = v
        
        # 更新全局配置
        for k, v in params.items():
            config[k] = v
            
        try:
            # 训练模型
            train_model(run_name)
            run_names.append(run_name)
            
            # 创建标签
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
            labels.append(f"{run_name} ({param_str})")
            
        except Exception as e:
            print(f"Error during training '{run_name}': {e}")
            
        # 恢复原始配置
        for k, v in current_config.items():
            config[k] = v
            
    return run_names, labels


def main():
    parser = argparse.ArgumentParser(description='Compare training effects of different model parameters')
    parser.add_argument('--mode', choices=['train', 'compare'], default='compare',
                       help='Running mode (train: train with different parameters, compare: compare existing models)')
    parser.add_argument('--run_names', type=str, nargs='+', 
                       help='List of run names to compare (only used in compare mode)')
    parser.add_argument('--metrics', type=str, nargs='+', default=['bleu4', 'train_loss', 'val_loss'],
                       help='List of metrics to compare')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 定义不同的参数配置
        param_configs = [
            # 格式: (运行名称, 参数字典)
            ('baseline', {}),  # Use default parameters
            ('large_model', {'d_model': 512, 'nhead': 8, 'num_encoder_layers': 6, 'num_decoder_layers': 6}),
            ('small_model', {'d_model': 256, 'nhead': 4, 'num_encoder_layers': 3, 'num_decoder_layers': 3}),
            ('high_dropout', {'dropout': 0.3}),
            ('low_dropout', {'dropout': 0.1}),
        ]
        
        # Train models with different parameters
        run_names, labels = train_with_different_params(param_configs)
        
        # Compare training results
        compare_multiple_metrics(run_names, labels, args.metrics)
        
    elif args.mode == 'compare':
        if not args.run_names or len(args.run_names) < 2:
            print("Please provide at least two run names to compare")
            return
            
        # Create labels (simply use run names as labels)
        labels = args.run_names
        
        # Compare training results
        compare_multiple_metrics(args.run_names, labels, args.metrics)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration_mins, duration_secs = divmod(end_time - start_time, 60)
    print(f"Execution completed, total time: {int(duration_mins)} min {duration_secs:.2f} sec")
