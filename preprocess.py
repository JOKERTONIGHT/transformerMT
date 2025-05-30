import json
import os
import argparse
from collections import Counter


def tokenize(text, language='en'):
    """根据语言简单分词"""
    if language == 'en':
        # 英文按空格分词并处理标点符号
        text = text.lower()
        for char in ',.?!;:\'\"()[]{}':
            text = text.replace(char, ' ' + char + ' ')
        return text.split()
    else:
        # 中文按字符分词
        return list(text.strip().replace(' ', ''))


def build_vocab(corpus_file, language='en', min_freq=2, max_vocab=50000):
    """构建词汇表"""
    counter = Counter()
    
    # 统计词频
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= (2 if language == 'en' else 1):
                text = parts[0] if language == 'en' else parts[1]
                tokens = tokenize(text, language)
                counter.update(tokens)
    
    # 构建词汇表，添加特殊标记
    word2int = {
        '<pad>': 0,
        '<bos>': 1,
        '<eos>': 2,
        '<unk>': 3,
    }
    
    # 按频率排序并添加到词汇表
    sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    # 只保留出现频率大于min_freq的词，并限制词汇表大小
    for i, (word, freq) in enumerate(sorted_words):
        if i >= max_vocab - 4 or freq < min_freq:  # 减去4是因为已添加了4个特殊标记
            break
        word2int[word] = i + 4  # 索引从4开始 (0-3已被特殊标记使用)
    
    # 创建反向映射
    int2word = {str(v): k for k, v in word2int.items()}
    
    return word2int, int2word


def preprocess_data(input_file, output_file, src_word2int, tgt_word2int):
    """预处理数据，将文本转换为token id"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                src_text, tgt_text = parts
                
                # 分词并转换为id
                src_tokens = tokenize(src_text, 'en')
                tgt_tokens = tokenize(tgt_text, 'zh')
                
                # 将OOV词替换为<unk>
                src_ids = [src_word2int.get(token, src_word2int['<unk>']) for token in src_tokens]
                tgt_ids = [tgt_word2int.get(token, tgt_word2int['<unk>']) for token in tgt_tokens]
                
                # 将id序列写入文件
                f_out.write(' '.join(map(str, src_ids)) + '\t' + ' '.join(map(str, tgt_ids)) + '\n')


def main():
    parser = argparse.ArgumentParser(description='预处理机器翻译数据')
    parser.add_argument('--raw_data', type=str, required=True, help='原始平行语料路径')
    parser.add_argument('--output_dir', type=str, default='./data', help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--min_freq', type=int, default=2, help='最小词频')
    parser.add_argument('--max_vocab', type=int, default=50000, help='最大词汇量')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 分割数据集
    lines = []
    with open(args.raw_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    train_size = int(total_lines * args.train_ratio)
    val_size = int(total_lines * args.val_ratio)
    
    train_data = lines[:train_size]
    val_data = lines[train_size:train_size + val_size]
    test_data = lines[train_size + val_size:]
    
    # 写入分割后的原始数据
    with open(os.path.join(args.output_dir, 'train_raw.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    
    with open(os.path.join(args.output_dir, 'val_raw.txt'), 'w', encoding='utf-8') as f:
        f.writelines(val_data)
    
    with open(os.path.join(args.output_dir, 'test_raw.txt'), 'w', encoding='utf-8') as f:
        f.writelines(test_data)
    
    # 构建词汇表
    print("构建英文词汇表...")
    en_word2int, en_int2word = build_vocab(
        os.path.join(args.output_dir, 'train_raw.txt'),
        language='en',
        min_freq=args.min_freq,
        max_vocab=args.max_vocab
    )
    
    print("构建中文词汇表...")
    zh_word2int, zh_int2word = build_vocab(
        os.path.join(args.output_dir, 'train_raw.txt'),
        language='zh',
        min_freq=args.min_freq,
        max_vocab=args.max_vocab
    )
    
    # 保存词汇表
    with open(os.path.join(args.output_dir, 'word2int_en.json'), 'w', encoding='utf-8') as f:
        json.dump(en_word2int, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'int2word_en.json'), 'w', encoding='utf-8') as f:
        json.dump(en_int2word, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'word2int_cn.json'), 'w', encoding='utf-8') as f:
        json.dump(zh_word2int, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(args.output_dir, 'int2word_cn.json'), 'w', encoding='utf-8') as f:
        json.dump(zh_int2word, f, ensure_ascii=False, indent=2)
    
    # 预处理训练集、验证集和测试集
    print("预处理训练集...")
    preprocess_data(
        os.path.join(args.output_dir, 'train_raw.txt'),
        os.path.join(args.output_dir, 'training.txt'),
        en_word2int,
        zh_word2int
    )
    
    print("预处理验证集...")
    preprocess_data(
        os.path.join(args.output_dir, 'val_raw.txt'),
        os.path.join(args.output_dir, 'validation.txt'),
        en_word2int,
        zh_word2int
    )
    
    print("预处理测试集...")
    preprocess_data(
        os.path.join(args.output_dir, 'test_raw.txt'),
        os.path.join(args.output_dir, 'testing.txt'),
        en_word2int,
        zh_word2int
    )
    
    print("预处理完成！")


if __name__ == '__main__':
    main()
