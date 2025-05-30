import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import json


def load_vocab(int2word_path):
    """加载词汇表"""
    with open(int2word_path, 'r', encoding='utf-8') as f:
        int2word = json.load(f)
    return int2word


def tokens_to_words(tokens, int2word):
    """将token转换为词"""
    words = []
    for token in tokens:
        if str(token) in int2word:
            words.append(int2word[str(token)])
        else:
            words.append("UNK")
    return words


def calculate_bleu_score(references, candidates):
    """
    计算BLEU分数
    
    Args:
        references: 参考翻译列表的列表，如 [['我', '喜欢', '编程'], ['我', '爱', '编程']]
        candidates: 候选翻译列表，如 ['我', '喜欢', '程序设计']
        
    Returns:
        bleu分数 (0-1)
    """
    # 使用平滑函数，避免0分的情况
    smoothie = SmoothingFunction().method1
    
    # 计算BLEU-1到BLEU-4的权重
    weights = {
        'bleu-1': (1, 0, 0, 0),
        'bleu-2': (0.5, 0.5, 0, 0),
        'bleu-3': (0.33, 0.33, 0.33, 0),
        'bleu-4': (0.25, 0.25, 0.25, 0.25)
    }
    
    # 计算不同类型的BLEU分数
    results = {}
    for name, weight in weights.items():
        score = corpus_bleu(
            references, 
            candidates, 
            weights=weight, 
            smoothing_function=smoothie
        )
        results[name] = score * 100  # 转换为百分比
    
    return results


def evaluate_translations(reference_file, candidate_file, tgt_int2word_path, 
                         use_nltk=True, special_tokens=None):
    """
    评估翻译质量
    
    Args:
        reference_file: 包含参考翻译的文件路径 (每行一个句子)
        candidate_file: 包含候选翻译的文件路径 (每行一个句子)
        tgt_int2word_path: 目标语言int2word词汇表路径
        use_nltk: 是否使用NLTK的BLEU实现
        special_tokens: 特殊token的索引，如 {"BOS": 1, "EOS": 2, "PAD": 0}
        
    Returns:
        BLEU分数及其他评估指标
    """
    # 加载词汇表
    int2word = load_vocab(tgt_int2word_path)
    
    # 设置特殊token (如果未提供)
    if special_tokens is None:
        special_tokens = {"BOS": 1, "EOS": 2, "PAD": 0}
    
    # 读取参考翻译
    references = []
    with open(reference_file, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            references.append([words])  # 每个参考翻译是一个列表的列表
    
    # 读取候选翻译
    candidates = []
    with open(candidate_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 忽略空行
                words = line.strip().split()
                # 过滤掉特殊token (如果有)
                if words and words[0] == str(special_tokens.get("BOS")):
                    words = words[1:]
                if words and words[-1] == str(special_tokens.get("EOS")):
                    words = words[:-1]
                candidates.append(words)
    
    # 确保参考和候选翻译数量一致
    assert len(references) == len(candidates), "参考和候选翻译数量不一致"
    
    # 使用NLTK计算BLEU分数
    if use_nltk:
        bleu_scores = calculate_bleu_score(references, candidates)
        
        print("BLEU Scores:")
        for name, score in bleu_scores.items():
            print(f"{name}: {score:.2f}")
            
        # 返回BLEU-4作为主要评估指标
        return bleu_scores['bleu-4']
    
    return None
