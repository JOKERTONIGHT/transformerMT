# 基于Transformer的英中机器翻译系统

这是一个基于PyTorch实现的Transformer模型的神经机器翻译（NMT）系统，用于将英文翻译为中文。

## 项目结构

```
MT/
├── data/                   # 数据目录
│   ├── training.txt        # 训练数据
│   ├── validation.txt      # 验证数据
│   ├── testing.txt         # 测试数据
│   ├── word2int_en.json    # 英文词汇表（词到索引的映射）
│   ├── word2int_cn.json    # 中文词汇表（词到索引的映射）
│   ├── int2word_en.json    # 英文索引到词的映射
│   └── int2word_cn.json    # 中文索引到词的映射
├── models/                 # 保存训练好的模型
├── config.py               # 配置文件
├── dataset.py              # 数据加载和处理
├── transformer_model.py    # Transformer模型定义
├── trainer.py              # 模型训练与评估
├── evaluate.py             # BLEU评估工具
├── preprocess.py           # 数据预处理脚本
├── main.py                 # 主程序入口
└── README.md               # 项目说明
```

## 环境要求

- Python 3.6+
- PyTorch 1.8+
- NLTK 3.5+
- NumPy
- tqdm

## 安装依赖

```bash
pip install torch nltk numpy tqdm
```

## 使用方法

### 1. 数据预处理

如果您要使用自己的数据，可以使用预处理脚本进行处理：

```bash
python preprocess.py --raw_data path/to/parallel/corpus.txt --output_dir ./data
```

注意：原始数据应为平行语料，每行包含一对英文和中文句子，以制表符（\t）分隔。

### 2. 训练模型

```bash
python main.py --mode train
```

训练过程会自动保存最佳模型（根据验证集BLEU分数）到models目录。

### 3. 测试模型

```bash
python main.py --mode test
```

这将在测试集上评估模型，并计算BLEU分数。

### 4. 翻译文本

翻译单个句子：

```bash
python main.py --mode translate --input_text "This is a test sentence."
```

翻译整个文件：

```bash
python main.py --mode translate --input_file input.txt --output_file output.txt
```

## 模型架构

该项目实现了完整的Transformer模型，包括：

- 多头自注意力机制
- 位置编码
- 编码器-解码器架构
- 带掩码的解码器自注意力
- 交叉注意力机制
- 残差连接与层归一化

## 性能评估

使用BLEU（Bilingual Evaluation Understudy）作为主要评估指标，计算模型生成的翻译与参考翻译之间的相似度。

## 参考资料

- Vaswani, A., et al. (2017). Attention is all you need.
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch官方文档](https://pytorch.org/docs/stable/nn.html#transformer-layers)
