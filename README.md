# 基于Transformer的机器翻译系统

这是一个基于PyTorch实现的Transformer模型的机器翻译（NMT）系统，用于将英文翻译为中文。

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
├── plots/                  # 训练过程可视化图表
├── src/                    # 源代码目录
│   ├── config.py           # 配置文件
│   ├── dataset.py          # 数据加载和处理
│   ├── evaluate.py         # BLEU评估工具
│   ├── preprocess.py       # 数据预处理脚本
│   ├── trainer.py          # 模型训练与评估
│   ├── transformer_model.py # Transformer模型定义
│   └── translator.py       # 翻译器类，用于加载模型和执行翻译
├── main.py                 # 主程序入口（训练、测试、批量翻译）
├── demo.py                 # 交互式翻译演示
├── compare_params.py       # 不同参数配置比较工具
└── README.md               # 项目说明
```

## 环境要求

- Python 3.6+
- PyTorch 1.8+
- NLTK 3.5+
- NumPy
- tqdm
- matplotlib

## 安装依赖

```bash
pip install torch nltk numpy tqdm matplotlib
```

## 功能介绍

### 1. 数据预处理

如果您要使用自己的数据，可以使用预处理脚本进行处理：

```bash
python src/preprocess.py --raw_data path/to/parallel/corpus.txt --output_dir ./data
```

注意：原始数据应为平行语料，每行包含一对英文和中文句子，以制表符（\t）分隔。

### 2. 训练模型

使用 `main.py` 进行模型训练：

```bash
python main.py --mode train --run_name experiment1
```

参数说明:
- `--mode`: 操作模式，设为 `train` 进行训练
- `--run_name`: 训练任务名称，用于区分不同参数的训练结果

### 3. 测试模型

使用 `main.py` 在测试集上评估模型:

```bash
python main.py --mode test --model_path ./models/best_model_bleu.pt
```

参数说明:
- `--mode`: 操作模式，设为 `test` 进行测试
- `--model_path`: 可选，指定要评估的模型路径

### 4. 翻译文本

#### 交互式翻译

使用 `demo.py` 进行交互式翻译:

```bash
python demo.py --model ./models/best_model_bleu.pt
```

参数说明:
- `--model`: 可选，指定要使用的模型路径

#### 批量翻译

使用 `main.py` 进行批量翻译:

```bash
python main.py --mode translate --input_file ./data/input.txt --output_file ./outputs/output.txt
```

参数说明:
- `--mode`: 操作模式，设为 `translate` 进行翻译
- `--input_file`: 输入文件路径
- `--output_file`: 输出文件路径
- `--model_path`: 可选，指定要使用的模型路径

也可以翻译单个句子：

```bash
python main.py --mode translate --input_text "This is a test sentence."
```

### 5. 参数对比实验

使用 `compare_params.py` 比较不同参数配置的训练效果:

```bash
# 使用不同参数进行训练
python compare_params.py --mode train

# 比较已有的训练结果
python compare_params.py --mode compare --run_names baseline large_model small_model
```

参数说明:
- `--mode`: 运行模式，`train` 进行多组参数训练，`compare` 比较已有训练结果
- `--run_names`: 要比较的训练任务名称列表
- `--metrics`: 要比较的指标，可选 `bleu`, `train_loss`, `val_loss`, `learning_rate`

## 训练可视化

训练过程会自动生成以下图表，保存在 `plots` 目录:

1. 训练和验证损失曲线
2. BLEU分数变化曲线
3. 学习率变化曲线

这些图表可以帮助您:
- 监控模型训练过程
- 诊断过拟合或欠拟合问题
- 比较不同参数配置的效果
- 选择最佳模型

## 模型架构

该项目实现了完整的Transformer模型，包括：

- 多头自注意力机制
- 位置编码
- 编码器-解码器架构
- 带掩码的解码器自注意力
- 交叉注意力机制
- 残差连接与层归一化

## 性能评估

使用BLEU（Bilingual Evaluation Understudy，范围0~1，本项目使用百分数表示，结果范围为0~100）作为主要评估指标，计算模型生成的翻译与参考翻译之间的相似度。

- 初始模型测试集分数 21.05
- 添加平滑处理，表现有所下降；认为数据集制约了模型性能进一步提升

## 参考资料

- Vaswani, A., et al. (2017). Attention is all you need.
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch官方文档](https://pytorch.org/docs/stable/nn.html#transformer-layers)
