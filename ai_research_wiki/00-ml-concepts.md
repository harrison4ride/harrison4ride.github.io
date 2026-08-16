# 00. 机器学习核心概念

[返回目录](README.md)

---

## 1. 机器学习的三种主要类型

### 30 秒回答

按监督信号来源划分：

- **监督学习（Supervised）**：训练数据有输入和标签，学习 $x\to y$ 的映射。分类、回归。
- **无监督学习（Unsupervised）**：只有输入没有标签，发现结构。聚类、降维、密度估计。
- **强化学习（Reinforcement）**：智能体与环境交互，通过奖励信号学习策略，目标是最大化长期回报。
- **自监督学习（Self-Supervised）**：从无标注数据中构造监督信号，例如语言模型预测下一个 token、BERT 预测被 mask 的词。它形式上像监督学习，但标签来自数据本身，不需要人工标注。

### 面试怎么说？

> 经典三分法是监督、无监督、强化学习。但要补一句：当前 LLM 的预训练本质是自监督，用 next-token prediction 把海量无标注文本变成监督信号，这是它能 scale 的关键。

## 2. Underfitting

训练集损失（Train Loss）**高**；验证集损失（Val Loss）**高**。这说明模型在当前设置下连训练数据本身蕴含的规律都无法有效捕捉（High Bias 状态）。

原因：
- **训练不充分（Optimization Issues）**：Epochs 太少，模型还没有收敛；学习率过小会导致训练过慢，过大则可能导致 Loss 震荡或无法下降；优化器或参数初始化不合适也可能让训练变得困难。
- **模型表达能力不足（Insufficient Capacity）**：模型太简单，无法表示数据中的复杂规律。例如，真实关系是 $y=x^2$，但使用的只是线性模型。
- **输入特征缺乏有效信息（Weak Features）**：输入中缺少与预测目标相关的信息，或者原始特征没有经过合适的编码和处理。
- **正则化过强（Excessive Regularization）**：L1，L2 Regularization, Weight Decay、Dropout 或数据增强过强，限制了模型对训练数据的拟合能力。
- **数据或标签存在问题（Data and Label Issues）**：标签错误、标签噪声过大、输入与标签没有正确对应，或者数据预处理破坏了重要信息。
- **实现存在错误（Implementation Issues）**：Loss、标签格式、输入处理、梯度更新或评估代码存在问题，导致模型无法正常学习。

常见处理顺序：

1. 检查简单基线能否正常训练，并尝试在一个很小的数据子集上过拟合。
2. 检查 loss、标签、输入和评估代码是否正确。
3. 延长训练或调整学习率、优化器和初始化。
4. 减弱过强的 weight decay、dropout 或其他正则化。
5. 增加模型容量，或加入更有信息量的特征。
6. 如果输入本身缺乏相关信息，重新设计数据或任务。

“能否过拟合一个很小的数据集”是一个实用的 sanity check：如果一个高容量模型连几十个样本都无法拟合，通常应优先怀疑实现或优化问题。

## 3. Overfitting

训练集损失（Train Loss）很低，但验证集损失（Validation Loss）明显更高。这说明模型在训练集上表现很好，却不能很好地处理没有见过的数据。

原因
- 模型容量过大（Excessive Capacity）：相对于数据量和任务难度，模型过于复杂，容易记住训练样本中的细节和噪声。
- 训练数据不足（Insufficient Data）：样本太少或覆盖范围有限，模型无法学到稳定、普遍的规律。
- 训练时间过长（Too Many Epochs）：模型先学到有效规律，之后继续拟合训练数据中的噪声。
- 正则化不足（Insufficient Regularization）：Weight Decay、Dropout 或数据增强太弱，无法限制模型对训练数据的过度拟合。
- 数据或标签噪声（Data and Label Noise）：模型记住了错误标签、异常样本或训练集中的偶然模式。
- 训练数据存在偏差（Spurious Correlation）：模型学到了只在训练集中成立的简单规律。例如，训练集中的狼经常出现在雪地里，模型可能根据“雪”判断是否有狼。
- 数据划分不合理（Improper Data Split）：训练集和验证集的来源、时间或类别比例不同，会造成类似 Overfitting 的表现，但根本原因是数据分布不一致。

常见处理顺序
- 检查训练集和验证集的数据划分是否合理，并排除重复样本。
- 确认训练集和验证集来自相近的数据分布。
- 根据 Validation Loss 保存最佳模型，使用 Early Stopping。
- 增加训练数据，尤其是模型表现较差的场景。
- 使用合理的数据增强。
- 增强 Weight Decay、Dropout 等正则化。
- 减少模型层数、隐藏维度或其他模型容量。
- 检查错误标签、异常样本和容易被模型利用的无关特征。
- 使用多个随机种子或交叉验证，确认结果不是偶然现象。

判断 Overfitting 时，不要只看训练结束时的结果。最好同时观察 Train Loss 和 Validation Loss 随 Epoch 的变化。如果 Train Loss 持续下降，而 Validation Loss 先下降后上升，通常说明模型开始过度拟合训练数据。

## 4. Bias-Variance 分解

Bias-Variance 分解用于解释模型的预测误差来自哪里。对于使用平方误差的回归任务，期望预测误差可以写为：

$$
\text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}.
$$

- **Bias**：不同训练集上模型平均预测与真实函数之间的系统偏差。
- **Variance**：更换训练集后模型预测的波动。
- **Irreducible noise**：数据本身无法由输入解释的噪声。

- Bias（偏差）：不同训练集上模型平均预测值与真实函数之间的差距。
-- 高 Bias 即 Underfitting：打靶时弹孔整体偏离靶心，说明模型假设太弱。
- Variance（方差）：更换训练集后，模型预测值本身的波动程度。
--高 Variance 即 Overfitting：打靶时弹孔极其分散，说明模型对数据抖动太敏感。

---

## 5. 维度灾难（Curse of Dimensionality）

维度越高，空间越**稀疏**，样本间距离趋于相近，基于距离的度量（KNN、K-Means）失效，模型也**更容易过拟合**，所需**样本量随维度指数增长**。

### 应对

特征选择、正则化、降维（PCA / UMAP）、收集更多数据，或换用对维度更鲁棒的模型。

## 目录

| 章节                                         |
| -------------------------------------------- |
| [00. 机器学习核心概念](00-ml-concepts.md)    |
| [01. 基础与神经网络机制](01-foundations.md)  |
| [02. 模型评估与指标](02-evaluation.md)       |
| [04. 经典机器学习](04-classical-ml.md)       |
| [05. NLP、RNN 与词向量](05-nlp-rnn.md)       |
| [06. LLM 基础](06-llm-foundations.md)        |
| [07. 训练与系统](07-training-and-systems.md) |
| [08. 对齐与 RLHF](08-alignment-and-rlhf.md)  |
| [09. 推理与部署](09-inference-and-serving.md) |
| [12. ML Coding](12-ml-coding.md)             |
| [参考资料](references.md)                    |