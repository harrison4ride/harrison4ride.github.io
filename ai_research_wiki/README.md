# AI Research Intern Interview Wiki

面向 **LLM / Agent / RAG / Alignment Research Intern** 面试的复习 Wiki。

本文档不建议逐字背诵。更有效的使用方式是：

1. 先用每题的“30 秒回答”建立主干。
2. 再掌握公式、工程权衡和常见追问。
3. 最后结合自己的项目，把答案改写为“问题 - 方法 - 实验 - 结论 - 反思”。

## 目录

### 第一部分：通用机器学习基础

| 章节 | 内容 |
| --- | --- |
| [00. 机器学习核心概念](00-ml-concepts.md) | AI/ML/DL、机器学习三大类型、参数 vs 超参数、归纳 vs 演绎、Bias-Variance、过拟合/欠拟合、维度灾难 |
| [01. 基础与神经网络机制](01-foundations.md) | L1/L2 正则与 Weight Decay、Loss/Huber、防过拟合、梯度下降、反向传播、梯度消失、激活函数、Softmax、Dropout、Perceptron、CNN、迁移学习 |
| [02. 模型评估与指标](02-evaluation.md) | 混淆矩阵、Type I/II、Precision/Recall/F1、ROC-AUC/PR-AUC、Log Loss 与校准、模型比较与统计检验 |
| [03. 数据处理与实验方法](03-data-experiment.md) | 验证集/测试集、交叉验证、缺失值、特征缩放、编码、Outlier、类别不平衡、数据泄漏、PCA |
| [04. 经典机器学习](04-classical-ml.md) | 线性/逻辑回归、MSE/KL、SVM、决策树、K-Means、EM、GMM、Random Forest、Naive Bayes、KNN、XGBoost |

### 第二部分：NLP 与大模型

| 章节 | 内容 |
| --- | --- |
| [05. NLP、RNN 与词向量](05-nlp-rnn.md) | RNN/LSTM、LSTM 参数量、梯度问题、Attention、Language Model、N-Gram、Word2Vec、Stemming/Lemmatization |
| [06. LLM 基础](06-llm-foundations.md) | Transformer、BERT、位置编码、RoPE、MHA/MQA/GQA、架构、Scaling Laws、解码、Tokenizer、MoE |
| [07. 训练与系统](07-training-and-systems.md) | 大模型训练挑战、并行策略、显存估算、稳定性、Qwen/DeepSeek、论文阅读方法 |
| [08. 对齐与 RLHF](08-alignment-and-rlhf.md) | SFT、RM、PPO、DPO、GRPO、DAPO、GSPO、RLAIF、Reward Hacking |

### 第三部分：应用系统与工程

| 章节 | 内容 |
| --- | --- |
| [09. Agent](09-agent.md) | Agent 定义、Harness、ReAct、规划、Memory、Tool Use、多智能体、安全 |
| [10. RAG 与搜索](10-rag-and-search.md) | 完整流水线、切块、Embedding、混合检索、重排、GraphRAG、自适应 RAG、评估 |
| [11. 项目经验与 Production ML](11-production-ml.md) | 离线上线差异、NaN/Inf、Data Shift、有限标注、特征缺失、监控、A/B、Feature Store、时间序列验证、公平性 |
| [12. ML Coding](12-ml-coding.md) | NumPy 手写 KMeans、Logistic/Linear Regression、Softmax、Attention、MHA、手写题检查清单 |
| [参考资料](references.md) | 论文、官方文档与进一步阅读 |

## 推荐复习顺序

### 第一轮：建立主干

`核心概念 -> 基础机制 -> 评估指标 -> 数据处理 -> 经典 ML -> NLP/RNN -> Transformer -> 训练目标 -> SFT/RLHF -> RAG -> Agent`

目标：每个主题能在 30 秒内说清楚“是什么、为什么、核心权衡”。

### 第二轮：补公式与工程细节

重点掌握：

- Attention、RoPE、交叉熵、Bradley-Terry、PPO/GRPO/DPO 目标函数。
- L1/L2、Softmax 反向传播、分类指标和概率校准。
- KV Cache、训练显存、数据/张量/流水线/序列并行。
- RAG 的 Recall@K、MRR、NDCG，以及生成端的 faithfulness。
- Agent 的成功率、步骤数、成本、延迟、恢复能力和安全性。

### 第三轮：绑定项目

每个项目准备以下六句话：

1. 场景和用户是谁？
2. 原方案为什么不够？
3. 你做了什么关键设计？
4. 数据和评估集怎么构造？
5. 指标提升多少，代价是什么？
6. 如果再做一次，你会改什么？

## 回答通用框架

技术题可以使用：

> **定义 -> 动机 -> 机制 -> 优缺点 -> 使用场景 -> 评估方式**

论文题可以使用：

> **问题 -> 既有方法的缺陷 -> 核心方法 -> 实验设置 -> 结果 -> 局限与可扩展方向**

系统设计题可以使用：

> **需求与约束 -> Baseline -> 模块划分 -> 数据流 -> 指标 -> 失败模式 -> 迭代方案**

## 关于“追问”

面试官的深度往往体现在追问。本 Wiki 中标记为 **“追问：…”** 的小节，是高频的第二、第三层问题。复习时应主动问自己：

- 这个方法的反例或失败模式是什么？
- 换一个约束（数据少、延迟严、类别不平衡）答案会怎么变？
- 这个指标什么时候会误导？
- “算法创新”和“工程实现”分别是什么？

## 面试中的高频失分点

- 只讲概念，不讲为什么这样设计。
- 把参数量、激活参数量、训练 FLOPs 和推理 FLOPs 混为一谈。
- 只报最终准确率，不讨论数据泄漏、成本、方差和统计显著性。
- 把 RAG、微调和 Prompt Engineering 当作互斥方案。
- 把 Agent 描述成“LLM 加工具”，却不讨论状态、反馈闭环和终止条件。
- 对最新方法只背缩写，不清楚它具体修改了哪个目标函数或训练环节。
- 声称某种能力在固定参数规模“必然涌现”。涌现没有统一参数阈值，且可能受指标离散化影响。
