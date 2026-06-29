# 03. 数据处理与实验方法

[返回目录](README.md)

本章覆盖数据预处理、特征工程、数据切分与泄漏防范——也就是“把数据准备对、把实验做干净”。评估指标见 [02. 模型评估与指标](02-evaluation.md)，上线后的数据漂移见 [11. Production ML](11-production-ml.md)。

## 1. Validation Set vs Test Set

- **Validation set**：开发阶段用来选模型、调超参数、选阈值。
- **Test set**：最终估计泛化能力，不能反复看了再改模型。

如果一直根据 test set 改模型，本质上就是把 test set 也用来训练决策了，最终分数会虚高。

### 追问：数据少时怎么避免 test set 被用烂？

- train/validation 用来开发和调参。
- test set 最后少量使用。
- 如果数据很少，用 cross-validation 选模型，再留一个最终 holdout。
- 多次调参后最好重新构造一次 blind test 或线上 A/B。

---

## 2. Cross-Validation

K-fold cross-validation 把训练数据分成 $K$ 份，轮流拿一份做验证，最后平均指标。它适合数据量较小时评估模型稳定性。

注意：

- 分类任务常用 stratified K-fold 保持类别比例。
- 用户级数据要按 user/group 切，避免同一用户同时出现在 train 和 val。
- 时间序列不能随机 K-fold，要按时间向前验证（详见 [11. Production ML](11-production-ml.md) 的时间序列验证）。

---

## 3. Missing / Corrupted Data

先判断缺失是不是随机的：

- 数值特征：均值、中位数、分位数或模型填补；偏态和 outlier 多时常用中位数。
- 类别特征：填 `Unknown`，再做 one-hot 或 embedding。
- 缺失本身有含义：加 missing indicator。
- 脏数据：用范围、类型、枚举值、时间戳等规则拦截并记录。

所有 imputer 都只能在 train split 上 fit，再应用到 validation/test，避免 leakage。

### 追问：什么时候用中位数而不是均值填补？

当特征分布偏态或有较多 outlier 时，均值会被极端值拉偏，中位数更鲁棒、更能代表“典型值”。例如收入、价格、时长这类长尾分布，通常用中位数。近似对称、无极端值时，均值也可以。

---

## 4. Feature Scaling

Feature scaling 是把数值特征放到相近尺度，避免某个大数值特征支配距离或梯度。

- Standardization：$z=(x-\mu)/\sigma$。
- Min-Max：$(x-\min)/(\max-\min)$。
- Robust scaling：用 median / IQR，适合 outlier 多的场景。

KNN、K-Means、SVM、PCA、神经网络通常需要 scaling；决策树和随机森林通常不敏感。

### 追问：为什么树模型不需要 scaling？

树是按单个特征的阈值切分，只关心特征值的相对顺序，不涉及跨特征的距离或内积，所以单调缩放不改变分裂结果。而 KNN、SVM、K-Means 依赖距离，线性/神经网络依赖梯度尺度，都对量纲敏感。

---

## 5. Label Encoding vs One-Hot

- **Label encoding**：类别映射成整数，省空间，但可能引入假的大小关系。
- **One-hot**：每个类别一列，没有大小关系，但维度可能很高。
- **High-cardinality 类别**：可考虑 hashing、target encoding、embedding，但要特别防 leakage。

例子：颜色 `red/blue/green` 用 label encoding 变成 `0/1/2`，树模型还能处理，线性模型可能误以为 green > red。

### 追问：Target Encoding 怎么防止泄漏？

Target encoding 是用类别的历史 label 统计作为特征，例如某广告主历史 CTR。防泄漏方式：

- 训练集中用 out-of-fold encoding。
- 时间数据用过去窗口统计，不能用未来 label。
- 对低频类别做 smoothing。
- validation/test 只用训练期统计。

---

## 6. Outlier 怎么处理？

先分清楚是错误值还是少见但真实的样本。

- 错误值：修数据、删掉或按规则置缺失。
- 真实极端值：winsorize/capping、log transform、robust scaler，或使用对 outlier 不敏感的模型和 loss。

欺诈、风控、异常检测里，outlier 可能正是信号，不能机械删除。

### 追问：怎么自动识别 outlier？

- 单变量：z-score、IQR（$Q1-1.5\text{IQR}$、$Q3+1.5\text{IQR}$ 外）、分位数。
- 多变量：马氏距离、Isolation Forest、LOF、One-Class SVM、autoencoder 重构误差。

统计规则只是候选，是否处理仍要结合业务判断是否为真实信号。

---

## 7. 如何处理类别不平衡数据？

类别不平衡是指某一类样本远多于另一类（如欺诈占 0.1%）。处理分三个层面：

### 数据层面

- **过采样少数类**：随机复制，或用 **SMOTE** 在少数类样本间插值生成新样本（注意只能在训练集上做，且对高维/含噪数据可能引入伪样本）。
- **欠采样多数类**：随机或基于聚类丢弃多数类样本，省算力但可能丢信息。
- 组合采样，如 SMOTE + Tomek/ENN 清理边界。

### 算法层面

- **Class weight / cost-sensitive learning**：给少数类更大的损失权重（如 `class_weight='balanced'`、Focal Loss），不改变数据分布。
- 极端不平衡时，转成 **异常检测** 框架。

### 决策层面

- 调整分类阈值，而不是默认 0.5，按 PR 曲线或业务成本选。
- 概率重校准。

### 评估层面

不要看 Accuracy。用 Precision/Recall、F1、PR-AUC、固定容量下的业务收益，见 [02. 模型评估与指标](02-evaluation.md)。

### 面试怎么说？

> 处理不平衡有四个层面：数据上做重采样或 SMOTE，算法上用 class weight 或 Focal Loss，决策上调阈值，评估上换成 PR-AUC、F1。要强调采样只能在训练集做，且最终要看业务成本而不是 Accuracy。

---

## 8. Data Leakage

Data leakage 是训练时用了预测时拿不到的信息，离线指标会虚高，线上会崩。

常见来源：

- 切分前先对全量数据做标准化、填补、特征选择。
- 时间序列用了未来信息。
- 特征直接或间接包含 label。
- 同一用户/商品/样本泄漏到 train 和 test 两边。

防法：先 split，再 fit preprocessing；按时间或 group 切分；检查每个特征在预测时是否真的可用。

---

## 9. PCA 与维度灾难

维度越高，空间越稀疏，距离度量会变差，KNN/K-Means 这类距离模型尤其受影响，也更容易过拟合（维度灾难概念见 [00. 核心概念](00-ml-concepts.md)）。

PCA 是线性降维方法，找方差最大的正交方向，把数据投影到低维空间：

- 优点：降维、去相关、压缩噪声。
- 缺点：方向不一定和 label 最相关，可解释性变差，只能捕捉线性结构。

应对高维问题：特征选择、正则化、PCA/UMAP、更多数据，或用更合适的模型。

### 追问：PCA 前为什么要标准化？

PCA 找方差最大的方向，如果不标准化，量纲大的特征（如收入）会天然方差大，主成分会被它主导，而与单位选择无关的结构被淹没。所以通常先做 standardization，让各特征在可比尺度上贡献方差。
