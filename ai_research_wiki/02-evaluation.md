# 02. 模型评估与指标

[返回目录](README.md)

本章覆盖分类/回归指标、概率校准和模型比较。

## 1. 混淆矩阵、Precision、Recall、F1 与 Accuracy

### 混淆矩阵

|          | 预测为正 | 预测为负 |
| -------- | -------: | -------: |
| 真实为正 |       TP |       FN |
| 真实为负 |       FP |       TN |

### Accuracy

$$
Accuracy
=
\frac{TP+TN}
{TP+FP+TN+FN}.
$$

Accuracy 回答的是：

> 所有样本中，模型预测正确了多少？

### Precision

$$
Precision
=
\frac{TP}{TP+FP}.
$$

Precision 回答的是：

> 模型预测为正的样本中，有多少是真的？

### Recall

$$
Recall
=
\frac{TP}{TP+FN}.
$$

Recall 也叫 Sensitivity 或 True Positive Rate，回答的是：

> 所有真实正样本中，模型成功找到了多少？

### F1 Score

$$
F_1
=
\frac{2\cdot Precision\cdot Recall}
{Precision+Recall}.
$$

也可以直接根据混淆矩阵计算：

$$
F_1
=
\frac{2TP}{2TP+FP+FN}.
$$

F1 是 Precision 和 Recall 的调和平均。只有两者都较高时，F1 才会较高。

---

## 3. ROC-AUC 与 PR-AUC

Precision、Recall 和 F1 都依赖某个固定阈值。ROC-AUC 和 PR-AUC 则通过改变阈值，评估模型在所有阈值下的整体表现。

### ROC Curve

ROC 曲线的横轴是 False Positive Rate：

$$
FPR
=
\frac{FP}{FP+TN}.
$$

纵轴是 True Positive Rate，也就是 Recall：

$$
TPR
=
\frac{TP}{TP+FN}.
$$

从高到低改变分类阈值，会得到不同的：

$$
(FPR,TPR)
$$

点，这些点组成 ROC 曲线。

### ROC-AUC

ROC-AUC 是 ROC 曲线下面的面积。

它也可以解释为：

> 随机抽取一个正样本和一个负样本，模型给正样本更高分的概率。

对于分数相同的情况，通常按一半概率计算。

因此：

- ROC-AUC = 1：所有正样本都排在负样本前面。
- ROC-AUC = 0.5：排序能力接近随机。
- ROC-AUC < 0.5：排序方向可能反了。

ROC-AUC 只关心排序，不要求模型输出准确的概率。

---

## 4. Log Loss 与概率校准

二分类 Log Loss：

$$
\mathcal L
=-\frac1N\sum_i
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right]
$$

它不仅要求排序正确，也惩罚错误的概率置信度；对“高置信度但预测错误”惩罚尤其大。CTR、风险预测等需要概率参与后续决策的场景，Log Loss 通常比只看 AUC 更重要。

### 排序和校准的区别

- AUC：样本顺序是否正确。
- Calibration：预测 $p=0.2$ 的样本中，长期是否约有 20% 为正。

单调变换分数可能保持 AUC 不变，却破坏校准。常见校准指标与方法：

- Brier Score。
- Reliability Diagram。
- Expected Calibration Error（ECE，依赖分桶方式）。
- Platt Scaling、Isotonic Regression、Temperature Scaling。

### 数值稳定的 Log Loss

不要先显式计算 softmax/sigmoid 再取 log。使用 log-sum-exp 或框架的 `CrossEntropyLoss` / `BCEWithLogitsLoss`，避免概率下溢到 0 后出现 $\log 0$。

---

# 目录

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