# 02. 模型评估与指标

[返回目录](README.md)

本章覆盖分类/回归指标、概率校准和模型比较。数据切分与交叉验证见 [03. 数据处理与实验方法](03-data-experiment.md)。

## 1. 混淆矩阵与 Type I / Type II 错误

二分类混淆矩阵包含 TP、FP、TN、FN：

- **TP**：真实正，预测正。
- **FP**：真实负，预测正。
- **TN**：真实负，预测负。
- **FN**：真实正，预测负。

### Type I 与 Type II 错误

- **Type I error（第一类错误，假阳性 FP）**：把负样本判成正，即“误报 / 假警报”。统计上是错误地拒绝了为真的原假设。
- **Type II error（第二类错误，假阴性 FN）**：把正样本判成负，即“漏报”。统计上是错误地接受了为假的原假设。

记忆法：Type I = False Positive = 误报；Type II = False Negative = 漏报。

### 追问：两类错误哪个更严重？

取决于业务成本。

- 疾病筛查：漏诊（Type II / FN）后果严重，优先降低 FN，即提高 Recall。
- 垃圾邮件拦截：误杀正常邮件（Type I / FP）代价高，优先降低 FP，即提高 Precision。

阈值就是在两类错误之间做权衡，应由 FP/FN 的真实成本决定。

---

## 2. Precision、Recall、F1 与 Accuracy

$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

Precision：模型说是正样本的里面，有多少是真的。

Recall：所有真实正样本里，模型找到了多少。

$$
F_1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}
$$

F1 是 Precision 和 Recall 的调和平均数。只要其中一个很低，F1 就会低。它适合你想同时兼顾误报和漏报的场景。

$$
Accuracy=\frac{TP+TN}{TP+FP+TN+FN}
$$

### 类别不平衡时为什么不能只看 Accuracy？

若正样本只占 1%，模型全部预测为负也有 99% Accuracy，但正类 Recall 为 0。此时应根据目标关注 Precision、Recall、F1、PR 曲线和 PR-AUC，并同时报告类别比例与阈值。

“不能用 Accuracy”可更准确地说成：**不能把 Accuracy 作为唯一主指标**。如果业务成本对称且同时报告分组结果，Accuracy 仍可作为辅助指标。

如何在数据层面处理不平衡（重采样、类别权重、SMOTE 等），见 [03. 数据处理与实验方法](03-data-experiment.md)。

### Precision 与 Recall

- 垃圾邮件拦截：误杀正常邮件代价高，通常优先 Precision。
- 疾病初筛：漏诊代价高，通常优先 Recall，再通过后续检查过滤假阳性。

实际阈值应由 FP/FN 的业务成本、处理容量和风险约束共同决定，而不是机械地追求某个指标最大。

### F1 的局限

- 不考虑 TN。
- 默认 Precision 和 Recall 同等重要。
- 不评估概率校准。

若业务更重视 Recall 或 Precision，可使用 $F_\beta$：

$$
F_\beta
=(1+\beta^2)
\frac{PR}{\beta^2P+R}
$$

$\beta>1$ 更重视 Recall，$\beta<1$ 更重视 Precision。

### 何时用 F1 而非 Accuracy？

类别不平衡、或 FP/FN 代价不对称时，用 F1（或 $F_\beta$）比 Accuracy 更能反映模型对少数类的处理能力。

---

## 3. ROC-AUC 与 PR-AUC

### ROC-AUC 如何解释？

ROC-AUC 等于随机抽取一个正样本和一个负样本时，模型给正样本更高分的概率；若考虑并列分数，通常给 tie 计一半概率。

它衡量跨阈值的排序能力，与某个固定阈值无关。但在正类极少时，大量 TN 会让 ROC 曲线显得乐观，因此 PR-AUC 往往更有解释力。

ROC 曲线的两个坐标为：

$$
TPR=\frac{TP}{TP+FN}
$$

$$
FPR=\frac{FP}{FP+TN}
$$

改变分类阈值会得到不同的 $(\operatorname{FPR},\operatorname{TPR})$ 点。AUC 高不代表某个业务阈值下 Precision、Recall 或校准一定好。

### 追问：ROC-AUC 什么时候会误导？

正样本极少时，ROC-AUC 可能看起来不错，但真正上线的 Precision 很低。欺诈检测、疾病筛查、异常检测更应看 PR-AUC、Precision、Recall 和固定处理容量下的业务收益。

### PR-AUC

PR-AUC 是 Precision-Recall 曲线下面的面积。

- 横轴：Recall。
- 纵轴：Precision。
- 改变分类阈值，会得到不同的 Precision/Recall 点。

PR-AUC 越大，说明模型在不同阈值下都能保持较好的 Precision/Recall 平衡。它特别适合类别极度不平衡的任务，比如欺诈检测、疾病筛查、异常检测。

注意：PR-AUC 对正样本比例敏感。随机模型的 PR-AUC baseline 约等于正样本比例。

### F1 和 PR-AUC 的区别

- **F1**：某一个固定阈值下的 Precision 和 Recall 综合分数。
- **PR-AUC**：所有阈值下 Precision-Recall 表现的整体面积。

如果你已经确定业务阈值，可以看 F1、Precision、Recall。如果你还在比较模型整体排序能力，尤其是类别不平衡时，可以看 PR-AUC。

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

## 5. 回归指标：MSE

$$
\operatorname{MSE}
=\frac1N\sum_{i=1}^N
(y_i-\hat y_i)^2
$$

常用于连续值回归，对大误差惩罚较强。它对应预测条件均值；在同方差高斯噪声假设下也对应负对数似然。

常见同类指标：RMSE（量纲与目标一致）、MAE（抗 outlier）、$R^2$（解释方差比例）。选择取决于对异常值的敏感度和可解释性需求。MSE/MAE/Huber 的对比见 [01. 基础与神经网络机制](01-foundations.md)。

---

## 6. 如何判断两个模型谁更好？

### 基本流程

1. 先定义业务目标、主指标和不能退化的 guardrail 指标。
2. 使用独立测试集，确保两个模型在相同样本、预处理和阈值条件下比较。
3. 报告差值、置信区间和分组结果，不只比较单个均值。
4. 根据任务选 paired statistical test。
5. 最终用线上 A/B test 验证真实用户和系统影响。

### 追问：什么时候不用 Deep Learning？

数据少、特征是结构化表格、需要强可解释、延迟/成本很严格时，先用 Logistic Regression、Random Forest、XGBoost。深度模型不是默认答案，尤其在 tabular data 上，树模型经常更强也更稳。

### 常用方法

- **Paired bootstrap**：对同一测试样本成对重采样，估计指标差的置信区间，适合 Accuracy、F1、BLEU 等复杂指标。
- **Permutation test**：在零假设下交换两个模型的样本级结果，构造差值分布。
- **McNemar test**：比较两个分类器在同一批样本上的不一致错误。
- **交叉验证**：数据少时评估跨切分稳定性；模型选择和最终评估应使用 nested CV，避免调参泄漏。
- **A/B test**：需要预先定义样本量、显著性水平、最小可检测效应和停止规则。

### 注意

- Statistical significance 不等于 practical significance。
- 多指标、多切片反复检验要控制 multiple comparisons。
- 时间序列、用户行为等相关样本不能当作独立样本随意 bootstrap。
- 若模型用于概率决策，应同时比较 discrimination、calibration 和业务效用。
