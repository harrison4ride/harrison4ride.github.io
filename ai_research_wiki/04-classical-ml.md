# 01. 经典机器学习

[返回目录](README.md)

## 1. 线性回归

### 模型

$$
y=X\beta+\epsilon
$$

OLS 最小化残差平方和：

$$
\hat\beta
=\arg\min_\beta\|y-X\beta\|_2^2
$$

若 $X^\top X$ 可逆，闭式解为：

$$
\hat\beta=(X^\top X)^{-1}X^\top y
$$

实际计算通常使用 QR、SVD 或数值优化，而不是显式求逆。

### 基本假设

- 模型对参数是线性的，条件均值形式设定正确。
- 外生性：$\mathbb E[\epsilon\mid X]=0$。
- 没有完全多重共线性。
- 样本/误差相关结构符合使用场景；普通 OLS 推断常假设独立。
- 同方差使经典 OLS 在 Gauss-Markov 条件下成为 BLUE；异方差下系数仍可能无偏，但标准误和效率受影响。
- 残差正态性主要用于小样本下的精确置信区间和假设检验，不是求出 OLS 或保持无偏的必要条件。

### 如何解释回归系数？

在其他变量保持不变时，$x_j$ 增加一个单位，模型预测的条件均值改变 $\beta_j$。该解释依赖模型设定，不能自动解释为因果效应。

### 多重共线性

高度相关特征会导致：

- 系数方差大、符号和数值不稳定。
- 单个系数显著性难解释。
- 样本内预测可能仍好，但分布变化时不稳。

诊断可使用相关矩阵、VIF、condition number。处理包括领域驱动的特征选择、合并变量、PCA、Ridge 或收集更多有区分度的数据。Ridge 不会“消除相关性”，而是稳定估计。

### 最小二乘为什么对应最大似然？

若：

$$
\epsilon_i\overset{iid}{\sim}\mathcal N(0,\sigma^2)
$$

则负对数似然除去常数后为：

$$
-\log p(y\mid X,\beta)
=\frac{1}{2\sigma^2}
\sum_i(y_i-x_i^\top\beta)^2+\text{const}
$$

因此最大化高斯似然等价于最小化平方误差。若噪声分布不同，对应的 MLE 损失也会改变，例如 Laplace 噪声对应绝对误差。

### 非线性关系还能用线性回归吗？

“线性”指对参数线性，不要求对原始输入只能是一条直线。可以加入：

- 多项式或 spline basis。
- 分段线性项。
- 变量变换。
- 交互项 $x_1x_2$。

交互项表示一个变量的效应取决于另一个变量。例如：

$$
y=\beta_0+\beta_1x_1+\beta_2x_2
+\beta_3x_1x_2
$$

此时 $x_1$ 的边际效应为 $\beta_1+\beta_3x_2$。加入交互项后通常保留对应主效应，并注意标准化和可解释性。

---

## 2. MAE、MSE、KL Divergence 与 Cross-Entropy

#### MAE

L1 loss 也叫 Mean Absolute Error：

$$
L_1=\frac1n\sum_i |y_i-\hat y_i|
$$

特点：

- 对异常值没那么敏感。
- 误差从 10 变 100，惩罚也只是线性变大。
- 缺点是在误差为 0 的地方不光滑，优化时可能没 MSE 顺。

适合：数据里有 outlier，或者你不想让少数极端误差主导训练。

### MSE

$$
\operatorname{MSE}
=\frac1N\sum_{i=1}^N
(y_i-\hat y_i)^2
$$

常用于连续值回归，对大误差惩罚较强。它对应预测条件均值；在同方差高斯噪声假设下也对应负对数似然。

### KL Divergence

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

它衡量用 $Q$ 近似 $P$ 时的额外编码代价。KL 非负但不对称，不满足距离度量的全部性质。

### Cross-Entropy

$$
H(P,Q)=-\sum_xP(x)\log Q(x)
$$

关系为：

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q)
$$

当真实分布 $P$ 固定时，最小化 cross-entropy 等价于最小化 $D_{\mathrm{KL}}(P\|Q)$。Relative entropy 指 KL divergence，**不等于** cross-entropy。

---

## 3. Logistic Regression

### 模型与 Loss

二分类 logistic regression：

$$
z=w^\top x+b,\qquad
p(y=1\mid x)=\sigma(z)
$$

Bernoulli likelihood 为：

$$
p(y\mid x)
=p^y(1-p)^{1-y}
$$

对全体样本取负对数得到 binary cross-entropy：

$$
\mathcal L
=-\sum_i
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right]
$$

这就是从 Maximum Likelihood Estimation 推导出的 Log Loss。对线性 logit，BCE 是关于参数的凸函数；加 L1/L2 后仍为凸问题。

### 如何解释 Logistic Regression 系数？

对于二分类 logistic regression，系数 $w_j$ 表示 $x_j$ 增加 1 时，log-odds 增加 $w_j$。对应 odds ratio 是：

$$
\exp(w_j)
$$

但这个解释要求其他特征不变，并且特征尺度要清楚。标准化后的系数更方便比较特征影响大小。

### 为什么 Logistic Regression 不常用 MSE？

将 sigmoid 输出代入 MSE 后，目标关于 $w$ 一般不再凸，且高置信度错误区域可能梯度较弱。BCE 与 Bernoulli likelihood 匹配，梯度对 logit 可简化为 $p-y$，优化更自然。

### 多分类

Multinomial Logistic Regression 使用 softmax：

$$
p(y=k\mid x)
=\frac{e^{w_k^\top x}}
{\sum_j e^{w_j^\top x}}
$$

最大似然对应多分类 cross-entropy：

$$
\mathcal L=-\sum_i\log p(y_i\mid x_i)
$$

---

## 4. Logistic Regression 与 SVM

### 目标函数

Logistic Regression 使用 logistic loss，直接建模条件概率。线性 SVM 使用 hinge loss：

$$
\mathcal L_{\text{hinge}}
=\max(0,1-yf(x))
$$

软间隔 SVM 典型目标：

$$
\min_w
\frac12\|w\|_2^2
+C\sum_i\max(0,1-y_iw^\top x_i)
$$

它关注最大化 margin，而不是直接输出概率。

### 比较

| 维度       | Logistic Regression                | SVM                                     |
| ---------- | ---------------------------------- | --------------------------------------- |
| 输出       | 概率，仍需检查校准                 | decision score / margin                 |
| Loss       | Logistic / cross-entropy           | Hinge                                   |
| 关键样本   | 所有样本都有非零影响但远端影响变小 | 主要由 margin 内的 support vectors 决定 |
| 非线性     | 特征映射或核方法                   | Kernel trick 成熟                       |
| 大规模训练 | SGD/二阶方法方便                   | 线性 SVM 可扩展；核 SVM 随样本数变贵    |

不能简单说 SVM 天然比 Logistic Regression 抗异常值。严重错分点的 hinge loss 和 logistic loss 都会持续增长；鲁棒性取决于 $C$、正则化、特征尺度、异常值类型和是否使用鲁棒损失。

### Support Vectors 是什么？

Support vectors 是离决策边界最近、真正决定 margin 的训练样本。在线性 SVM 中，如果删掉远离边界且分类正确的样本，最优超平面通常不变；但删掉 support vector 可能改变边界。

### Kernel Trick 是什么？

Kernel trick 用核函数直接计算高维特征空间中的内积，而不用显式构造高维特征：

$$
K(x_i,x_j)=\phi(x_i)^\top\phi(x_j)
$$

常见核：

- Linear kernel：适合线性可分或高维稀疏特征。
- Polynomial kernel：建模特征交互。
- RBF kernel：常用非线性核，但大数据上计算贵，也需要调 $\gamma$。

面试一句话：

> Kernel trick 让 SVM 在隐式高维空间里找线性边界，从原空间看就是非线性边界。

### 追问：SVM 里的 $C$ 变大会怎样？

$C$ 是错分惩罚。

- $C$ 大：更不允许错分，margin 可能变窄，容易过拟合。
- $C$ 小：允许更多错分，margin 更宽，模型更简单。

RBF kernel 还要调 $\gamma$：$\gamma$ 大时单个样本影响范围小，边界更弯；$\gamma$ 小时边界更平滑。

### 追问：为什么 SVM 需要标准化？

SVM 依赖 margin 和内积。如果一个特征尺度特别大，比如收入是几万、年龄是几十，边界会被大尺度特征主导。标准化后，各特征才有可比较的影响。

---

## 5. Decision Tree

### 分类树如何选择分裂？

寻找使子节点更纯的特征和阈值。常见 impurity：

$$
\operatorname{Gini}(S)
=1-\sum_kp_k^2
$$

$$
H(S)=-\sum_kp_k\log p_k
$$

Information Gain：

$$
\operatorname{Gain}
=I(S)-\sum_{c}
\frac{|S_c|}{|S|}I(S_c)
$$

选择 impurity decrease 最大的分裂。

### 追问：决策树为什么容易过拟合？

树可以不断切分训练集，把噪声也记住。尤其 `max_depth` 很大、叶子样本很少时，训练误差会很低但泛化差。

### 回归树如何选择分裂？

常以子节点内平方误差最小为目标，每个叶节点预测该叶训练标签均值：

$$
\sum_{c\in\{\text{left,right}\}}
\sum_{i\in S_c}(y_i-\bar y_c)^2
$$

### 如何防止过拟合？

- 限制 `max_depth`、`max_leaf_nodes`。
- 设置 `min_samples_split`、`min_samples_leaf`。
- 要求最小 impurity decrease。
- Cost-complexity post-pruning。
- 通过交叉验证调参。
- 使用 Random Forest、Gradient Boosting 等 ensemble 降低单树方差。

---

## 6. K-Means

### 目标

$$
\min_{\{C_k,\mu_k\}}
\sum_{k=1}^{K}
\sum_{x_i\in C_k}
\|x_i-\mu_k\|_2^2
$$

### Lloyd 算法

1. 初始化 $K$ 个 centroid，常用 k-means++。
2. Assignment：把每个点分给最近 centroid。
3. Update：每个 centroid 更新为所属点的均值。
4. 重复直到停止。

每一步都不会增加目标函数；可能的离散分配有限，因此算法最终收敛。但通常只保证收敛到局部最优或稳定点，不保证全局最优。实践中使用多个随机初始化，选择 inertia 最小的结果。

### 停止条件

- 分配不再变化。
- centroid 移动小于阈值。
- 目标改善小于阈值。
- 达到最大迭代数。

### 如何选 K？

- Elbow method。
- Silhouette score。
- Gap statistic。
- 下游任务表现与领域可解释性。

这些方法都不是绝对标准。K-Means 假设欧氏距离有意义，偏好近似球形、尺度相近的簇，因此通常需要特征标准化，也不适合任意形状或大量异常点。

### 追问：Elbow Method 不明显怎么办？

说明数据可能没有清晰的簇数，或 KMeans 假设不合适。可以结合 silhouette score、gap statistic、业务解释性和下游任务指标，不要硬找一个“肘部”。

### 追问：KMeans 对非球形簇怎么办？

KMeans 偏好球形、尺度相近的簇。遇到长条形、环形、不同密度簇时，可以考虑：

- GMM：允许椭圆形簇和 soft assignment。
- DBSCAN：适合密度簇和噪声点。
- Spectral clustering：适合复杂形状，但计算更贵。
- 先做 embedding、kernel 或降维再聚类。

---

## 7. EM 算法

EM 用于含 latent variable 或缺失变量的概率模型。设观测为 $X$、隐变量为 $Z$、参数为 $\theta$。

### E-Step

用旧参数计算隐变量后验：

$$
q(Z)=p(Z\mid X,\theta^{\text{old}})
$$

并构造 complete-data log-likelihood 的期望：

$$
Q(\theta,\theta^{\text{old}})
=\mathbb E_{Z\sim q}
[\log p(X,Z\mid\theta)]
$$

### M-Step

$$
\theta^{\text{new}}
=\arg\max_\theta
Q(\theta,\theta^{\text{old}})
$$

EM 可从 Jensen inequality / evidence lower bound 推导。标准条件下每轮不会降低观测数据 log-likelihood，但只保证收敛到 stationary point，可能是局部最优或鞍点。初始化、局部最优和退化解都需要处理。

常见应用包括 GMM、隐变量模型、缺失数据估计和 HMM 的 Baum-Welch。

---

## 8. GMM 与 K-Means

### GMM

Gaussian Mixture Model 的密度为：

$$
p(x)
=\sum_{k=1}^{K}
\pi_k\mathcal N(x\mid\mu_k,\Sigma_k),
\qquad \sum_k\pi_k=1
$$

参数包括 mixture weights、均值和协方差，通常使用 EM 估计。E-step 计算 responsibility：

$$
\gamma_{ik}
=p(z_i=k\mid x_i)
$$

因此 GMM 是 soft clustering，还能用于密度估计、异常检测和生成。

### 与 K-Means 的区别

| 维度   | K-Means                | GMM                          |
| ------ | ---------------------- | ---------------------------- |
| 分配   | Hard assignment        | Soft probability             |
| 目标   | 最小化簇内平方距离     | 最大化 mixture likelihood    |
| 簇形状 | 偏好球形、相似尺度     | 协方差可描述椭圆形和不同尺度 |
| 输出   | centroid 与 cluster ID | 完整概率模型                 |
| 优化   | Lloyd algorithm        | EM                           |

K-Means 可视为在各 Gaussian 使用相同、各向同性且方差趋近 0 时，对 GMM 进行 hard assignment 的极限情形。

GMM 也有局限：需要选择 $K$ 和 covariance type；可能发生某个 covariance 收缩到单点导致 likelihood 退化，通常要加 covariance regularization。

---

## 9. Random Forest

Random Forest 是很多棵决策树的 bagging ensemble：

1. 对训练样本做 bootstrap sampling。
2. 每棵树分裂时只看随机子集特征。
3. 分类用多数投票，回归用平均。

为什么有效：

- 单棵树方差高，容易过拟合。
- 多棵低相关树平均后，方差下降。
- 随机特征子集让树之间更不一样。

常见超参数：

- `n_estimators`：树数量。
- `max_depth`、`min_samples_leaf`：控制单树复杂度。
- `max_features`：控制每次分裂看多少特征。

优点是鲁棒、少调参、能处理非线性；缺点是模型大、推理慢、外推能力弱，可解释性不如单棵树。

### 追问：Random Forest 加更多树一定更好吗？

树更多通常会降低方差，但收益会饱和，也会增加训练和推理成本。它不能解决单棵树 bias 太高的问题；如果每棵树都太浅，森林也可能欠拟合。

---

## 10. Naive Bayes

Naive Bayes 基于 Bayes rule：

$$
p(y\mid x)
\propto p(y)\prod_j p(x_j\mid y)
$$

“Naive” 指假设特征在给定类别后条件独立。这个假设通常不完全成立，但在文本分类中常常够用。

### 为什么适合文本？

Bag-of-words 特征高维稀疏，Naive Bayes 训练快、需要数据少。垃圾邮件分类、情感分类、新闻分类都可以作为 baseline。

常见版本：

- Gaussian NB：连续特征，假设类条件高斯分布。
- Multinomial NB：词频计数。
- Bernoulli NB：词是否出现。

注意要做 smoothing，否则没见过的词会让概率变成 0。

### 追问：Naive Bayes 假设明显不成立，为什么还能用？

它的概率校准可能不准，但分类边界有时仍然不错。文本分类中词特征高维稀疏，Naive Bayes 训练快、抗小数据，是很强 baseline。

---

## 11. KNN

KNN 不显式训练参数。预测时找离测试点最近的 $K$ 个训练样本：

- 分类：多数投票。
- 回归：均值或距离加权均值。

关键点：

- 距离度量和 feature scaling 很重要。
- $K$ 太小容易过拟合，$K$ 太大容易欠拟合。
- 维度高时距离变得不可靠，推理也慢。

KNN 适合作为简单 baseline，但大规模线上服务通常要用近似最近邻索引或换模型。

### 追问：为什么 KNN 需要标准化？

KNN 直接用距离找邻居。如果某个特征尺度特别大，距离几乎会被它主导，其他特征基本失效。标准化后，距离才更接近“多特征共同相似”。

---

## 12. Ensemble、Bagging、Boosting 与 XGBoost

### Ensemble Learning

Ensemble 是把多个模型组合起来，目标是比单模型更稳定或更准确。

- **Bagging**：并行训练多个模型，主要降方差。代表：Random Forest。
- **Boosting**：串行训练，每一步修正前面模型的错误，主要降 bias，也可能降 variance。代表：AdaBoost、GBDT、XGBoost。
- **Stacking**：把多个模型输出作为新特征，再训练 meta model。

### 追问：Bagging 和 Boosting 一句话区分？

> Bagging 是并行训练多个高方差模型再平均，主要降方差；Boosting 是串行训练，每一步修正前一步错误，主要降 bias，但更容易过拟合，需要控制学习率和树复杂度。

### XGBoost

XGBoost 是高效的梯度提升树实现。它逐棵加树，每棵树拟合当前目标的一阶/二阶梯度信息。

常见优势：

- 二阶优化，收敛快。
- 支持 L1/L2 正则和 shrinkage。
- 列采样、行采样降低过拟合。
- 缺失值默认方向、稀疏特征优化。
- 工程实现高效，适合结构化表格数据。

面试一句话：

> Random Forest 是多棵树并行投票，主要降低方差；XGBoost 是一棵棵树串行纠错，通常在表格数据上很强，但更需要调参防过拟合。

### 追问：XGBoost 怎么防过拟合？

- 降低 `max_depth`。
- 增大 `min_child_weight`。
- 降低 `learning_rate` 并配合更多树。
- 使用 `subsample`、`colsample_bytree`。
- 加 L1/L2 正则。
- early stopping。

表格数据上 XGBoost 强，但不是免调参模型。

---

## 13. Hierarchical Clustering

Hierarchical clustering 生成一棵层次聚类树，不需要一开始固定所有样本的硬分配。

常见做法是 agglomerative：

1. 每个点先是一个簇。
2. 每次合并距离最近的两个簇。
3. 直到剩下一个簇或达到停止条件。

簇间距离 linkage：

- Single linkage：两个簇最近点距离，容易链式连接。
- Complete linkage：两个簇最远点距离，簇更紧。
- Average linkage：平均距离。
- Ward linkage：最小化合并后的方差增加。

与 K-Means 比：

- K-Means 需要指定 $K$，适合球形簇和大数据。
- Hierarchical 可看不同粒度层次，但计算更贵，对噪声敏感。

### 追问：linkage 怎么选？

- Single linkage 容易把点串起来，适合发现链状结构但怕噪声。
- Complete linkage 产生更紧的簇。
- Average linkage 折中。
- Ward linkage 常用于数值特征，偏向紧凑球形簇。

---

## 14. Association Rules

Association rules 用于发现“买了 A 的人也常买 B”这类共现模式。

规则形式：

$$
A\Rightarrow B
$$

常见指标：

- **Support**：$A$ 和 $B$ 同时出现的比例。
- **Confidence**：出现 $A$ 时也出现 $B$ 的概率。
- **Lift**：比随机独立情况下强多少。

$$
\operatorname{lift}(A\Rightarrow B)
=\frac{P(A,B)}{P(A)P(B)}
$$

Lift 大于 1 表示正相关。常见算法包括 Apriori 和 FP-Growth。

### 追问：Association Rule 里 confidence 高就一定好吗？

不一定。比如很多人都会买牛奶，那么“买面包 $\Rightarrow$ 买牛奶”的 confidence 可能高，但不代表面包真的增加买牛奶的概率。要看 lift：

- lift $>1$：正相关。
- lift $\approx1$：基本独立。
- lift $<1$：负相关。

[返回目录](README.md)