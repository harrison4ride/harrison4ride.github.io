# 03. 经典机器学习

[返回目录](README.md)

| 任务       | 目标                     | 常见模型                                                  |
| ---------- | ------------------------ | --------------------------------------------------------- |
| 回归       | 预测连续值               | Linear Regression、Regression Tree                        |
| 分类       | 预测离散类别             | Logistic Regression、SVM、Decision Tree、Naive Bayes、KNN |
| 无监督学习 | 在没有标签时发现数据结构 | K-Means、GMM、Hierarchical Clustering、Association Rules  |

这些模型并不是互不相关：

- Linear Regression、Logistic Regression 和 Linear SVM 都从 $w^\top x+b$ 出发，但使用不同的输出与 Loss。
- Decision Tree 是基础模型；Random Forest 用 Bagging 降低单树的 Variance，XGBoost 用 Boosting 逐步修正错误。
- K-Means 做 Hard Clustering；GMM 做概率化的 Soft Clustering；EM 是训练 GMM 的通用算法。
- KNN、SVM 和 K-Means 都依赖距离或内积，因此对 Feature Scaling 很敏感。
- MAE、MSE 和 Cross-Entropy 不只是评估指标，也分别对应不同的概率假设和训练目标。

## 1. 线性回归

Linear Regression 假设目标的条件均值可以表示为输入特征的线性组合，并使用 OLS 最小化残差平方和。在高斯噪声假设下，最小化 MSE 等价于 Maximum Likelihood。这里的“线性”是对参数线性，所以仍然可以加入多项式、Spline 和交互项。

### 模型

$$
y=X\beta+\epsilon
$$

拆开看：$X$ 是 $n\times p$ 的设计矩阵（$n$ 个样本、$p$ 个特征，通常第一列全填 1 用来吸收截距），$\beta$ 是待估的系数向量，$\epsilon$ 是没被特征解释掉的随机噪声。整个式子在说一件事：**真实值 = 特征的线性组合 + 噪声**。

OLS 最小化残差平方和：

$$
\hat\beta
=\arg\min_\beta\|y-X\beta\|_2^2
$$

$y-X\beta$ 是残差向量（真实值减预测值），$\|\cdot\|_2^2$ 把每个残差平方后加起来。所以这一行读作：挑一组 $\beta$，让所有点到这条线的竖直距离平方和最小。平方有两个作用——正负误差不再相互抵消，以及大误差被加倍惩罚。

若 $X^\top X$ 可逆，闭式解为：

$$
\hat\beta=(X^\top X)^{-1}X^\top y
$$

它的来历是把平方误差对 $\beta$ 求导、令导数为 0，得到 normal equation $X^\top X\beta=X^\top y$，两边左乘逆矩阵。几何上它做的事是**把 $y$ 投影到 $X$ 的列空间**：预测值落在特征张成的那个平面里，残差与平面垂直——垂直才意味着「特征里再也榨不出信息了」。

反过来，$X^\top X$ 不可逆就意味着某一列特征能被其他列线性表示，投影的落点不唯一，解也就不唯一，这正是下面多重共线性的极端情形。

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

推导只有三步：单个样本的高斯密度是 $\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y_i-x_i^\top\beta)^2}{2\sigma^2}\right)$；取对数后 exp 被剥掉、样本间的连乘变成连加；前面那个与 $\beta$ 无关的系数被丢进 const。剩下的正好是残差平方和除以 $2\sigma^2$——而 $\frac{1}{2\sigma^2}$ 是个正常数，缩放不改变最优解的位置。

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

Loss 决定模型如何看待错误。不同 Loss 不只是惩罚方式不同，还对应不同的预测目标和概率假设：

| 数据与任务                           | 常用 Loss            | 对应解释                 |
| ------------------------------------ | -------------------- | ------------------------ |
| 连续值，误差近似 Gaussian            | MSE                  | 学习条件均值             |
| 连续值，异常值较多或误差近似 Laplace | MAE                  | 学习条件中位数           |
| 二分类，标签服从 Bernoulli           | Binary Cross-Entropy | 最大化二分类 Likelihood  |
| 多分类，标签服从 Categorical         | Cross-Entropy        | 最大化真实类别的预测概率 |
| 让模型分布接近目标分布               | KL Divergence        | 衡量两个概率分布的差异   |

### MAE

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

每个误差先平方再平均。平方的后果很直接：误差从 10 变成 100，惩罚从 100 变成 10000——翻了 100 倍。所以一个离群点能顶一百个普通点，训练容易被它带着跑。好处是处处可导、优化顺滑，这正是 MAE 缺的那一块。

常用于连续值回归，对大误差惩罚较强。它对应预测条件均值——因为在一组数里，让平方和最小的那个常数就是均值（同理，让绝对值之和最小的是中位数，所以 MAE 学的是条件中位数）。在同方差高斯噪声假设下，它也对应负对数似然。

### KL Divergence

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

一句话：它衡量用 $Q$ 近似 $P$ 时，平均每条消息多花的编码代价。下面把这句话拆开讲——用「发电报 / 压缩文件」的比喻最容易懂。

#### 先看码长：$\log_2\frac{1}{P(x)}$ 是怎么来的

信息传输的最小单位是 bit（0 和 1）：

- 1 个 bit 能区分 $2^1=2$ 种情况；
- 2 个 bit 能区分 $2^2=4$ 种（00、01、10、11）；
- $L$ 个 bit 能区分 $2^L$ 种。

反过来问：要唯一标识 $N$ 种等概率的情况，需要几个 bit？解 $2^L=N$ 得 $L=\log_2N$。

而一个概率为 $P(x)$ 的事件，相当于从 $N=\frac{1}{P(x)}$ 个等可能选项里抽中了它。所以它的最优码长是：

$$
L(x)=\log_2\frac{1}{P(x)}=-\log_2P(x)
$$

直观上就是**常发生的事用短码，罕见的事用长码**。太阳照常升起，大家心领神会，1 个 bit 就够；中头彩这种事，得用一长串才能说清楚。

底数 2 只是因为我们用二进制。换三进制就是 $\log_3$，单位从 bit 变成 trit；推导里用自然底数 $e$ 就写成 $\ln$，单位叫 nat。深度学习默认用 $\ln$（求导方便），不影响任何结论。

#### 一个例子

四种天气和它们的真实概率：

| 天气 | $P(x)$ | 最优码长 $\log_2\frac{1}{P(x)}$ | 编码  |
| ---- | ------ | ------------------------------- | ----- |
| 晴   | 1/2    | 1 bit                           | `0`   |
| 阴   | 1/4    | 2 bits                          | `10`  |
| 雨   | 1/8    | 3 bits                          | `110` |
| 雪   | 1/8    | 3 bits                          | `111` |

这正是 Huffman coding 在做的事：按概率分配码长，概率越大码越短。

#### 三个「熵」串起来

把这些码长按真实概率 $P$ 加权平均，就是**自熵**——用真实分布设计编码时，平均每条消息要花的 bit 数：

$$
H(P)=\sum_xP(x)\log_2\frac{1}{P(x)}=-\sum_xP(x)\log_2P(x)
$$

负号的来历很朴素：$P(x)\le1$ 所以 $\log P(x)$ 本身是负数，加个负号才让码长是正的。

现在换个设定：真实世界还是 $P$，但你手上只有一个估计 $Q$，编码表是照着 $Q$ 设计的。比如现实中晴天占 80%，你却以为这城市天天暴雨、晴天只有 1%，于是给晴天分了一个 10 bit 的长码。

这时你实际花掉的平均码长是**交叉熵**——用 $Q$ 的码长，去编 $P$ 的事件：

$$
H(P,Q)=\sum_xP(x)\log_2\frac{1}{Q(x)}=-\sum_xP(x)\log_2Q(x)
$$

对某一个事件 $x$，你多花了多少？拿实际用的码长减去本可以达到的最短码长：

$$
\underbrace{\log_2\frac{1}{Q(x)}}_{\text{实际用的}}
-\underbrace{\log_2\frac{1}{P(x)}}_{\text{本可以用的}}
=\log_2\frac{P(x)}{Q(x)}
$$

再按事件真正发生的频率 $P(x)$ 加权平均，把总账算出来，就是 KL：

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_x
\underbrace{P(x)}_{\text{真实发生频率}}
\cdot
\underbrace{\log_2\frac{P(x)}{Q(x)}}_{\text{这次多花的 bit}}
=H(P,Q)-H(P)
$$

所以三者的关系就是一笔大白话账：

- $H(P)$：最省能省到多少；
- $H(P,Q)$：你按错误认知实际花了多少；
- $D_{\mathrm{KL}}(P\|Q)$：差额，被浪费掉的部分。

估计完全准确时 $Q=P$，每项 $\log\frac{P(x)}{Q(x)}=\log1=0$，一个 bit 都不浪费，KL 为 0。估计越离谱浪费越多，所以 KL 恒非负（Gibbs 不等式）。

#### 但它不是距离

KL 非负、且为 0 当且仅当两分布处处相等，这两点像距离。但它不对称：$D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P)$，也不满足三角不等式，所以不是 metric。

从编码角度看，不对称很自然：「真实是 $P$，我按 $Q$ 编码」和「真实是 $Q$，我按 $P$ 编码」本来就是两件事，浪费的账当然不一样。

实践上的区别：

- Forward KL $D_{\mathrm{KL}}(P\|Q)$：在 $P(x)>0$ 而 $Q(x)\to0$ 的地方 log 会爆炸，逼着 $Q$ 覆盖 $P$ 的所有区域（mode-covering，结果偏糊）。
- Reverse KL $D_{\mathrm{KL}}(Q\|P)$：只在 $Q$ 有概率质量的地方算账，$Q$ 可以只贴住 $P$ 的一个峰（mode-seeking，结果偏尖）。

写 loss 时把哪个分布放前面，训出来的东西不一样。

### Cross-Entropy

$$
H(P,Q)=-\sum_xP(x)\log Q(x)
$$

关系为：

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q)
$$

也就是上面那笔账：**实际花的 = 理论最省的 + 浪费的**。

当真实分布 $P$ 固定时，$H(P)$ 是常数，最小化 cross-entropy 等价于最小化 $D_{\mathrm{KL}}(P\|Q)$。这就是监督学习里只写 cross-entropy、不写 KL 的原因：标签分布是死的，两者只差一个常数，梯度完全一样。Relative entropy 指 KL divergence，**不等于** cross-entropy。

### 为什么 RL 里到处是 KL？

监督学习几乎只用 cross-entropy，但到了 RL（TRPO、PPO）和大模型对齐（RLHF、DPO）里满屏都是 KL。三个原因。

**① 参照分布是动的，常数项没了**

上面那句「等价」有个前提：$P$ 固定，$H(P)$ 是常数。RL 里参照的分布是上一轮的旧策略 $\pi_{\theta_{\text{old}}}$（或 reference model），它每次迭代都在变，$H(\pi_{\theta_{\text{old}}})$ 不再是常数。这时 cross-entropy 的数值里混了一项一直在动的熵，没法用来衡量「新旧策略到底差了多少」。要量的是相对变化，就必须显式写 KL。

**② 给策略更新加一根安全绳**

策略 policy 本身就是一个由参数控制的概率分布 $\pi_\theta(a\mid s)$。梯度更新的步子只要稍微大一点，新策略 $\pi_{\theta_{\text{new}}}$ 就可能和旧策略差出十万八千里。

麻烦在于 RL 的训练数据是策略自己跑出来的：策略一变差，采集到的交互数据质量跟着暴跌，模型陷进去就再也爬不回来（policy collapse）。监督学习没这个问题，数据集是死的，练坏了重来就行。

做法是把 $D_{\mathrm{KL}}(\pi_{\theta_{\text{old}}}\|\pi_\theta)$ 加进 loss 当惩罚项，或者直接当成硬约束限制在一个信赖域内（TRPO）。PPO 的 clipping 是同一思路的省事版本：不显式算 KL，直接把新旧概率比 clip 住。作用都是一根安全绳——可以往前走，但不许一步迈太远。

**③ 防止 RLHF 里的 Reward Hacking**

用人类反馈训练 LLM 时，模型会按 reward model 的打分调整输出策略。如果不加限制，它很快会学会钻分数的漏洞：输出一些人类看着莫名其妙、但 RM 打分特别高的东西（reward hacking），顺手把原本的语言能力丢掉。

办法是在目标里加上与基座模型（reference model $\pi_{\mathrm{ref}}$）的 KL 惩罚：

$$
\mathcal L
=-\mathbb E[R(s,a)]
+\beta\,D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

$\beta$ 控制松紧：太大模型不敢动、学不到东西；太小就放飞自我。这一项保证模型在提高人类偏好得分的同时，输出分布不会脱离预训练模型的基本语法和表达范式。DPO 则把这个 KL 约束解析地解掉，形式上没有显式 RL loop，但安全绳还在。细节见 [08. 对齐与 RLHF](08-alignment-and-rlhf.md)。

一句话收尾：cross-entropy 用来「逼近一个固定目标」，KL 用来「限制离参照分布有多远」。RL 要的是后者。

---

## 3. Logistic Regression

Logistic Regression 与 Linear Regression 都先计算 $w^\top x+b$。区别是 Linear Regression 直接输出连续值，Logistic Regression 使用 Sigmoid 将线性分数转换为正类概率。

### 模型与 Loss

二分类 logistic regression：

$$
z=w^\top x+b,\qquad
p(y=1\mid x)=\sigma(z)
$$

$z$ 叫 logit，是一个可正可负、范围无限的分数，本身不能当概率用。Sigmoid 负责把它压进 $(0,1)$：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

$z=0$ 时 $\sigma=0.5$（模型完全没主意），$z\to+\infty$ 时趋近 1，$z\to-\infty$ 时趋近 0。所以 $w^\top x+b$ 决定「往哪边偏、偏多少」，sigmoid 只是把这个偏好翻译成概率。

Bernoulli likelihood 为：

$$
p(y\mid x)
=p^y(1-p)^{1-y}
$$

这是一个写成一行的开关：$y=1$ 时式子塌缩成 $p$，$y=0$ 时塌缩成 $1-p$。指数在这里不是真的在做幂运算，而是用 0/1 把两种情况选出来，好处是后面可以直接取对数、连乘变连加。

对全体样本取负对数得到 binary cross-entropy：

$$
\mathcal L
=-\sum_i
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right]
$$

同样地，每个样本只有一项在起作用：标签是 1 就罚 $-\log p_i$，标签是 0 就罚 $-\log(1-p_i)$。预测越接近正确答案，$\log$ 越接近 0，惩罚越小；而如果你用 99% 的把握把答案说反，$-\log(0.01)\approx4.6$，惩罚会非常重。**自信地犯错代价极高**，这是 log loss 的核心性格。

这就是从 Maximum Likelihood Estimation 推导出的 Log Loss。对线性 logit，BCE 是关于参数的凸函数；加 L1/L2 后仍为凸问题。

### 如何解释 Logistic Regression 系数？

先说 odds（发生比）：$\text{odds}=\frac{p}{1-p}$，即「发生的可能性是不发生的几倍」。$p=0.8$ 时 odds $=4$，读作发生的可能性是不发生的 4 倍。取对数就是 log-odds，也正好等于模型里的 $z=w^\top x+b$。

对于二分类 logistic regression，系数 $w_j$ 表示 $x_j$ 增加 1 时，log-odds 增加 $w_j$。log 上的加法等于原尺度上的乘法，所以对应的 odds ratio 是：

$$
\exp(w_j)
$$

举例：$\exp(w_j)=1.5$ 读作「$x_j$ 每增加一个单位，发生比变成原来的 1.5 倍」。注意变的是 odds 而不是概率——概率的增量取决于起点在哪（从 0.5 涨和从 0.95 涨完全不是一回事）。

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

每个类别先各自算一个分数 $w_k^\top x$（可正可负），然后：分子取指数保证结果为正，分母是所有类别指数分数之和，保证全部加起来等于 1。所以 softmax 做的事就是**把一组任意实数分数，压成一个合法的概率分布**，且分数越高概率越大。指数还会放大差距——分数差 1 分和差 3 分，概率上的差距不是线性的。二分类时它退化成 sigmoid。

最大似然对应多分类 cross-entropy：

$$
\mathcal L=-\sum_i\log p(y_i\mid x_i)
$$

式子里只出现了真实类别那一项：其他类别的概率不直接进 loss。但因为 softmax 的分母把所有类别绑在了一起，压低别的类同样会抬高真实类的概率，所以「打压错误答案」这件事是通过分母自动发生的。

---

## 4. Logistic Regression 与 SVM

两者都是常见的线性分类器，但优化目标不同：Logistic Regression 学习分类概率，SVM 寻找 Margin 较宽的分类边界。

### 目标函数

Logistic Regression 使用 logistic loss，直接建模条件概率。线性 SVM 使用 hinge loss：

$$
\mathcal L_{\text{hinge}}
=\max(0,1-yf(x))
$$

其中 $y\in\{-1,+1\}$ 是真实标签（注意不是 0/1），$f(x)=w^\top x+b$ 是模型输出的原始得分，没经过任何压缩，正数判正类、负数判负类。

**为什么叫 hinge（铰链）？** 因为函数图像长得像一扇门的合页：在 $yf(x)\ge1$ 的一侧完全平贴在 0 上，越过 1 之后折起来，呈线性上升。

**$yf(x)$ 代表什么？** 它是标签和得分的乘积：

- 符号：$yf(x)>0$ 说明预测符号和真实标签一致，分类正确；$yf(x)<0$ 说明反了，分类错误。
- 大小：代表样本离决策边界 $f(x)=0$ 有多远，也就是模型有多大底气。

**三种情况分开读：**

- $yf(x)\ge1$（分对了，且离边界足够远）：$\max(0,1-\text{大于等于 1})=0$，损失为 0。SVM 认为这个样本已经足够安全，**完全不给惩罚，也不再参与调整模型**。
- $0<yf(x)<1$（分对了，但离边界太近）：比如 $yf(x)=0.6$，损失是 $1-0.6=0.4$。虽然分对了，却落在 margin（安全间隔带）内部，SVM 认为不够安全，照罚。
- $yf(x)<0$（分错了）：比如 $yf(x)=-1.5$，损失是 $1-(-1.5)=2.5$。错得越离谱惩罚越重，随错分程度线性增长。

关键就是那个 1：SVM 不满足于「分对」，还要求分对之后留出一条安全带。对比 logistic loss $\log(1+e^{-yf(x)})$——它永远大于 0，哪怕分得再对也还会渗出一点梯度，所以每个样本都在轻微地推边界；hinge 直接把远处样本的梯度削成 0，这就是为什么最终边界只由少数几个样本（support vectors）决定。

### 软间隔 SVM 的完整目标函数

$$
\min_w
\underbrace{\frac12\|w\|_2^2}_{\text{正则化 / 扩大 margin}}
+\underbrace{C\sum_i\max(0,1-y_iw^\top x_i)}_{\text{经验风险 / hinge 惩罚}}
$$

软间隔（soft margin）允许训练集中存在少量噪声或交错点。这个目标函数本质上是一场权衡（trade-off）：

**① 第一项 $\frac12\|w\|_2^2$——追求更宽的间隔**

几何上，两个平行支撑超平面之间的物理宽度是：

$$
\text{Margin}=\frac{2}{\|w\|}
$$

要让 margin 尽可能宽，等价于让 $\|w\|$ 尽可能小，也就是最小化 $\frac12\|w\|_2^2$（$\frac12$ 和平方纯粹是为了求导好看，不影响最优解）。这一项同时也就是 L2 正则化：它既防止权重过大导致过拟合，又赋予了 SVM「寻找最大几何间隔」这一独特性质。

**② 第二项 $C\sum_i\mathcal L_{\text{hinge}}$——追求更少的分类错误**

把所有样本的 hinge loss 累加，衡量当前模型在训练集上的总错误代价。只有落在 margin 内部或分错的样本（即 support vectors 和错分点）才会贡献非零损失，其余样本这一项直接是 0。

**③ 超参数 $C$——严格程度的调节杠杆**

- $C$ 很大：几乎不允许任何错分，逼模型去找一个极窄但「纯净」的 margin，容易过拟合。
- $C$ 较小：允许更多样本掉进 margin 内部甚至被误分，换来更宽、更平滑的决策边界，泛化更好。

所以 SVM 关注的是最大化 margin，而不是直接输出概率。

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

$\phi$ 是把样本映到高维空间的映射，$\phi(x_i)^\top\phi(x_j)$ 是两个样本在那个高维空间里的内积（可以理解成「有多像」）。

trick 在于：SVM 的训练和预测**只用得到样本两两之间的内积**，从头到尾不需要单独的 $\phi(x)$。既然如此，就干脆跳过映射，直接用一个函数把内积算出来。比如 RBF kernel $K(x_i,x_j)=\exp(-\gamma\|x_i-x_j\|^2)$ 对应的 $\phi$ 是无穷维的，显式写出来根本不可能，但这个内积一行就算完了。

常见核：

- Linear kernel：适合线性可分或高维稀疏特征。
- Polynomial kernel：建模特征交互。
- RBF kernel：常用非线性核，但大数据上计算贵，也需要调 $\gamma$。

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

线性模型使用一个全局边界；Decision Tree 则反复切分特征空间，在不同区域使用不同规则，因此可以自然表达非线性和特征交互。

### 分类树如何选择分裂？

核心思想是**贪心**：在当前节点，遍历所有可能的特征和切割阈值，找那个能让切分后子节点的混乱度（impurity，不纯度）下降最多的组合。

#### 不纯度指标 $I(S)$

不纯度衡量一个集合 $S$ 里类别的混乱程度。设 $S$ 中共有 $K$ 个类别，$p_k$ 是第 $k$ 类样本在 $S$ 中的占比（$\sum_{k=1}^Kp_k=1$）。

**① 基尼指数（Gini Impurity）**

$$
\operatorname{Gini}(S)
=1-\sum_{k=1}^Kp_k^2
$$

直观解释：从 $S$ 里随机抽两个样本，**它们类别不一致的概率**。（$\sum_kp_k^2$ 是抽到同类的概率，用 1 减掉就是不同类。）

- 完全纯净（全是同一类，$p_1=1,p_2=0$）：$\operatorname{Gini}=1-(1^2+0^2)=0$，混乱度最低。
- 最为混乱（二分类各占一半，$p_1=p_2=0.5$）：$\operatorname{Gini}=1-(0.5^2+0.5^2)=0.5$，混乱度最高。

特点：只有乘法和加法，不用求对数，算得非常快，是 CART 决策树的默认指标。

**② 熵（Entropy）**

$$
H(S)=-\sum_{k=1}^Kp_k\log_2p_k
$$

直观解释：这个集合的不确定性有多大；换成第 2 节的说法，就是把这里的类别分布编码出来平均要花多少 bit。

- 完全纯净：$H(S)=0$，没有任何不确定性，一个 bit 都不用花。
- 最为混乱（二分类）：$H(S)=-(0.5\log_20.5+0.5\log_20.5)=1$ bit。

特点：出自信息论，对概率变化比 Gini 更敏感，但要算对数，开销略大。是 ID3 / C4.5 的基础。两者曲线形状很接近，实际选哪个对结果影响不大。

#### 信息增益（Information Gain）

$$
\operatorname{Gain}
=I(S)-\sum_{c}
\frac{|S_c|}{|S|}I(S_c)
$$

这个式子算的是：按某个特征切一刀之后，不纯度总共下降了多少。降得越多，说明这刀切得越有效。拆成三块看：

**① 切分前的不纯度 $I(S)$**

当前节点还没切之前的混乱度，用 $\operatorname{Gini}(S)$ 或 $H(S)$ 都行。

**② 切分后的加权平均不纯度 $\sum_c\frac{|S_c|}{|S|}I(S_c)$**

把 $S$ 拆成若干子节点 $S_c$（比如左右两个）之后：$I(S_c)$ 是子节点自己的不纯度，$\frac{|S_c|}{|S|}$ 是落进这个子节点的样本占比。

必须**按样本数加权**，不能直接取算术平均——样本多的子节点对整体影响更大，否则一个只有 2 个样本的纯净子节点就能把整个分裂的评分骗高。

**③ 两者相减**

$$
\operatorname{Gain}=\text{切分前的不纯度}-\text{切分后的期望不纯度}
$$

树会遍历所有可能的分裂点，选 $\operatorname{Gain}$ 最大的那个特征和阈值来切。

### 追问：决策树为什么容易过拟合？

树可以不断切分训练集，把噪声也记住。尤其 `max_depth` 很大、叶子样本很少时，训练误差会很低但泛化差。

### 回归树如何选择分裂？

常以子节点内平方误差最小为目标，每个叶节点预测该叶训练标签均值：

$$
\sum_{c\in\{\text{left,right}\}}
\sum_{i\in S_c}(y_i-\bar y_c)^2
$$

$\bar y_c$ 是子节点 $S_c$ 内标签的均值，也就是这个叶子将来的预测值。式子在问：按这个特征和阈值切一刀，左右两边各自用自己的均值去预测，总的平方误差是多少。枚举所有切法，取最小的那个。

和分类树是同一个套路，只是把「纯不纯」换成了「散不散」——分类看类别混不混，回归看数值齐不齐。之所以用均值当预测值，也正是因为在一组数里最小化平方和的那个常数就是均值（这一点和第 2 节 MSE 学的是条件均值是同一件事）。

### 如何防止过拟合？

- 限制 `max_depth`、`max_leaf_nodes`。
- 设置 `min_samples_split`、`min_samples_leaf`。
- 要求最小 impurity decrease。
- Cost-complexity post-pruning。
- 通过交叉验证调参。
- 使用 Random Forest、Gradient Boosting 等 ensemble 降低单树方差。

---

## 6. K-Means

从这一节开始进入无监督学习。K-Means、GMM 和 Hierarchical Clustering 都在没有标签时寻找数据结构，但它们对“什么算一个簇”有不同假设：

- K-Means：离同一个中心点较近的样本属于一簇。
- GMM：来自同一个概率分布成分的样本属于一簇。
- Hierarchical Clustering：样本可以形成多层嵌套结构。

### 目标

$$
\min_{\{C_k,\mu_k\}}
\sum_{k=1}^{K}
\sum_{x_i\in C_k}
\|x_i-\mu_k\|_2^2
$$

$C_k$ 是第 $k$ 个簇包含的样本集合，$\mu_k$ 是这个簇的中心（centroid）。内层求和把簇内每个点到自己中心的距离平方加起来，外层再把 $K$ 个簇加总——这个总量就是 sklearn 里的 **inertia**，可以理解成「所有点离自己中心的总偏离程度」。

难点在于 $\min$ 下面挂着两组未知量：**怎么分组**（$C_k$）和**中心放哪**（$\mu_k$），而且互相纠缠——中心定了才好分组，分组定了才能算中心。Lloyd 算法的办法就是轮流固定一个、优化另一个。

另外注意这里用的是平方欧氏距离，这正是 K-Means 偏爱球形簇、且对特征尺度敏感的根源。

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

---

## 7. EM 算法

K-Means 的簇分配是直接确定的，但 GMM 中“一个样本来自哪个 Gaussian”是看不见的 Latent Variable。EM 解决的正是这种“隐藏变量未知，模型参数也未知”的循环问题。

EM 用于含 Latent Variable 或缺失变量的概率模型。设观测为 $X$、隐变量为 $Z$、参数为 $\theta$。

### E-Step

用旧参数计算隐变量后验：

$$
q(Z)=p(Z\mid X,\theta^{\text{old}})
$$

意思是：拿现在手上这套参数，回过头去猜每个样本的隐变量。在 GMM 里就是问「这个点有多大概率来自第 2 个高斯」。注意猜出来的 $q$ 是一个**概率分布**，不是硬指派。

并构造 complete-data log-likelihood 的期望：

$$
Q(\theta,\theta^{\text{old}})
=\mathbb E_{Z\sim q}
[\log p(X,Z\mid\theta)]
$$

思路是这样：如果 $Z$ 是已知的（每个点属于哪个成分都摆在明面上），likelihood $\log p(X,Z\mid\theta)$ 会非常好写、好优化。可惜它未知，那就退一步——按刚才猜出的分布 $q$ 求个期望，得到一个「加权版」的 likelihood。

式子里两个 $\theta$ 分工不同：$\theta^{\text{old}}$ 只用来产生权重（已经固定住了），真正待优化的自变量是 $\theta$。

### M-Step

$$
\theta^{\text{new}}
=\arg\max_\theta
Q(\theta,\theta^{\text{old}})
$$

拿着这份加权 likelihood 去更新参数。在 GMM 里就是用 responsibility 当权重，重算每个成分的均值、协方差和混合权重。算完再回到 E-Step，用新参数重猜一遍隐变量，如此循环。

一句话：**E-Step 用参数猜隐变量，M-Step 用隐变量更新参数**，两边轮流往上爬。它和 K-Means 的「分配—更新」两步是同一个骨架，区别只在于 EM 猜的是概率、K-Means 猜的是硬归属。

EM 可从 Jensen inequality / evidence lower bound 推导。标准条件下每轮不会降低观测数据 log-likelihood，但只保证收敛到 stationary point，可能是局部最优或鞍点。初始化、局部最优和退化解都需要处理。

常见应用包括 GMM、隐变量模型、缺失数据估计和 HMM 的 Baum-Welch。

---

## 8. GMM 与 K-Means

GMM 可以看作 K-Means 的概率化版本：K-Means 将每个样本硬分给一个簇，GMM 则输出样本属于每个 Component 的概率。

### GMM

Gaussian Mixture Model 的密度为：

$$
p(x)
=\sum_{k=1}^{K}
\pi_k\mathcal N(x\mid\mu_k,\Sigma_k),
\qquad \sum_k\pi_k=1
$$

这个式子描述的是一个两步的生成过程：**先按概率 $\pi_k$ 掷一个 $K$ 面骰子选出一个成分，再从那个高斯里采一个点**。

- $\pi_k$：mixture weight，第 $k$ 个成分被选中的概率，加起来必须等于 1。
- $\mu_k,\Sigma_k$：这个成分的中心和形状（协方差决定它是圆是椭圆、朝哪个方向拉长）。

所以 GMM 就是「几个高斯叠在一起」，能拟合单个高斯搞不定的多峰数据。

参数包括 mixture weights、均值和协方差，通常使用 EM 估计。E-step 计算 responsibility：

$$
\gamma_{ik}
=p(z_i=k\mid x_i)
$$

这是把上面的生成过程反过来问：既然已经看到了 $x_i$ 这个点，它当初是从第 $k$ 个成分里出来的概率有多大。由 Bayes rule 得到：

$$
\gamma_{ik}
=\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}
$$

分子是「这个成分本身有多常见 × 它生成这个点的可能性」，分母把所有成分的这一项加起来做归一化，所以 $\sum_k\gamma_{ik}=1$。它就是 K-Means 里 0/1 硬分配的软化版——一个点可以 70% 属于簇 A、30% 属于簇 B。

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

这一节把 Decision Tree 与后面的 Ensemble Learning 连起来：单棵深树的主要问题是 Variance 高，而 Random Forest 通过训练许多低相关的树并取平均来降低 Variance。

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

Naive Bayes 与 Logistic Regression 都能做分类，但思路不同：Logistic Regression 直接学习 $p(y\mid x)$，Naive Bayes 先建模 $p(x\mid y)$ 和 $p(y)$，再通过 Bayes Rule 得到类别后验概率。

Naive Bayes 基于 Bayes rule：

$$
p(y\mid x)
\propto p(y)\prod_j p(x_j\mid y)
$$

完整的 Bayes rule 是 $p(y\mid x)=\frac{p(y)\,p(x\mid y)}{p(x)}$。分母 $p(x)$ 对每个类别都是同一个数，只是比大小的话可以直接扔掉，所以写成正比号 $\propto$。

剩下两项各有分工：

- $p(y)$ 是**先验**——这一类本来占多大比例（垃圾邮件本来就占三成，那它起跑线就高）。
- $p(x\mid y)$ 是**似然**——如果真是这一类，出现这些特征的可能性有多大。

“Naive” 指的就是把这个似然拆成连乘 $\prod_jp(x_j\mid y)$，即假设特征在给定类别后条件独立。真实语言里当然不独立（「免费」和「中奖」经常一起出现），但拆开之后每一项只需要数一个词的频率，参数量从指数级降到线性，这才是它训练快、少数据也能跑的原因。

预测时对每个类别算一遍再取 $\arg\max_y$。实践中会取对数把连乘变连加，否则几百个小概率乘在一起会直接下溢成 0。这个假设通常不完全成立，但在文本分类中常常够用。

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

KNN 与前面的模型不同：它几乎不进行参数训练，而是在预测时直接查找附近的训练样本，因此也叫 Instance-Based Learning。

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

前面的 Random Forest 是 Bagging 的代表。本节进一步比较 Bagging、Boosting 和 Stacking，解释树模型为什么可以通过不同组合方式获得更稳定或更准确的结果。

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

K-Means 和 GMM 直接给出某个 $K$ 下的聚类结果；Hierarchical Clustering 则生成一棵层次树，可以观察数据在不同粒度下如何合并。

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