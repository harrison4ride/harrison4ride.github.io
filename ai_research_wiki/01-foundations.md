# 01. 基础与神经网络机制

[返回目录](README.md)

本章覆盖正则化、损失函数、优化、反向传播、激活函数等深度学习的“机制层”。概念性定义见 [00. 核心概念](00-ml-concepts.md)，评估指标见 [02. 模型评估与指标](02-evaluation.md)。

## 1. L1 与 L2 正则化

### 30 秒回答

L1 和 L2 都是在原本的 loss 后面加一个“惩罚项”，目的都是不让模型权重太夸张，从而减少过拟合。

$$
\mathcal L_{\text{L1}}
=\mathcal L_{\text{data}}+\lambda\sum_i |w_i|
$$

$$
\mathcal L_{\text{L2}}
=\mathcal L_{\text{data}}+\lambda\sum_i w_i^2
$$

最重要的区别：

- **L1 更像“砍掉不重要特征”**：它容易把一些权重直接变成 0。
- **L2 更像“让所有权重都别太大”**：它通常把权重变小，但很少让权重刚好等于 0。

所以：

- 想做特征选择、希望模型更稀疏：用 L1。
- 大多数特征都有点用，只是不想某些权重特别大：用 L2。
- 不确定，或者既想稀疏又想稳定：用 Elastic Net，也就是 L1 + L2。

### L1 惩罚真的比 L2 弱吗？

不能这么说。**L1 和 L2 谁惩罚更强，取决于权重 $w$ 有多大。**

| $w$ | L1: $|w|$ | L2: $w^2$ |
| --- | ---: | ---: |
| 0.1 | 0.1 | 0.01 |
| 1 | 1 | 1 |
| 10 | 10 | 100 |

结论：

- 当 $|w|>1$，L2 惩罚更强，因为平方会把大权重放大。
- 当 $0<|w|<1$，L1 反而更强，因为 $|w| > w^2$。

所以 L2 对“大权重”打得更狠；L1 对“小权重”也会持续往 0 推。

### 为什么 L1 更容易让权重变成 0？

看梯度最直观。

L1 的惩罚是：

$$
|w|
$$

它的梯度大致是：

$$
\operatorname{sign}(w)
$$

意思是：只要 $w$ 不是 0，L1 都用差不多固定的力度把它往 0 推。

L2 的惩罚是：

$$
w^2
$$

它的梯度是：

$$
2w
$$

当 $w$ 很小，比如 $w=0.001$，L2 的梯度只有 $0.002$，推力也很小。所以 L2 更像是“慢慢缩小权重”，而不是直接把它清零。

一句话：

> **L1 会把不重要的权重清成 0；L2 会把权重整体压小，但通常保留它们。**

### 什么时候用 L1，什么时候用 L2？

#### 用 L1 的场景

- 特征特别多，但你觉得真正有用的只有一小部分。
- 你想让模型自动做 feature selection。
- 你希望模型更容易解释，因为很多特征权重会变成 0。
- 例子：文本 bag-of-words、one-hot 高维稀疏特征、广告/推荐里的大量类别特征。

#### 用 L2 的场景

- 你觉得大多数特征都有一点用，不想直接删掉。
- 特征之间相关性强，想让模型更稳定。
- 你主要担心某些权重过大导致过拟合。
- 例子：线性回归、逻辑回归、神经网络里的 weight decay。

#### 用 Elastic Net 的场景

Elastic Net = L1 + L2：

$$
\lambda_1\|w\|_1+\lambda_2\|w\|_2^2
$$

它适合：

- 特征很多。
- 特征之间还有相关性。
- 你既想要 L1 的稀疏性，又想要 L2 的稳定性。

### 面试怎么说？

可以这样回答：

> L1 和 L2 都是为了防止权重太大、减少过拟合。L1 用绝对值惩罚，容易把小权重压成 0，所以适合做特征选择；L2 用平方惩罚，对大权重惩罚特别强，所以更适合让模型整体更平滑、更稳定。不是简单说 L1 比 L2 惩罚弱，因为小权重时 L1 反而比 L2 强，大权重时 L2 更强。

### Weight Decay 是什么？

Weight decay 可以直观理解成：

> 每次更新参数时，顺手把权重往 0 拉一点点。

普通梯度下降大概是：

$$
w \leftarrow w - \eta \nabla L
$$

加上 weight decay 后，可以理解为：

$$
w \leftarrow (1-\eta\lambda)w - \eta \nabla L
$$

其中：

- $\eta$ 是 learning rate。
- $\lambda$ 是 weight decay 强度。
- $(1-\eta\lambda)w$ 表示先把权重缩小一点。

所以 weight decay 的效果是：

- 防止权重越来越大。
- 让模型不要过分依赖某几个特别大的参数。
- 通常能减少过拟合。

### Weight Decay 和 L2 是一回事吗？

在 **普通 SGD** 里，L2 regularization 和 weight decay 基本等价。

也就是说，把 L2 加进 loss：

$$
\mathcal L = \mathcal L_{\text{data}}+\lambda\sum_i w_i^2
$$

最后产生的效果就像每次更新时让权重衰减一点。

但是在 **Adam / AdamW** 这类自适应优化器里，要小心：

- Adam + L2：L2 的梯度会进入 Adam 的动量和自适应缩放里。
- AdamW：直接把 weight decay 和梯度更新分开做，更接近“每次把权重乘小一点”这个直觉。

所以现在训练神经网络时，很多人更常用 **AdamW**，而不是 Adam + L2。

面试一句话：

> Weight decay 就是每次参数更新时，把权重往 0 缩一点，防止权重过大。在 SGD 里它和 L2 正则基本等价，但在 Adam 里不完全等价，所以常用 AdamW 做 decoupled weight decay。

### 为什么常用 L1/L2，而较少使用 L3/L4？

- L1 是凸的并能产生稀疏解，L2 光滑、旋转对称且优化简单。
- $p>2$ 的 $L_p$ 惩罚会更强地惩罚大权重，但通常没有 L1 的稀疏性，也缺少 L2 的统计、数值和解析便利。
- $0<p<1$ 更接近直接计算非零参数个数，但非凸，优化更困难。

这并不意味着 L3/L4 “不能用”。正则项应由先验和目标决定，例如 group lasso、nuclear norm、total variation 都适合特定结构。

---

## 2. L1 / L2 / L3 Loss

先分清楚：

- **Loss**：惩罚预测错了多少，比如 $\hat y$ 和 $y$ 差多远。
- **Regularization**：惩罚模型权重多大，比如 $w$ 多大。

它们名字都叫 L1/L2，但对象不一样。

### L1 Loss：MAE

L1 loss 也叫 Mean Absolute Error：

$$
L_1=\frac1n\sum_i |y_i-\hat y_i|
$$

特点：

- 对异常值没那么敏感。
- 误差从 10 变 100，惩罚也只是线性变大。
- 缺点是在误差为 0 的地方不光滑，优化时可能没 MSE 顺。

适合：数据里有 outlier，或者你不想让少数极端误差主导训练。

### L2 Loss：MSE

L2 loss 也叫 Mean Squared Error：

$$
L_2=\frac1n\sum_i (y_i-\hat y_i)^2
$$

特点：

- 对大误差惩罚很重。
- 误差从 10 变 100，惩罚从 100 变 10000。
- 光滑，好优化。
- 缺点是对 outlier 敏感。

适合：普通回归问题，尤其假设误差接近高斯分布时。

### L3 Loss 是什么？

如果直接写：

$$
\sum_i (y_i-\hat y_i)^3
$$

这通常不是好 loss，因为误差有正有负，三次方也有正有负，可能互相抵消，loss 甚至可能越小越奇怪。

如果写成绝对值三次：

$$
\sum_i |y_i-\hat y_i|^3
$$

它会比 MSE 更狠地惩罚大误差。

但 L3 loss 很少作为默认选择，因为：

- 对 outlier 更敏感。
- 大误差梯度更容易爆。
- 实际上 MSE、MAE、Huber loss 已经覆盖了大多数需求。

### Huber Loss：MAE 和 MSE 的折中

Huber loss 小误差像 MSE，大误差像 MAE：

$$
L_\delta(e)=
\begin{cases}
\frac12 e^2, & |e|\le \delta\\
\delta(|e|-\frac12\delta), & |e|>\delta
\end{cases}
$$

其中 $e=y-\hat y$。

它适合：你希望小误差处好优化，同时又不想被 outlier 影响太大。

### 面试怎么说？

> L1/L2 loss 是在惩罚预测误差，L1 是绝对误差 MAE，抗 outlier；L2 是平方误差 MSE，对大误差惩罚更重，优化更平滑。L3 loss 不常用，直接三次方会有正负抵消，绝对值三次又太受 outlier 影响。实际常用 MAE、MSE 或 Huber。

> MSE、KL Divergence 与 Cross-Entropy 三者的关系，见 [04. 经典机器学习](04-classical-ml.md) 的对应小节。

---

## 3. 防止过拟合的方法

### 数据

- 收集更多有代表性的数据。
- Data augmentation。
- 去重、纠正标签噪声，避免 train/test leakage。

### 模型

- 降低容量。
- L1/L2、weight decay。
- Dropout、stochastic depth。
- 参数共享或预训练后进行受限微调。

### 训练

- Early stopping。
- 学习率和训练轮数控制。
- Label smoothing。
- 交叉验证和可靠的 validation split。

交叉验证本身不直接约束模型参数，不属于正则项；它通过更可靠地选择模型、超参数和停止点，降低因验证集偶然性造成的过拟合。

### BatchNorm 是否属于正则化？

**先说它做什么**：对一个 mini-batch 内、每个特征/通道在 batch 维度做标准化，再用可学习的 $\gamma, \beta$ 放缩平移：

$$
\hat x=\frac{x-\mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2+\epsilon}},\qquad y=\gamma\hat x+\beta
$$

它的首要作用是**优化**：平滑 loss 曲面、允许更大学习率、加速并稳定训练，降低对初始化的敏感度。

**算不算正则化？** 算「隐式/附带」正则，但不是正经的正则化方法：

- **为什么有正则效果**：每个样本的归一化依赖同一 batch 里的其他样本，而 batch 是随机采样的，统计量 $\mu, \sigma$ 带噪 → 相当于给激活注入随机扰动（机制上类似 Dropout）。用了 BN 后常可减弱甚至去掉 Dropout。
- **但本质是优化技巧**：正则只是副产品；不应把它当作通用防过拟合方法。
- 正则强度随 batch 变化：batch 越小统计越抖、正则越强（也越不稳）。小 batch、序列模型或分布漂移时这种效果不可靠。
- **训练/推理不一致**：训练用当前 batch 统计量，推理用训练期间维护的滑动平均（running mean/var）。

> 面试一句话：BatchNorm 首要是优化技巧、不是防过拟合方法，但因依赖 batch 统计噪声而带有隐式正则的副作用。

### BatchNorm vs LayerNorm

核心区别只有一句：**沿哪个维度算均值和方差。**

| | BatchNorm | LayerNorm |
| --- | --- | --- |
| 归一化维度 | 跨**样本**（batch 维），每个特征独立 | 跨**特征**（单样本内），每个样本独立 |
| 依赖 batch | 是，统计量来自整个 batch | 否，batch size = 1 也行 |
| 训练/推理 | 不一致，需 running mean/var | 完全一致，无需滑动统计 |
| 小 batch / 变长序列 | 差、不稳 | 不受影响 |
| 典型场景 | CNN / 视觉 | Transformer / RNN / NLP |
| 正则副作用 | 有（batch 噪声） | 基本没有 |

形象地说（行 = 样本、列 = 特征）：**BN 竖着归一化**（看一列特征在整个 batch 上的分布），**LN 横着归一化**（看一个样本自己所有特征的分布）。Transformer 用 LN，是因为序列变长、batch 内长度不齐会让 BN 统计量不稳，且自回归推理常一次只来一个 token，而 LN 与 batch 无关、训练推理一致，天然合适。

---

## 4. 梯度下降的类型

- **Batch GD**：每步用全量数据，稳定但慢。
- **SGD**：每步用一个样本，噪声大但更新快。
- **Mini-batch GD**：每步用一小批样本，深度学习最常用。

Batch size 太小，梯度噪声大；太大，显存高、泛化和优化也可能变差。

### 追问：Mini-batch size 会影响什么？

- 小 batch：梯度噪声大，**常泛化更好**，但训练不稳定、吞吐低。
- 大 batch：吞吐高、梯度稳，但显存大，且不调学习率时易掉进尖锐极小值、泛化变差，常需 learning-rate warmup 和调参。

不要只说“大 batch 更快”，还要提优化稳定性和泛化。

### 为什么小 batch 往往泛化更好？

一句话：**小 batch 的梯度噪声大，相当于隐式正则，把模型推向「平坦极小值」，对训练/测试分布的偏移更鲁棒。**

- **噪声即正则**：batch 估计的梯度是对全量梯度的带噪估计，噪声协方差 $\propto 1/\text{batch}$。batch 越小越抖，SGD 在 loss 曲面上带噪游走，类似给参数加扰动 → 抑制过拟合。
- **平坦 vs 尖锐极小值**（Keskar et al. 2016）：大 batch 易收敛到**尖锐极小值**（曲率大、对扰动敏感），泛化差；小 batch 因噪声落在**平坦极小值**（周围一片低 loss），train/test 曲面有轻微错位时 loss 也不易暴涨 → 泛化稳。
- **关键纠正——真正起作用的是「噪声尺度」而非 batch 本身**：噪声尺度 $\approx \text{lr}/\text{batch}$。所以「小 batch 泛化好」一部分是因为加大 batch 时没同步加大学习率。**线性缩放法则**（Goyal et al. 2017）：batch 扩大 $k$ 倍、学习率也乘 $k$（配 warmup），大 batch 能补回大部分泛化差距。

### 那 batch=1 岂不是泛化最好？

不会——泛化与 batch 的关系是**倒 U**，不是单调下降，batch=1 在另一端噪声过头：

- **过量噪声会破坏收敛本身**：噪声当正则的前提是「还能收敛到低训练 loss，只在低 loss 解里挑平坦的」。batch=1 抖到到不了低 loss 区，直接**欠拟合**，train/test 一起变差——这不是泛化好，是优化坏了。
- **噪声尺度反噬**：为了训得动必须把学习率调得很小，于是噪声尺度 $\text{lr}/\text{batch}$ 又被拉回来，想要的那点正则没了，还换来极慢收敛——「最大正则」和「能收敛」无法兼得。
- **BN 失效 + 算力浪费**：1 个样本算不出有意义的方差（=0），BatchNorm 不可用；且用不上 GPU 并行，墙钟极慢。

经验甜点区：视觉常 **2–32**，大模型/任务常 **32–512**——小，但几乎不是 1。

### 追问：SGD、Momentum、Adam 怎么选？

- **SGD(+Momentum)**：泛化常更好，是 CV 里很多 SOTA 的选择，但对学习率敏感、收敛慢。
- **Adam / AdamW**：自适应学习率，收敛快、对学习率不那么敏感，是 Transformer/LLM 训练的默认。AdamW 把 weight decay 解耦，更稳。
- 经验：大模型语言任务用 AdamW；追求极致泛化且能精调学习率时用 SGD+Momentum。

---

## 5. 反向传播

反向传播是链式法则在计算图上的高效实现。若：

$$
z=f(x),\qquad y=g(z),\qquad \mathcal L=h(y)
$$

则：

$$
\frac{\partial\mathcal L}{\partial x}
=\frac{\partial\mathcal L}{\partial y}
\frac{\partial y}{\partial z}
\frac{\partial z}{\partial x}
$$

反向传播从 loss 开始，按计算图逆拓扑顺序传播 vector-Jacobian product，并累加一条变量通过多条路径对 loss 的贡献。

### 与梯度下降的区别

- Backprop：计算梯度。
- Optimizer：使用梯度更新参数，例如 SGD、AdamW。

### 常见工程问题

- 梯度消失/爆炸。
- 忘记清零梯度导致意外累积。
- 不恰当的 `detach` 或原地操作破坏计算图。
- 混合精度下 overflow/underflow。
- 张量 shape 广播成功但语义错误。

---

## 6. 梯度消失与梯度爆炸

### 梯度消失

梯度消失是反向传播时多层导数连乘后越来越小，前层几乎学不动。RNN 长序列、sigmoid/tanh 饱和区、很深网络都容易出现。

常见处理：

- ReLU/GELU 等非饱和激活。
- 残差连接、LayerNorm/BatchNorm。
- 合理初始化（如 He、orthogonal）。
- LSTM/GRU 或 Attention 缩短依赖路径。
- 更短的反传路径和更密集监督。

### 梯度爆炸

- Gradient clipping，常按 global norm：

$$
g\leftarrow
g\cdot\min\left(1,\frac{\tau}{\|g\|}\right)
$$

- 降低学习率。
- 稳定初始化和归一化。
- 检查异常序列与 loss 数值。

Gradient clipping 主要处理爆炸，不能恢复已经消失的梯度。RNN 场景下的更多细节见 [05. NLP、RNN 与词向量](05-nlp-rnn.md)。

---

## 7. 激活函数

### 为什么需要非线性激活？

没有非线性，多层线性变换的复合仍是线性，网络退化为单层线性模型，无法拟合复杂函数。激活函数引入非线性，使网络能逼近任意函数。

### Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

输出 $0$ 到 $1$，可做概率，但两端饱和、容易梯度消失，且非零中心。

### Tanh

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

输出 $-1$ 到 $1$，零中心，但仍可能饱和。

### ReLU

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

简单高效，缓解梯度消失，但负半轴梯度为零，可能出现 dead ReLU。

### GELU / SiLU / SwiGLU

- **GELU**：$x\Phi(x)$，按输入大小平滑门控，BERT、GPT 常用。
- **SiLU/Swish**：$x\sigma(x)$，平滑、非单调。
- **SwiGLU**：门控线性单元，现代 Transformer FFN 常用，公式与动机见 [06. LLM 基础](06-llm-foundations.md)。

这些现代激活在相近计算预算下通常优于 ReLU。

### 追问：Dying ReLU 怎么办？

Dying ReLU 是某些神经元长期输出 0，梯度也接近 0。常见处理：

- 降低 learning rate。
- 使用 He initialization。
- 换 Leaky ReLU、GELU、SiLU。
- 加 normalization 或 residual connection。

---

## 8. Softmax 及其反向传播

### 前向

$$
s_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

数值稳定实现先减去最大 logit：

$$
s_i
=\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}
$$

因为 softmax 对所有 logits 加同一常数不变，这可避免指数溢出。

### Jacobian

$$
\frac{\partial s_i}{\partial z_j}
=s_i(\delta_{ij}-s_j)
$$

即：

- 对角线：$s_i(1-s_i)$。
- 非对角线：$-s_is_j$。

矩阵形式：

$$
J=\operatorname{diag}(s)-ss^\top
$$

### Softmax + Cross-Entropy

若标签 one-hot 为 $y$，

$$
\mathcal L=-\sum_i y_i\log s_i
$$

则梯度可化简为：

$$
\frac{\partial\mathcal L}{\partial z}=s-y
$$

面试手写时通常不应显式构造 $C\times C$ Jacobian；直接利用该化简更高效、稳定。手写实现见 [12. ML Coding](12-ml-coding.md)。

---

## 9. Dropout

### 直观理解

Dropout 做的事很简单：

> **训练时，随机把一部分神经元输出变成 0；推理时，所有神经元都正常工作。**

比如某一层有 10 个神经元，dropout rate $p=0.2$，训练时每次大约随机关掉 2 个。下一次 forward 又随机关掉另一批。

这样做的目的：

- 不让模型太依赖某几个神经元。
- 逼模型用更多不同特征一起做判断。
- 减少过拟合。

可以把它理解成：训练时不断让模型在“残缺网络”上练习，所以它不会死记某条固定路径。

### 为什么训练时要除以 $1-p$？

如果 dropout rate 是 $p=0.5$，原来两个神经元输出都是 1：

```text
不 Dropout: 1 + 1 = 2
Dropout 后: 只剩一个 1，总和变成 1
```

输出平均变小了。为了让训练时的输出规模和推理时差不多，现代框架会在训练时把保留下来的神经元放大：

$$
\tilde h_i=\frac{m_i h_i}{1-p}
$$

其中：

- $m_i=0$：这个神经元被关掉。
- $m_i=1$：这个神经元保留。
- $p$：被关掉的概率。

例子：$p=0.5$ 时，保留下来的输出会乘以 $1/(1-0.5)=2$。

这样训练时的期望不变：

$$
\mathbb E[\tilde h_i]=h_i
$$

这叫 **inverted dropout**。好处是：推理时直接关掉 dropout，不需要再手动缩放。

### 为什么有效？

- **减少依赖**：某个神经元可能随时被关掉，所以其他神经元也要学会有用特征。
- **类似集成**：每次训练都像在训练一个不同的子网络，最后共享同一套参数。
- **增加噪声**：训练更难一点，但泛化通常更好。

### 什么时候用？

- 模型明显过拟合时可以加。
- 数据少、模型大时常有帮助。
- CNN、MLP、Transformer 都可能用，但位置和比例要调。

### 注意

- 推理时必须调用 evaluation mode，否则输出会随机。
- Dropout rate 过大，信息丢太多，会欠拟合。
- BatchNorm 与 Dropout 同用时，随机激活可能影响 batch 统计。
- 大模型数据很多时，不一定需要很高 dropout。

### 面试怎么说？

> Dropout 是一种防过拟合方法。训练时随机把一部分神经元输出置 0，迫使模型不要依赖某几个特征；推理时关闭 Dropout。现代框架用 inverted dropout，训练时把保留的激活除以 $1-p$，这样推理时不需要额外缩放。

---

## 10. Perceptron

Perceptron 是早期线性二分类模型：

$$
\hat y=\operatorname{sign}(w^\top x+b)
$$

如果样本被错分，就按标签方向更新权重。它只能学线性可分边界，是理解线性分类和神经网络的基础。

### 追问：Perceptron 为什么不能解决 XOR？

Perceptron 是线性分类器，只能画一条线/超平面。XOR 不是线性可分，所以单层 perceptron 解决不了；需要非线性特征、多层网络或 kernel 方法。

---

## 11. CNN 基础

CNN 用卷积核在局部区域共享参数，适合图像等有局部结构的数据。

- **Convolution**：提取局部模式。
- **Pooling / stride**：降采样，扩大感受野。
- **Parameter sharing**：同一个卷积核扫过不同位置，参数少于全连接。

现在视觉任务也大量使用 ViT，但 CNN 仍是理解局部归纳偏置的重要基础。

### 追问：为什么 CNN 适合图像？

- **局部连接**：相邻像素相关性强，卷积核只看局部感受野。
- **参数共享**：同一模式（如边缘）出现在任何位置都能被同一核检测，大幅减少参数。
- **平移等变性**：物体平移后特征图也对应平移，符合图像的归纳偏置。

---

## 12. 迁移学习（Transfer Learning）

### 30 秒回答

迁移学习是把在一个任务/领域上学到的知识（通常是预训练模型的参数）迁移到一个新的、数据较少的目标任务上，而不是从零训练。

它之所以有效，是因为大规模预训练学到的低层和中层表示（边缘、纹理、语法、语义）往往是通用的，目标任务只需在此基础上做少量适配。

### 常见做法

- **Feature extraction**：冻结大部分 backbone，只训练新的输出 head。数据极少时首选。
- **Fine-tuning**：解冻部分或全部层，用较小学习率继续训练。数据稍多时效果更好。
- **逐步解冻**：先训 head，再自上而下逐步解冻，兼顾稳定和效果。
- **PEFT**：LoRA、Adapter、Prompt Tuning 等只训练少量参数，是 LLM 时代主流，省显存、易部署多任务。

### 什么时候用？

- 目标任务标注数据少。
- 存在与目标相近领域的强预训练模型。
- 想节省训练成本和时间。

### 追问：迁移学习可能失败或有害吗？

会。当源任务与目标任务差异过大时可能出现 **负迁移（negative transfer）**，预训练知识反而拖累目标任务。此外：

- 学习率太大会破坏预训练表示（catastrophic forgetting）。
- 源域偏差/不公平也会被迁移过来。
- 目标数据分布和源差异大时，需更多 fine-tuning 或领域自适应。

标注数据有限时的完整策略，见 [11. Production ML](11-production-ml.md)。

### 面试怎么说？

> 迁移学习是用预训练模型的通用表示初始化新任务，再做特征提取或微调，数据少时特别有效。现在 LLM 主要用 LoRA 这类 PEFT 做轻量适配。但要注意源目标差异过大会负迁移，微调学习率太大会灾难性遗忘。
