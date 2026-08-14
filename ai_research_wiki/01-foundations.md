# 01. 基础与神经网络机制

[返回目录](README.md)

## 1. L1 与 L2 正则化

### 为什么需要正则化？

模型出现 Overfitting, 从 Bias-Variance 的角度看，这通常意味着模型的 **Variance 过高**：模型对训练数据过于敏感，学到了训练集中的噪声或偶然规律。

正则化的基本思路是：

> 在拟合训练数据的同时，限制模型参数不要变得过于极端。

L1 和 L2 都是在原本的 loss 后面加一个“惩罚项”，目的都是不让模型权重太夸张，正则化本质上是在 **降低 Variance**，但也可能带来更高的 **Bias**。

$$
\mathcal L_{\text{L1}}
=\mathcal L_{\text{data}}+\lambda\sum_i |w_i|
$$

$$
\mathcal L_{\text{L2}}
=\mathcal L_{\text{data}}+\lambda\sum_i w_i^2
$$

| 对比           | L1                  | L2                     |
| -------------- | ------------------- | ---------------------- |
| 惩罚方式       | 权重绝对值          | 权重平方               |
| 对小权重的效果 | 持续向 0 推         | 越接近 0，推力越小     |
| 最终结果       | 容易产生大量 0 权重 | 通常只把权重变小       |
| 主要作用       | 稀疏化、特征选择    | 限制大权重、提高稳定性 |
| 常见场景       | 高维稀疏特征        | 线性模型和神经网络     |

简单来说：

- **L1 更像删除不重要的特征。**
- **L2 更像让模型不要过度依赖某些特征。**

### 为什么 L1 会产生稀疏权重？

L1 的惩罚项是：

$$
|w|.
$$

当 $w\neq 0$ 时，它对权重的影响大致为：

$$
\frac{\partial |w|}{\partial w}
=
\operatorname{sign}(w).
$$

这意味着，只要权重还没有变成 0，L1 就会用近似固定的力度把它向 0 推。

此外，$|w|$ 在 $w=0$ 的位置有一个尖点。使用适合 L1 的优化方法时，一个较小且不重要的权重到达 0 后，可以直接保留在 0。

因此，L1 容易得到这样的结果：

$$
w=[1.2,\;0,\;0,\;-0.7,\;0].
$$

这相当于模型只保留少数有用特征，因此 L1 也可以用于自动 Feature Selection。

### 为什么 L2 通常只缩小权重？

L2 的惩罚项是：

$$
\frac{1}{2}w^2.
$$

它对权重的梯度是：

$$
\frac{\partial \frac{1}{2}w^2}{\partial w}
=
w.
$$

权重越大，L2 把它向 0 拉的力度越大；权重越小，这个力度也越小。

例如：

- 当 $w=10$ 时，L2 会产生很强的限制；
- 当 $w=0.001$ 时，L2 的影响已经非常小。

所以 L2 更容易得到这样的结果：

$$
w=[0.8,\;0.02,\;-0.01,\;-0.5,\;0.03].
$$

大权重被明显压小，但小权重通常不会刚好变成 0。因此，L2 主要用于控制权重大小，而不是删除特征。

### 什么时候使用 L1？

L1 适合“特征很多，但真正有用的可能只有少数几个”的情况，例如：

- 文本中的 Bag-of-Words 特征；
- 大量 One-Hot 类别特征；
- 高维广告或推荐系统特征；
- 希望模型能够自动选择特征；
- 希望模型更加稀疏、容易解释。

例如，一个线性模型原本使用 10,000 个特征。加入 L1 后，可能只有几百个特征的权重不为 0。

但如果多个特征高度相关，L1 可能只保留其中一个，并将其他相关特征设为 0。更换训练数据后，它选择的特征也可能发生变化。

### 什么时候使用 L2？

L2 适合“大多数特征都有一定作用，但不希望某些权重过大”的情况，例如：

- 线性回归；
- 逻辑回归；
- 特征之间存在较强相关性；
- 神经网络训练；
- 主要目标是提高模型稳定性，而不是删除特征。

当多个特征高度相关时，L2 通常会把权重分配给多个相关特征，而不是只保留其中一个。因此，它的结果通常比 L1 更稳定。

### 什么时候使用 Elastic Net？

如果特征很多，而且特征之间还存在较强相关性，可以结合 L1 和 L2：

$$
\mathcal L
=
\mathcal L_{\text{data}}
+
\lambda_1\sum_i|w_i|
+
\frac{\lambda_2}{2}\sum_iw_i^2.
$$

这就是 Elastic Net。

它同时具有：

- L1 的稀疏性，可以删除不重要的特征；
- L2 的稳定性，可以处理相互关联的特征。

因此，可以这样选择：

- 希望获得稀疏模型或自动选择特征：优先考虑 L1；
- 希望限制大权重并提高稳定性：优先考虑 L2；
- 既希望选择特征，又存在大量相关特征：考虑 Elastic Net。

### Weight Decay 是什么？

> 每次更新参数时，除了根据 Loss 更新权重，还额外把权重向 0 缩小一点。

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

> 在普通 SGD 中，L2 Regularization 和 Weight Decay 基本等价。

### 为什么 Adam 中更常使用 AdamW？

在 Adam 中，L2 产生的梯度也会经过 Adam 的动量计算和自适应缩放。因此，不同参数受到的实际衰减程度可能不同。

AdamW 将 Weight Decay 与 Loss 的梯度更新分开：

1. Adam 根据 Loss 的梯度更新参数；
2. Weight Decay 单独把参数缩小一点。

因此，AdamW 更符合“每次更新时让权重衰减一点”的定义，也更容易单独控制正则化强度。

简单来说：

- **SGD + L2**：通常等价于 Weight Decay；
- **Adam + L2**：不完全等价于 Weight Decay；
- **AdamW**：将 Weight Decay 与梯度更新分开处理。

这也是神经网络训练中经常使用 AdamW 的原因之一。

### 为什么常用 L1/L2，而较少使用 L3/L4？

- L1 是凸的并能产生稀疏解，L2 光滑、旋转对称且优化简单。
- $p>2$ 的 $L_p$ 惩罚会更强地惩罚大权重，但通常没有 L1 的稀疏性，也缺少 L2 的统计、数值和解析便利。
- $0<p<1$ 更接近直接计算非零参数个数，但非凸，优化更困难。

---

## 2. BatchNorm 与 LayerNorm

上一节的 L1、L2 是在 Loss 中直接限制模型权重，属于显式正则化。BatchNorm 和 LayerNorm 则作用于网络的中间激活值，主要目的是改善训练过程。
- **L1/L2**：限制模型参数，主要用于控制 Overfitting。
- **BatchNorm/LayerNorm**：调整激活值的分布，主要用于稳定优化。
- BatchNorm 可能附带一定正则效果，但这不是它的主要目的。

### BatchNorm

BatchNorm 对同一个 mini-batch 中、同一特征或通道的激活值进行标准化：

$$
\hat{x}
=
\frac{x-\mu_{\text{batch}}}
{\sqrt{\sigma_{\text{batch}}^2+\epsilon}},
\qquad
y=\gamma\hat{x}+\beta.
$$

其中：

- $\mu_{\text{batch}}$ 和 $\sigma_{\text{batch}}^2$ 是当前 batch 的均值和方差；
- $\epsilon$ 用于防止除以 0；
- $\gamma$ 和 $\beta$ 是可学习参数，让模型可以重新调整激活值的范围。

BatchNorm 的主要作用是让训练更加稳定，通常允许使用更大的学习率，并降低模型对参数初始化的敏感程度。由于每个 batch 的统计量存在随机变化，它也会带来一些类似正则化的噪声，但这种效果只是副作用。

### BatchNorm 为什么有助于训练？

如果不同层的激活值尺度差异很大，某些方向的梯度可能特别大，另一些方向则特别小，模型会更难优化。

BatchNorm 将激活值调整到相对稳定的范围，因此通常可以：

- 减少不同层之间的数值尺度差异；
- 让梯度更新更加稳定；
- 使用更大的 Learning Rate；
- 加快模型收敛；
- 降低对参数初始化的敏感程度。

因此，BatchNorm 首先是一种优化方法，而不是专门的防止 Overfitting 方法。

### BatchNorm 是否属于正则化？

BatchNorm 不属于 L1、L2 这种显式正则化，但可能产生隐式正则效果。

训练时，每个样本的标准化结果依赖当前 batch 中的其他样本。由于 mini-batch 是随机采样的，不同 batch 的均值和方差会有所不同，相当于给激活值加入了一些随机扰动。

这种扰动可能让模型减少对训练样本细节的依赖，因此有时能够缓解 Overfitting。

> BatchNorm 的主要作用是改善优化，正则化只是由 batch 统计噪声带来的附带效果。不能因为模型使用了 BatchNorm，就默认不再需要 Weight Decay、数据增强或其他正则化方法。

### BatchNorm 与 LayerNorm

两者的核心区别是：

> BatchNorm 在不同样本之间统计同一特征；LayerNorm 在单个样本内部统计不同特征。

| 对比                        | BatchNorm                     | LayerNorm                         |
| --------------------------- | ----------------------------- | --------------------------------- |
| 统计范围                    | 同一 batch 中的同一特征或通道 | 单个样本或 token 内的所有隐藏特征 |
| 是否依赖 Batch Size         | 是                            | 否                                |
| 训练与推理                  | 使用不同统计量                | 计算方式一致                      |
| 是否需要 Running Statistics | 需要                          | 不需要                            |
| 小 Batch 下的表现           | 可能不稳定                    | 通常不受影响                      |
| 常见场景                    | CNN、计算机视觉               | Transformer、LLM、NLP             |
| 隐式正则效果                | 可能有                        | 通常较弱                          |

假设一个 batch 可以表示为一个矩阵，其中行是样本、列是特征：

- BatchNorm 可以理解为沿列计算：比较不同样本的同一个特征。
- LayerNorm 可以理解为沿行计算：比较一个样本内部的不同特征。

对于 CNN，BatchNorm 通常还会同时使用空间位置上的数据，为每个通道计算统计量。

### 为什么 Transformer 通常使用 LayerNorm？

Transformer 的输入通常表示为：

$$
[\text{batch},\ \text{sequence length},\ \text{hidden dimension}].
$$

LayerNorm 对每个 token 的 Hidden Dimension 独立进行归一化，因此：

- 不依赖 Batch Size；
- 不需要不同样本具有相同的序列长度；
- 训练和自回归推理的计算方式一致；
- 即使一次只生成一个 token，也可以正常工作。

BatchNorm 则依赖 batch 中其他样本的统计量。在小 batch、变长序列和自回归生成场景中，这些统计量不够方便和稳定。

因此：

- CNN 通常更常使用 BatchNorm；
- Transformer 和 LLM 通常更常使用 LayerNorm 或 RMSNorm。

### Batch Size 如何影响 BatchNorm？

Batch Size 越小，用于估计均值和方差的样本越少：

- 统计量的随机性更大；
- 可能带来更强的隐式正则效果；
- 但统计量也可能不准确，导致训练不稳定。

Batch Size 越大：

- Batch Statistics 更稳定；
- BatchNorm 的随机扰动更小；
- 附带的正则效果可能减弱。

如果由于显存限制只能使用很小的 batch，可以考虑：

- LayerNorm；
- GroupNorm；
- 在多张 GPU 之间同步统计量的 SyncBatchNorm。

---

## 3. Batch Size、梯度下降与 Optimizer

模型训练的目标是找到一组参数，使整个训练集上的平均 Loss 最小。不同梯度下降方法的区别是：

> 每次更新参数时，使用多少训练样本来估计梯度。

### 三种梯度下降

#### Batch Gradient Descent

每一步都使用完整训练集计算梯度。

优点：

- 梯度准确、稳定；
- 相同参数下每次得到的梯度相同。

缺点：

- 数据量大时计算很慢；
- 每次更新都需要遍历完整数据集；
- 很难充分利用频繁更新带来的优化速度。

#### Stochastic Gradient Descent

每一步只使用一个样本计算梯度。

优点：

- 参数更新频繁；
- 单次更新计算量小；
- 梯度噪声可能带来一定隐式正则效果。

缺点：

- 梯度波动很大；
- 通常需要更小的学习率；
- 难以充分发挥 GPU 的并行计算能力。

#### Mini-batch Gradient Descent

每一步使用一小批样本计算梯度，是深度学习中最常见的方法。

它在两方面取得平衡：

- 比单样本 SGD 的梯度更稳定；
- 比完整 Batch GD 的更新更频繁、计算成本更低；
- 可以利用 GPU 对矩阵运算进行并行加速。

实际深度学习中常说的“SGD 训练”，通常使用的也是 Mini-batch SGD，而不是每次只使用一个样本。

### Batch Size 会影响什么？

Batch Size 不只是决定一次放多少数据，还会同时影响：

1. **梯度噪声**：Batch 越小，梯度估计的随机性通常越大。
2. **更新次数**：在相同 Epoch 数下，Batch 越大，每个 Epoch 的参数更新次数越少。
3. **显存使用**：Batch 越大，通常需要更多显存。
4. **硬件吞吐**：在一定范围内，Batch 越大越容易发挥 GPU 并行能力。
5. **泛化能力**：适量的梯度噪声有时可以减少 Overfitting。
6. **BatchNorm 统计量**：Batch 越小，BatchNorm 的均值和方差越不稳定。

因此，比较不同 Batch Size 时，不能只比较相同的 Epoch 数。因为 Batch Size 改变后，参数更新次数也发生了变化。

### Learning Rate 为什么要和 Batch Size 一起调整？

Batch Size 增大后，梯度通常更稳定，但每个 Epoch 的更新次数更少。如果仍然使用原来的 Learning Rate，模型的训练行为会发生明显变化。

常见经验是线性缩放：

$$
B_{\text{new}}=kB_{\text{old}}
\quad\Rightarrow\quad
\eta_{\text{new}}\approx k\eta_{\text{old}}.
$$

也就是 Batch Size 扩大 $k$ 倍时，Learning Rate 也尝试扩大 $k$ 倍。

但这只是起始调参规则，不是严格定律。Learning Rate 过大时，训练初期可能不稳定，因此大 Batch 训练通常还会配合 Learning Rate Warmup。

可以把 Batch Size 和 Learning Rate 的关系粗略理解为：

$$
\text{Gradient Noise Scale}
\propto
\frac{\text{Learning Rate}}{\text{Batch Size}}.
$$

因此，改变 Batch Size 时，通常也需要重新调整 Learning Rate 和训练步数。

### Batch Size 等于 1 是否泛化最好？

不一定。梯度噪声只有在模型能够正常收敛的情况下，才可能产生有益的正则效果。

Batch Size 等于 1 时：

- 梯度的随机性很大；
- 训练可能需要较小的 Learning Rate；
- GPU 并行利用率很低；
- 训练时间通常更长；
- 普通 BatchNorm 的统计量可能不可靠。

但 Batch Size 等于 1 并不是一定无法训练。在线学习、强化学习或某些小模型中也会使用单样本更新。

正确结论是：

> Batch Size 不是越小越好，而是在优化稳定性、泛化能力、显存和训练速度之间进行权衡。

不存在适用于所有任务的最佳 Batch Size。通常先根据显存和硬件效率选择可行范围，再通过验证集比较不同设置。

### Gradient Accumulation 是什么？

当显存无法容纳较大的 batch 时，可以连续计算多个小 batch 的梯度，累积后再更新一次参数。

如果：

- 每张 GPU 的 Micro-batch Size 为 $B$；
- 使用 $N$ 张 GPU；
- 累积 $K$ 步后更新参数；

那么 Effective Batch Size 为：

$$
B_{\text{effective}}
=
B\times N\times K.
$$

Gradient Accumulation 可以在较少显存下模拟大 Batch 的梯度更新，但需要注意：

- 梯度噪声主要由 Effective Batch Size 决定；
- BatchNorm 统计量通常只根据当前 Micro-batch 计算；
- 因此，Gradient Accumulation 不能让 BatchNorm 获得更大的统计 batch。

这也是小 Micro-batch 场景中 LayerNorm 和 GroupNorm 更稳定的原因之一。

### SGD、Momentum、Adam 和 AdamW

Batch Size 决定“使用多少样本估计梯度”，Optimizer 决定“如何使用这个梯度更新参数”。它们是两个不同的选择。

| Optimizer      | 核心特点                     | 优点                             | 缺点                                | 常见场景                       |
| -------------- | ---------------------------- | -------------------------------- | ----------------------------------- | ------------------------------ |
| SGD            | 直接沿梯度方向更新           | 简单、内存开销小                 | 对 Learning Rate 敏感，收敛可能较慢 | 简单模型                       |
| SGD + Momentum | 累积过去的更新方向           | 减少震荡、加快收敛               | 通常需要仔细调整 Learning Rate      | CNN、视觉任务                  |
| Adam           | 为不同参数调整更新大小       | 前期收敛快，适合稀疏或不均匀梯度 | Weight Decay 处理不够直接           | NLP、复杂模型                  |
| AdamW          | Adam 加上独立的 Weight Decay | 收敛快，正则化更容易控制         | 内存开销通常高于 SGD                | Transformer、LLM、现代深度网络 |

Momentum 可以理解为给梯度更新加入“惯性”：

- 如果多个步骤的梯度方向一致，就加速前进；
- 如果梯度方向反复变化，就减少来回震荡。

Adam 除了使用 Momentum，还会根据每个参数过去的梯度大小，分别调整其更新步长。

AdamW 则进一步将 Weight Decay 从梯度计算中分离，使它更符合“每次更新时直接缩小权重”的定义。

### 应该如何选择？

没有一个 Optimizer 在所有任务上都最好，可以按照以下经验开始：

- Transformer、LLM 和大多数 NLP 任务：通常从 AdamW 开始。
- CNN 或传统视觉任务：可以比较 SGD + Momentum 和 AdamW。
- 梯度非常稀疏或不同参数尺度差异较大：Adam/AdamW 通常更容易训练。
- 追求更高泛化性能时：应该在相同计算预算下实际比较，而不能默认 SGD 一定优于 AdamW。

选择 Batch Size 和 Optimizer 后，还需要一起调整：

- Learning Rate；
- Learning Rate Schedule；
- Warmup；
- Weight Decay；
- 训练步数。

这些参数共同决定模型能否收敛以及最终的泛化能力，不能完全独立调节。

---

## 5. 反向传播

神经网络的一次训练包含四个步骤：

1. **Forward Pass**：根据输入和当前参数计算预测。
2. **Loss**：比较预测和真实标签。
3. **Backpropagation**：计算每个参数对 Loss 的影响。
4. **Optimizer Step**：使用梯度更新参数。

因此，反向传播负责回答：

> 如果某个参数发生一点变化，最终的 Loss 会变化多少？

### 30 秒回答

反向传播是链式法则在计算图上的高效实现。

假设：

$$
z=f(x),\qquad y=g(z),\qquad \mathcal L=h(y),
$$

那么：

$$
\frac{\partial\mathcal L}{\partial x}
=
\frac{\partial\mathcal L}{\partial y}
\frac{\partial y}{\partial z}
\frac{\partial z}{\partial x}.
$$

前向传播按照 $x\rightarrow z\rightarrow y\rightarrow\mathcal L$ 计算结果；反向传播则从 Loss 开始，沿相反方向计算梯度。

### 反向传播为什么高效？

如果分别计算每个参数对 Loss 的影响，会重复进行大量相同计算。

反向传播会先计算后面节点的梯度，然后将结果重复利用。例如，已经得到：

$$
\frac{\partial\mathcal L}{\partial y},
$$

就可以继续计算：

$$
\frac{\partial\mathcal L}{\partial z}
=
\frac{\partial\mathcal L}{\partial y}
\frac{\partial y}{\partial z}.
$$

现代自动微分框架通常不会显式构造完整的 Jacobian，而是直接计算当前梯度与局部 Jacobian 的乘积，即 Vector-Jacobian Product。

这样可以用接近一次前向传播的计算量，得到所有参数的梯度。

### 一条变量经过多条路径时怎么办？

如果变量 $x$ 通过两条路径影响 Loss：

$$
x\rightarrow a\rightarrow\mathcal L,
\qquad
x\rightarrow b\rightarrow\mathcal L,
$$

那么两条路径的贡献需要相加：

$$
\frac{\partial\mathcal L}{\partial x}
=
\frac{\partial\mathcal L}{\partial a}
\frac{\partial a}{\partial x}
+
\frac{\partial\mathcal L}{\partial b}
\frac{\partial b}{\partial x}.
$$

这也是残差连接和多分支网络在反向传播时梯度会自动累加的原因。

### 反向传播与梯度下降的区别

- **Backpropagation**：计算梯度。
- **Optimizer**：使用梯度更新参数。
- **Learning Rate**：控制每次更新的大小。

以 SGD 为例：

$$
w
\leftarrow
w-\eta\frac{\partial\mathcal L}{\partial w}.
$$

其中，$\partial\mathcal L/\partial w$ 由反向传播得到，SGD 负责使用它更新 $w$。

因此：

> Backpropagation 告诉模型应该往哪个方向改，Optimizer 决定具体怎么改。

### 常见工程问题

- **忘记清零梯度**：很多框架默认累积梯度，多次 Backward 前没有清零会导致意外累加。
- **错误使用 `detach`**：`detach` 会切断计算图，使前面的参数无法收到梯度。
- **原地操作**：直接修改反向传播需要的中间结果，可能破坏计算图。
- **数值溢出或下溢**：混合精度训练中，梯度可能变成 `inf`、`NaN` 或 0。
- **广播错误**：Tensor Shape 可以成功广播，但计算的含义可能不正确。
- **未进入训练模式**：Dropout 和 BatchNorm 在 Train Mode 与 Evaluation Mode 下行为不同。

### 面试回答

> 反向传播是链式法则在计算图上的高效实现。它从 Loss 开始，按照计算图的反方向传播梯度，并将一条变量经过不同路径产生的梯度相加。反向传播只负责计算梯度，SGD、AdamW 等 Optimizer 才负责使用梯度更新参数。

---

## 6. 梯度消失与梯度爆炸

梯度消失和梯度爆炸都来自反向传播中的连续乘法。

对于一个深层网络，前面层的梯度可以写成多个局部导数的乘积：

$$
\frac{\partial\mathcal L}{\partial h_l}
=
\frac{\partial\mathcal L}{\partial h_L}
\prod_{k=l}^{L-1}
\frac{\partial h_{k+1}}{\partial h_k}.
$$

如果这些导数大多小于 1，连乘后梯度会越来越小；如果大多大于 1，梯度就可能快速增大。

### 梯度消失

梯度消失表示靠近输入的层收到的梯度非常小，参数几乎无法更新。

常见表现：

- 前面层的 Gradient Norm 接近 0；
- 后面层还在学习，但前面层基本不变；
- Train Loss 很早进入 Plateau；
- 增加训练时间也没有明显改善。

常见原因：

- 网络过深；
- Sigmoid 或 Tanh 长期进入饱和区；
- RNN 处理很长的序列；
- 参数初始化不合适；
- 多层变换的局部导数持续小于 1。

常见处理：

- 使用 ReLU、GELU、SiLU 等激活函数；
- 使用 Residual Connection；
- 使用合适的参数初始化，如 Xavier 或 He Initialization；
- 使用 LayerNorm 或 BatchNorm 稳定激活值；
- RNN 中使用 LSTM、GRU 或 Attention；
- 增加中间监督，缩短 Loss 到前面层的传播路径。

### 为什么残差连接有帮助？

普通网络为：

$$
h_{l+1}=F(h_l).
$$

残差网络为：

$$
h_{l+1}=h_l+F(h_l).
$$

反向传播时：

$$
\frac{\partial h_{l+1}}{\partial h_l}
=
I+\frac{\partial F(h_l)}{\partial h_l}.
$$

其中的 $I$ 提供了一条直接传播梯度的路径。即使 $F$ 内部的梯度比较小，梯度仍然可以通过残差连接向前传播。

这也是深层 ResNet 和 Transformer 普遍使用 Residual Connection 的重要原因。

### 梯度爆炸

梯度爆炸表示梯度在反向传播过程中快速增大，导致参数更新过大。

常见表现：

- Gradient Norm 突然变得非常大；
- Loss 剧烈震荡或突然升高；
- 参数或 Loss 出现 `inf`、`NaN`；
- 混合精度训练频繁出现 Overflow。

常见原因：

- Learning Rate 过大；
- 参数初始化尺度过大；
- RNN 长序列中的重复矩阵乘法；
- 输入或标签中存在异常值；
- Loss 计算数值不稳定。

### 如何处理梯度爆炸？

最常用的方法是 Gradient Clipping。按 Global Norm 裁剪时：

$$
g
\leftarrow
g\cdot
\min\left(
1,\frac{\tau}{\|g\|_2}
\right),
$$

其中：

- $g$ 是所有参数的梯度；
- $\|g\|_2$ 是整体 Gradient Norm；
- $\tau$ 是允许的最大 Gradient Norm。

如果梯度没有超过 $\tau$，它保持不变；如果超过，就按比例缩小。

其他处理方法包括：

- 降低 Learning Rate；
- 使用合适的参数初始化；
- 使用 LayerNorm 或 BatchNorm；
- 检查异常输入、标签和 Loss；
- 混合精度训练中使用 Loss Scaling。

需要注意：

> Gradient Clipping 可以限制过大的梯度，但不能恢复已经消失的梯度。

### 面试回答

> 梯度消失和梯度爆炸都来自反向传播中的多层导数连乘。导数长期小于 1 时梯度会消失，长期大于 1 时梯度会爆炸。梯度消失通常通过非饱和激活、残差连接、归一化和合理初始化处理；梯度爆炸通常通过 Gradient Clipping、降低学习率和稳定数值范围处理。

---

## 7. 激活函数

神经网络的一层通常写为：

$$
h=\phi(Wx+b),
$$

其中，$Wx+b$ 是线性变换，$\phi$ 是激活函数。

### 为什么需要非线性激活？

如果没有激活函数，多层线性变换仍然可以合并成一个线性变换：

$$
W_2(W_1x+b_1)+b_2
=
W'x+b'.
$$

无论叠加多少层，模型仍然只能学习线性关系。

激活函数引入非线性，使多层网络能够学习曲线、复杂决策边界和层次化特征。

### 常见激活函数

| 激活函数   | 输出范围           | 优点                         | 主要问题                     | 常见位置               |
| ---------- | ------------------ | ---------------------------- | ---------------------------- | ---------------------- |
| Sigmoid    | $(0,1)$            | 可以表示概率                 | 两端饱和、梯度消失、非零中心 | 二分类输出层、门控结构 |
| Tanh       | $(-1,1)$           | 零中心                       | 两端仍会饱和                 | 传统 RNN、门控结构     |
| ReLU       | $[0,\infty)$       | 简单、计算快、正区间梯度稳定 | 可能出现 Dying ReLU          | CNN、MLP               |
| Leaky ReLU | $(-\infty,\infty)$ | 负区间仍保留小梯度           | 负区间斜率需要设置           | CNN、MLP               |
| GELU       | $(-\infty,\infty)$ | 平滑地控制输入通过程度       | 计算比 ReLU 复杂             | BERT、部分 Transformer |
| SiLU       | $(-\infty,\infty)$ | 平滑且负区间保留梯度         | 计算比 ReLU 复杂             | 现代视觉与语言模型     |

### Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

Sigmoid 将输入映射到 0 到 1，适合表示二分类概率。

它的导数为：

$$
\sigma'(x)=\sigma(x)(1-\sigma(x)).
$$

当 $x$ 很大或很小时，导数接近 0。因此，在深层网络中重复使用 Sigmoid 容易造成梯度消失。

### Tanh

$$
\tanh(x)
=
\frac{e^x-e^{-x}}{e^x+e^{-x}}.
$$

Tanh 的输出范围是 $-1$ 到 $1$，并且以 0 为中心。相比 Sigmoid，它通常更适合表示隐藏状态，但输入绝对值较大时仍然会进入饱和区。

### ReLU

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

当 $x>0$ 时，ReLU 的梯度为 1，因此可以缓解正区间中的梯度消失。

当 $x<0$ 时，输出和梯度都为 0。这可能导致某些神经元长期无法更新，即 Dying ReLU。

### GELU、SiLU 与 SwiGLU

GELU：

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。它会根据输入大小平滑地保留或减弱输入。

SiLU：

$$
\operatorname{SiLU}(x)=x\sigma(x).
$$

SiLU 在负区间仍保留一定梯度，并且整体更加平滑。

SwiGLU 不只是一个单独的激活函数，而是一种带门控的 FFN 结构，可以简化表示为：

$$
\operatorname{SwiGLU}(x)
=
\operatorname{SiLU}(xW_g)\odot(xW_v).
$$

其中一个分支生成内容，另一个分支决定多少内容可以通过。许多现代 Transformer 使用 SwiGLU 作为 FFN 的核心结构。

不能简单说某个激活函数在所有任务上都最好。实际效果还取决于模型结构、参数量和计算预算。

### Dying ReLU 是什么？

如果某个 ReLU 神经元的输入长期小于 0：

$$
Wx+b<0,
$$

它的输出和梯度都会一直为 0，参数无法继续更新。

常见原因：

- Learning Rate 太大，一次更新将神经元推入负区间；
- 参数初始化不合适；
- 输入分布发生明显偏移。

常见处理：

- 降低 Learning Rate；
- 使用 He Initialization；
- 使用 Leaky ReLU、GELU 或 SiLU；
- 使用合适的 Normalization；
- 使用 Residual Connection。

### 输出层如何选择激活函数？

- **回归任务**：通常使用线性输出，不加激活。
- **二分类**：通常使用一个 Logit，并通过 Sigmoid 得到概率。
- **多分类且类别互斥**：使用 Softmax。
- **多标签分类**：每个标签独立使用 Sigmoid。

### 面试回答

> 激活函数为神经网络引入非线性。如果没有激活函数，多层线性变换仍然等价于一层线性模型。Sigmoid 和 Tanh 可能在饱和区造成梯度消失；ReLU 简单高效，但可能出现 Dying ReLU；GELU、SiLU 和 SwiGLU 更平滑，因此常用于现代 Transformer。

---

## 8. Softmax 及其反向传播

神经网络的最后一层通常输出一组没有范围限制的数值，称为 Logits：

$$
z=[z_1,z_2,\ldots,z_C].
$$

Softmax 将这些 Logits 转换为一个概率分布：

$$
s_i
=
\frac{e^{z_i}}
{\sum_j e^{z_j}},
\qquad
\sum_i s_i=1.
$$

因此，Softmax 适合“多个类别中只能选择一个”的多分类任务。

### Softmax 与 Sigmoid 的区别

- **Softmax**：所有类别相互竞争，概率之和为 1，适合单标签多分类。
- **Sigmoid**：每个类别独立判断，适合二分类或多标签分类。

例如，一张图片只能属于猫、狗、鸟中的一个类别时使用 Softmax；一篇文章可以同时属于 AI、Security 和 Software Engineering 时，使用多个 Sigmoid。

### 为什么要减去最大 Logit？

直接计算 $e^{z_i}$ 时，如果 $z_i$ 很大，可能发生数值溢出。

稳定实现为：

$$
s_i
=
\frac{e^{z_i-\max(z)}}
{\sum_j e^{z_j-\max(z)}}.
$$

Softmax 对所有 Logits 同时加减同一个常数不会改变结果：

$$
\operatorname{softmax}(z)
=
\operatorname{softmax}(z+c).
$$

因此，减去最大值不会改变概率，但可以保证最大的指数项为 $e^0=1$，避免 Overflow。

### Softmax 的 Jacobian

Softmax 中每个输出都依赖所有 Logits：

$$
\frac{\partial s_i}{\partial z_j}
=
s_i(\delta_{ij}-s_j).
$$

其中：

- 当 $i=j$ 时：

$$
\frac{\partial s_i}{\partial z_i}
=
s_i(1-s_i);
$$

- 当 $i\neq j$ 时：

$$
\frac{\partial s_i}{\partial z_j}
=
-s_is_j.
$$

矩阵形式为：

$$
J
=
\operatorname{diag}(s)-ss^\top.
$$

### Softmax 与 Cross-Entropy

对于 One-Hot 标签 $y$，Cross-Entropy Loss 为：

$$
\mathcal L
=
-\sum_i y_i\log s_i.
$$

将 Softmax 和 Cross-Entropy 一起求导后，可以化简为：

$$
\frac{\partial\mathcal L}{\partial z}
=
s-y.
$$

这个结果非常直观：

- 对正确类别，梯度为 $s_i-1$，会推动其 Logit 增大；
- 对错误类别，梯度为 $s_i$，会推动其 Logit 减小。

实际实现中不需要显式构造 $C\times C$ 的 Jacobian。深度学习框架通常会将 LogSoftmax 和 Cross-Entropy 合并计算，以提高效率和数值稳定性。

### 面试回答

> Softmax 将一组 Logits 转换为和为 1 的概率分布，适合类别互斥的多分类任务。数值稳定实现会先减去最大 Logit。Softmax 的 Jacobian 是 $\operatorname{diag}(s)-ss^\top$，与 Cross-Entropy 结合后，对 Logits 的梯度可以直接化简为 $s-y$。

---

## 9. Dropout

L1、L2 通过限制参数减少 Overfitting；Dropout 则通过在训练过程中随机删除部分激活值，降低模型对固定特征组合的依赖。

因此，Dropout 是一种带随机性的正则化方法。

### 30 秒回答

训练时，Dropout 随机将一部分神经元输出设为 0：

$$
\tilde h_i
=
\frac{m_i h_i}{1-p},
\qquad
m_i\sim\operatorname{Bernoulli}(1-p),
$$

其中：

- $p$ 是 Dropout Rate，即删除概率；
- $m_i=0$ 表示该激活被删除；
- $m_i=1$ 表示该激活被保留。

推理时关闭 Dropout，所有神经元都正常工作。

### Dropout 为什么能够减少 Overfitting？

如果没有 Dropout，模型可能形成一条固定的预测路径，并过度依赖少数神经元。

加入 Dropout 后，任何神经元都可能在当前 Forward Pass 中被删除，因此模型必须让不同特征共同承担预测任务。

它主要产生三种效果：

- **减少固定依赖**：模型不能只依赖少数神经元。
- **注入训练噪声**：每次 Forward Pass 使用不同的激活组合。
- **近似模型集成**：训练过程可以看作在共享参数的多个子网络上训练。

从 Bias-Variance 的角度看，Dropout 通常通过增加训练难度来降低 Variance，但 Dropout Rate 过大也会增加 Bias，导致 Underfitting。

### 为什么训练时要除以 $1-p$？

Dropout 后，只剩下大约 $1-p$ 比例的激活值。如果不进行缩放，整层输出的平均大小会下降。

使用 Inverted Dropout：

$$
\tilde h_i
=
\frac{m_i h_i}{1-p},
$$

其期望为：

$$
\mathbb E[\tilde h_i]
=
\frac{\mathbb E[m_i]h_i}{1-p}
=
h_i.
$$

因此，训练时和推理时的输出期望保持一致。

现代框架通常使用 Inverted Dropout：

- 训练时：随机置 0，并将保留值除以 $1-p$；
- 推理时：直接关闭 Dropout，不需要额外缩放。

### Dropout 应该什么时候使用？

Dropout 更适合以下情况：

- Train Loss 很低，但 Validation Loss 明显更高；
- 模型较大，但训练数据有限；
- 增加数据比较困难；
- 全连接层或 Transformer 出现明显 Overfitting。

如果模型已经 Underfitting，通常不应该继续增加 Dropout。

现代大模型拥有大量训练数据，并且还会使用 Weight Decay、数据清洗和其他正则化，因此不一定需要很高的 Dropout Rate。

### Dropout 与 BatchNorm

Dropout 会随机改变激活值分布，而 BatchNorm 需要根据当前 batch 估计激活值的均值和方差。

如果 Dropout 放在 BatchNorm 前面，它产生的随机变化可能影响 BatchNorm 的统计量。因此，两者同时使用时需要注意顺序和实际验证结果。

常见做法是先进行 BatchNorm，再使用激活函数和 Dropout，但具体顺序仍取决于网络结构。

### 常见工程问题

- 推理前没有切换到 Evaluation Mode，导致输出仍然随机；
- Dropout Rate 太大，模型出现 Underfitting；
- 验证时错误地保持 Dropout 开启；
- 误以为 Dropout 会减少参数量或推理成本。

Dropout 只是暂时屏蔽激活值，不会真正删除模型参数，因此通常不会减少模型大小或常规推理成本。

### 面试回答

> Dropout 是一种随机正则化方法。训练时，它随机将部分激活值设为 0，减少模型对少数神经元和固定特征组合的依赖；推理时关闭 Dropout。现代框架使用 Inverted Dropout，在训练时将保留的激活除以 $1-p$，使训练和推理阶段的输出期望保持一致。

---

## 10. Perceptron

Perceptron 可以看作最简单的人工神经元：

1. 对输入进行线性加权；
2. 加上 Bias；
3. 使用阶跃函数输出分类结果。

它是理解线性分类器和神经网络基本结构的起点。

### 前向计算

对于输入 $x$，Perceptron 首先计算：

$$
z=w^\top x+b.
$$

然后根据 $z$ 的正负进行分类：

$$
\hat y
=
\operatorname{sign}(w^\top x+b).
$$

它的决策边界为：

$$
w^\top x+b=0,
$$

这是一条直线、一个平面或更高维空间中的超平面。因此，Perceptron 本质上是线性分类器。

### Perceptron 如何学习？

假设标签为：

$$
y\in\{-1,+1\}.
$$

当样本被正确分类时：

$$
y(w^\top x+b)>0,
$$

不需要更新参数。

当样本被错误分类时：

$$
y(w^\top x+b)\leq 0,
$$

按照标签方向更新：

$$
w
\leftarrow
w+\eta yx,
$$

$$
b
\leftarrow
b+\eta y.
$$

这个更新会让当前样本下一次更可能被正确分类。

如果训练数据线性可分，Perceptron Learning Algorithm 可以在有限次更新后找到一个能够正确分类训练数据的超平面。如果数据不是线性可分的，它可能持续更新而无法收敛。

### Perceptron 与神经网络有什么关系？

Perceptron 已经包含神经网络单元的基本结构：

$$
\text{output}
=
\text{activation}(w^\top x+b).
$$

但传统 Perceptron 使用不可微的阶跃函数，因此不能直接使用现代反向传播训练多层网络。

现代神经网络将阶跃函数替换为 ReLU、Sigmoid、GELU 等可训练的激活函数，并将多个神经元和多层结构组合起来。

### Perceptron、Logistic Regression 与神经网络

| 模型                | 变换                     | 输出       | 决策边界     |
| ------------------- | ------------------------ | ---------- | ------------ |
| Perceptron          | $w^\top x+b$             | 硬分类结果 | 线性         |
| Logistic Regression | $\sigma(w^\top x+b)$     | 分类概率   | 线性         |
| 多层神经网络        | 多层线性变换和非线性激活 | 概率或数值 | 可以是非线性 |

Logistic Regression 虽然使用了 Sigmoid，但其决策边界仍然由 $w^\top x+b=0$ 决定，因此仍然是线性分类器。

多层神经网络通过隐藏层学习新的特征表示，才能形成复杂的非线性决策边界。

### Perceptron 为什么不能解决 XOR？

XOR 的输入与输出为：

| $x_1$ | $x_2$ |  $y$ |
| ----: | ----: | ---: |
|     0 |     0 |    0 |
|     0 |     1 |    1 |
|     1 |     0 |    1 |
|     1 |     1 |    0 |

正类位于两个对角位置，负类位于另外两个对角位置。无法用一条直线将两类完全分开，因此 XOR 不是线性可分问题。

单层 Perceptron 只能学习线性决策边界，所以不能解决 XOR。

解决方法包括：

- 人工加入非线性特征，例如 $x_1x_2$；
- 使用 Kernel Method；
- 使用带隐藏层和非线性激活函数的多层神经网络。

### 面试回答

> Perceptron 是一个线性二分类模型，先计算 $w^\top x+b$，再通过阶跃函数输出类别。样本被错分时，它按照标签方向更新权重。如果数据线性可分，Perceptron 可以收敛；如果数据不是线性可分的，它可能无法收敛。XOR 不是线性可分问题，所以单层 Perceptron 无法解决，需要非线性特征或多层神经网络。

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