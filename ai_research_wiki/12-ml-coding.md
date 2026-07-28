# 03. ML Coding

[返回目录](README.md)

本页整理常见手写模型，统一用**简洁的函数式写法**——先看懂十几行的核心逻辑，再谈工程封装。面试时不只要写出能跑的代码，还要主动说明：

- 输入输出 shape。
- 时间/空间复杂度。
- 数值稳定性。
- 边界条件。
- 和框架实现的差异。

> **面试策略**：先用十几行核心函数证明你懂原理、NumPy 向量化扎实；面试官若要求，再补面向对象的 `fit`/`predict` 封装、空簇处理、`inertia_` 误差记录等工程细节（“先写短的，再聊长的”）。

---

## 1. KMeans

原理就两步循环：**分配**（每个点归到最近的中心）+ **更新**（每个簇的均值当新中心），直到中心不再移动。

### 核心实现

```python
import numpy as np


def kmeans(X, k, max_iter=100, tol=1e-4):
    X = np.asarray(X, dtype=float)
    # 1. 随机选 k 个点当初始中心
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iter):
        # 2. 算每个点到每个中心的平方距离，找最近的中心
        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. 更新中心 = 每个簇内点的均值（空簇则保留旧中心，避免 NaN）
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        # 4. 中心几乎不动就收敛（浮点比较用 tol，别用 ==）
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return labels, centroids
```

### Example

```python
np.random.seed(42)
X = np.random.rand(100, 2)

labels, centroids = kmeans(X, k=3)
print(centroids)
```

### 广播机制拆解（算距离那行）

`distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)` 是最容易写晕、也最常考的一行。它一次性算出「$n$ 个点到 $k$ 个中心」的距离矩阵，靠的是 NumPy 广播：

- `X` 形状 $(n, d)$，`X[:, None, :]` 插入一个空维 → $(n, 1, d)$。
- `centroids` 形状 $(k, d)$，`centroids[None, :, :]` → $(1, k, d)$。
- 相减时把大小为 1 的维广播拉伸对齐 → $(n, k, d)$，物理意义是「每个点分别减去每个中心」的坐标差。
- `** 2` 后 `.sum(axis=2)` 沿最后一维（坐标维 $d$）求和，得到 $(n, k)$ 的平方距离矩阵。
- 再 `np.argmin(distances, axis=1)`：`axis=1` 是第二维（$k$ 个中心；NumPy 从 0 数轴），对每个点挑距离最小的中心 → 长度 $n$ 的 label。

这样避免了两层 Python for 循环，是向量化标准写法（把 `np.` 换成 `torch.` 几乎就是 PyTorch）。

### K-Means++ 初始化

纯随机初始化很看运气：若初始中心挤在一起，收敛慢、易陷入差的局部最优。K-Means++ 只改**初始化**（后面 assign/update 不变），让初始中心尽量**相互远离**：

1. 从数据里随机选 1 个点作为第一个中心。
2. 对每个点 $x$，算它到**已选中心里最近那个**的平方距离 $D(x)^2$。
3. 以正比于 $D(x)^2$ 的概率抽下一个中心（离已有中心越远越可能被选中）。
4. 重复 2–3 直到选够 $k$ 个。

```python
def kmeans_pp_init(X, k, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = X.shape[0]
    centroids = [X[rng.integers(n)]]                     # 1. 随机第一个中心
    for _ in range(1, k):
        # 每个点到「最近已选中心」的平方距离 D(x)^2
        d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centroids], axis=0)
        probs = d2 / d2.sum()                            # 2-3. 正比于 D(x)^2 的概率
        centroids.append(X[rng.choice(n, p=probs)])
    return np.array(centroids, dtype=float)
```

通常比纯随机收敛更快、结果更稳，是 sklearn `KMeans` 的默认初始化。

### 关键追问

- **为什么不用 `sqrt`？** 最近 centroid 的 argmin 不受平方根影响，平方距离更省。
- **收敛到全局最优吗？** 不保证，只保证目标函数单调不增并收敛到局部最优或稳定点。
- **空簇怎么办？** 保留旧 centroid（上面的写法）、随机重置、或重置到当前误差最大的点。
- **复杂度？** 每轮 $O(nkd)$，其中 $n$ 是样本数，$k$ 是簇数，$d$ 是维度。
- **实际优化？** k-means++ 初始化、多次随机重启、标准化特征。

---

## 2. Logistic Regression

名字叫回归，其实是**二分类**：线性打分 → sigmoid 压成概率 → 二元交叉熵 + 梯度下降。

### 核心实现

```python
import numpy as np


def sigmoid(z):
    # 数值稳定：z>=0 与 z<0 分开算，避免 exp 溢出
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def logistic_regression(X, y, lr=0.1, max_iter=1000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, d = X.shape
    w, b = np.zeros(d), 0.0

    for _ in range(max_iter):
        p = sigmoid(X @ w + b)          # 前向：预测概率
        # 梯度（对 logits 求导后非常干净：预测 - 真实）
        dw = X.T @ (p - y) / n
        db = (p - y).mean()
        w -= lr * dw                    # 梯度下降更新
        b -= lr * db

    return w, b
```

### Example

```python
np.random.seed(42)
X = np.random.randn(100, 2)
y = (sigmoid(X @ np.array([1.0, -2.0]) + 0.2) > 0.5).astype(int)

w, b = logistic_regression(X, y)
pred = (sigmoid(X @ w + b) >= 0.5).astype(int)
print("accuracy:", (pred == y).mean())
```

### 关键追问

- **梯度为什么这么干净？** 对 logits 求导后正好是 $\hat y-y$，参数梯度是 $X^\top(\hat y-y)/n$。
- **MSE 做 Logistic 是凸的吗？** 一般不是；BCE + 线性 logits 才是凸的。
- **算 loss 时为什么别直接 `np.log(y_hat)`？** 概率接近 0/1 会 `log(0)`；要么用 logits 形式的稳定 BCE，要么 `np.clip`。
- **要加正则时截距要不要罚？** 通常**不**对截距 $b$ 做 L2，它只控制整体基准概率。
- **阈值一定 0.5 吗？** 不一定，由业务成本、Precision/Recall 和校准决定；类别不平衡时 accuracy 也不可靠。

---

## 3. Multiple Linear Regression

**多元** = 多个特征（不是多项式）。目标是让预测与真实的均方误差 MSE 最小，有一步到位的闭式解。

### 核心实现

```python
import numpy as np


def linear_regression(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # 左边拼一列 1 当截距项：b 变成权重向量的第 0 项
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    # lstsq 基于 SVD，稳定；不要显式求逆
    theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return theta                        # theta[0] 是截距 b，其余是各特征权重


def linreg_predict(X, theta):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return Xb @ theta
```

### Example

```python
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=float)
y = np.array([5, 7, 9, 11, 13], dtype=float)

theta = linear_regression(X, y)
print("theta:", theta)
print("pred:", linreg_predict(np.array([[6, 7]], dtype=float), theta))
```

### 为什么不要显式求逆？

原始闭式解（对 MSE 求导令其为 0 推得）是：

$$
\theta=(X^\top X)^{-1}X^\top y
$$

但显式算 `np.linalg.inv(X.T @ X)` 数值不稳定，且当 $X^\top X$ 奇异或病态（特征共线、或特征数 > 样本数）时会失败。更好的做法：

- `np.linalg.lstsq`：基于 SVD 的最小二乘，稳定。
- `np.linalg.pinv`：伪逆。
- Ridge：共线性强时加 L2 正则，让矩阵恒可逆。

### Ridge 版本

在对角线加 $\alpha I$，让 $(X^\top X+\alpha I)$ 恒可逆，同时把权重压小、抑制过拟合。

```python
def ridge_regression(X, y, alpha=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])

    reg = alpha * np.eye(Xb.shape[1])
    reg[0, 0] = 0.0                     # 不对截距做正则
    # 用 solve 解 (X^T X + alpha I) theta = X^T y，比显式求逆稳
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
```

---

## 3.1 多项式回归

多项式回归**不是新算法**，只是「多元线性回归 + 特征扩展」：把单特征 $x$ 展开成 $[1, x, x^2, \dots, x^{\text{degree}}]$ 当作新特征，再套用上面的线性回归。

```python
def poly_features(x, degree):
    # x: (N,) -> (N, degree+1): [1, x, x^2, ..., x^degree]
    return np.vander(x, degree + 1, increasing=True)

# 用法：Xp = poly_features(x, 3); theta, *_ = np.linalg.lstsq(Xp, y, rcond=None)
```

面试常问的坑：

- **过拟合**：degree 越高越容易剧烈震荡去穿过每个训练点（Runge 现象），泛化差 → 用 Ridge/Lasso 正则压制高次权重。
- **维度爆炸**：多特征做高次展开会产生大量交叉项（$x_1x_2$、$x_1^2x_2$ …），特征数急剧膨胀。
- 面试极少让从零手写，多作为概念题；关键是能说清「它本质就是特征工程 + 线性回归」。

---

## 4. Softmax 与 Cross-Entropy

```python
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)   # 减最大值防溢出
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def cross_entropy_from_logits(logits, y):
    """
    logits: shape (batch, num_classes)
    y: integer labels, shape (batch,)
    """
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)

    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))  # log-sum-exp
    return -log_probs[np.arange(logits.shape[0]), y].mean()
```

### 两个必说的点

- **Softmax 为什么减最大值？** 直接 $e^{z}$ 在 $z$ 很大时会溢出（`inf`）。减去每行最大值 $z-\max z$ 在数学上**完全等价**（分子分母同乘 $e^{-\max z}$），但把指数压回安全范围。
- **交叉熵用 logits 还是概率？** 上面直接吃 **logits**（用 log-sum-exp 把 softmax 和 log 合并），比「先 softmax 出概率、再取 `log`」更稳、更快，这也是 PyTorch `CrossEntropyLoss` 的做法。若手上只有概率，要 `np.clip(p, 1e-15, 1)` 防 `log(0)`。合并后梯度也很干净：**softmax 概率 $-$ one-hot 标签**。

---

## 5. Scaled Dot-Product Attention

三板斧：$Q$ 和 $K$ 算相似度 → 缩放 + softmax 成权重 → 加权求和 $V$，即 $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$。

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, q_len, d_k)   K: (batch, kv_len, d_k)   V: (batch, kv_len, d_v)
    mask: 可广播到 (batch, q_len, kv_len)，True 表示「可见/允许」。
    返回 output (batch, q_len, d_v) 和 weights (batch, q_len, kv_len)。
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)   # 缩放防 softmax 饱和

    if mask is not None:
        scores = np.where(mask, scores, -1e9)            # 不可见处 -> -inf -> 权重 ~0

    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights
```

### Causal Mask（防止偷看未来）

Decoder 生成时第 $t$ 步只能看 $\le t$ 的位置，否则等于抄答案。用下三角矩阵把「未来」位置的分数打成 $-\infty$。

```python
def causal_mask(q_len, kv_len=None):
    if kv_len is None:
        kv_len = q_len
    # (q_len, kv_len)，下三角为 True（可见），上三角为 False（未来，屏蔽）
    return np.tril(np.ones((q_len, kv_len), dtype=bool))


batch, seq_len, d_model = 2, 4, 8
Q = np.random.randn(batch, seq_len, d_model)
K = np.random.randn(batch, seq_len, d_model)
V = np.random.randn(batch, seq_len, d_model)

mask = causal_mask(seq_len)[None, :, :]
out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
print(out.shape, weights.shape)   # (2, 4, 8) (2, 4, 4)
```

### 关键追问

- **为什么除以 $\sqrt{d_k}$？** 点积是 $d_k$ 项之和，方差随维度增大到约 $d_k$；不缩放会让 softmax 饱和（赢者通吃）、梯度趋 0。除以 $\sqrt{d_k}$ 把方差拉回常数量级。
- **Mask 用 True 还是 False 表示可见？** 面试时必须说清楚；上面用 True 表示可见。
- **为什么用 `-1e9`？** 让 masked 位置的 softmax 权重近似 0；真实框架常用 dtype 的最小值。
- **复杂度？** 标准 attention 的注意力矩阵计算与显存都是 $O(n^2)$。

---

## 6. Multi-Head Attention

多头 = 把特征切成 $h$ 份，让 $h$ 个「专家」从不同视角各算一遍 attention，再拼回来过 $W_o$。`split_heads` 的本质就是 reshape：$(B, L, d_{\text{model}}) \to (B, h, L, d_{\text{model}}/h)$。

```python
import numpy as np


def split_heads(x, num_heads):
    # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_head)
    batch, seq_len, d_model = x.shape
    assert d_model % num_heads == 0
    d_head = d_model // num_heads
    x = x.reshape(batch, seq_len, num_heads, d_head)
    return np.transpose(x, (0, 2, 1, 3))


def combine_heads(x):
    # (batch, num_heads, seq_len, d_head) -> (batch, seq_len, d_model)
    batch, num_heads, seq_len, d_head = x.shape
    x = np.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq_len, num_heads * d_head)


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    # X: (batch, seq_len, d_model); Wq/Wk/Wv/Wo: (d_model, d_model)
    Q = split_heads(X @ Wq, num_heads)
    K = split_heads(X @ Wk, num_heads)
    V = split_heads(X @ Wv, num_heads)

    d_head = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_head)   # 每个头并行算
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)
    context = combine_heads(weights @ V)                    # 拼回去
    return context @ Wo, weights
```

### Example

```python
batch, seq_len, d_model, num_heads = 2, 5, 16, 4
X = np.random.randn(batch, seq_len, d_model)
Wq, Wk, Wv, Wo = [np.random.randn(d_model, d_model) / np.sqrt(d_model) for _ in range(4)]

mask = causal_mask(seq_len)[None, None, :, :]
out, weights = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=mask)
print(out.shape, weights.shape)   # (2, 5, 16) (2, 4, 5, 5)
```

---

## 7. 决策树：信息熵与信息增益

决策树每一步选「问哪个特征、在哪切」的标准是：切完之后子集**最纯**。用信息熵衡量混乱度，用信息增益选特征。

- **熵（Entropy）**：$H(y)=-\sum_c p_c\log_2 p_c$。全同类 → $H=0$（纯）；各类均匀 → $H$ 最大（乱）。
- **信息增益**：$IG=H(\text{父})-\sum_{\text{子}}\frac{n_{\text{子}}}{n}H(\text{子})$。选 $IG$ 最大的特征分裂。

```python
import numpy as np


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()          # np.unique 保证 p>0，无需 +eps
    return -np.sum(p * np.log2(p))


def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    child = (len(y_left) / n) * entropy(y_left) + (len(y_right) / n) * entropy(y_right)
    return entropy(y_parent) - child
```

面试最常让手写的就是 `entropy`；Gini 不纯度 $1-\sum_c p_c^2$ 是另一常见替代（CART 默认）。

---

## 8. SVM（支持向量机）

面试极少让手写完整 SMO，重点是几何直觉：

- **最大间隔**：在能分开两类的超平面里，选让「离它最近的点（支持向量）到它的间隔最大」的那个——走正中间、两边最空旷的路，泛化更好。
- **核技巧（Kernel Trick）**：线性不可分时，用核函数（如 RBF 高斯核）把数据隐式映射到高维，在高维里线性可分，而无需显式算高维坐标。
- 代码通常直接调 `sklearn.svm.SVC(kernel='rbf', C=1.0)`；`C` 越大越不容忍误分（间隔越硬、越易过拟合）。

---

## 9. 面试时主动指出的坑

### KMeans

- 空簇处理（空簇算均值会得 NaN）。
- 初始化敏感，需要多次重启或 k-means++。
- 特征尺度会影响欧氏距离，通常先标准化。
- 浮点收敛判断用 `tol`，别用 `centroids == new_centroids`。

### Logistic Regression

- Sigmoid 和 log loss 要数值稳定。
- 阈值不一定为 0.5。
- 类别不平衡时 accuracy 不可靠。
- 截距通常不正则化。

### Linear Regression

- 不要显式求逆。
- 共线性导致 $X^\top X$ 病态。
- 训练前检查 shape，尤其 $y$ 是 `(n,)` 还是 `(n, 1)`。

### Attention

- Mask 语义（True/False 谁可见）必须说清楚。
- 注意 softmax 的 axis。
- Q/K/V 的 shape 必须能矩阵乘。
- 标准 attention 有 $O(n^2)$ 注意力矩阵。

---

## 10. 手写与实现题通用检查清单

### 张量维度

每一步写出 shape，特别关注：

- batch 维和 sequence 维的顺序。
- 矩阵乘法的收缩维。
- broadcasting 是否符合语义。
- `view/reshape/transpose` 后内存是否 contiguous。

### 数值稳定性

- Softmax 使用减最大值或 `logsumexp`。
- 概率损失优先接收 logits 的 fused 实现。
- 除法加入有依据的 $\epsilon$，同时测试全零或极小分母。
- 监控 NaN/Inf、梯度范数和混合精度溢出。
- 稀疏更新要处理重复 index 的累加语义，而不只是“索引碰撞”。

### Python / PyTorch 工程陷阱

- 文件名不要遮蔽标准库或第三方包，例如 `torch.py`、`random.py`。
- 变量名不要覆盖导入模块或函数。
- 检查 train/eval mode、device、dtype 和随机种子。
- 不要用能运行的广播掩盖 shape bug。
- 用 `torch.autograd.gradcheck` 对自定义算子做有限差分检查。

---

## 11. 常见手写题清单

应能在不调用高级封装的情况下实现并解释：

- 稳定 Softmax、LogSoftmax、Cross-Entropy。
- 单头/多头 Attention 与 causal mask。
- LayerNorm、RMSNorm。
- 线性回归、Logistic Regression、Ridge。
- K-Means（含 k-means++ 初始化）、PCA 的核心步骤。
- 决策树的熵与信息增益（Entropy / Information Gain）。
- LSTM 单步前向和参数量。
- Precision、Recall、F1、ROC-AUC 的计算。

评价手写代码时不只看结果，还要检查 shape、时间/空间复杂度、边界条件、数值稳定性和梯度正确性。
