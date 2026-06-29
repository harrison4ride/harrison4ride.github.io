# 03. ML Coding

[返回目录](README.md)

本页整理常见手写模型。面试时不只要写出能跑的代码，还要主动说明：

- 输入输出 shape。
- 时间/空间复杂度。
- 数值稳定性。
- 边界条件。
- 和框架实现的差异。

---

## 1. KMeans

### 面试版实现

```python
import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed n_samples")
        indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        return X[indices].astype(float)

    def _assign(self, X, centroids):
        # squared distances: shape (n_samples, n_clusters)
        distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(distances, axis=1), distances

    def _update(self, X, labels, old_centroids):
        new_centroids = np.empty_like(old_centroids)
        for k in range(self.n_clusters):
            points = X[labels == k]
            if len(points) == 0:
                # Empty-cluster fallback: keep the previous centroid.
                # Other choices: reinitialize to farthest point or random point.
                new_centroids[k] = old_centroids[k]
            else:
                new_centroids[k] = points.mean(axis=0)
        return new_centroids

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            labels, distances = self._assign(X, centroids)
            new_centroids = self._update(X, labels, centroids)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        labels, distances = self._assign(X, centroids)
        self.centroids = centroids
        self.inertia_ = distances[np.arange(X.shape[0]), labels].sum()
        return self

    def predict(self, X):
        if self.centroids is None:
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        labels, _ = self._assign(X, self.centroids)
        return labels
```

### Example

```python
np.random.seed(42)
X = np.random.rand(100, 2)

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

print(model.centroids)
print(model.inertia_)
```

### 关键追问

- **为什么不用 `sqrt`？** 最近 centroid 的 argmin 不受平方根影响，平方距离更省。
- **收敛到全局最优吗？** 不保证，只保证目标函数单调不增并收敛到局部最优或稳定点。
- **空簇怎么办？** 保留旧 centroid、随机重置、或重置到当前误差最大的点。
- **复杂度？** 每轮 $O(nkd)$，其中 $n$ 是样本数，$k$ 是簇数，$d$ 是维度。
- **实际优化？** k-means++ 初始化、多次随机重启、标准化特征。

---

## 2. Logistic Regression

### 数值稳定实现

```python
import numpy as np


def sigmoid(z):
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)

    pos = z >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy_with_logits(logits, y):
    # Stable form:
    # max(z, 0) - z*y + log(1 + exp(-abs(z)))
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)
    loss = np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return loss.mean()


class LogisticRegressionGD:
    def __init__(self, lr=0.1, max_iter=1000, l2=0.0, fit_intercept=True):
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.w = None
        self.loss_history = []

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        Xb = self._add_intercept(X)

        n_samples, n_features = Xb.shape
        self.w = np.zeros(n_features)
        self.loss_history = []

        for _ in range(self.max_iter):
            logits = Xb @ self.w
            probs = sigmoid(logits)

            grad = Xb.T @ (probs - y) / n_samples

            if self.l2 > 0:
                reg = self.w.copy()
                if self.fit_intercept:
                    reg[0] = 0.0  # do not regularize intercept
                grad += self.l2 * reg

            self.w -= self.lr * grad

            logits = Xb @ self.w
            loss = binary_cross_entropy_with_logits(logits, y)
            if self.l2 > 0:
                reg_w = self.w[1:] if self.fit_intercept else self.w
                loss += 0.5 * self.l2 * np.dot(reg_w, reg_w)
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._add_intercept(X)
        return sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

### Example

```python
np.random.seed(42)
X = np.random.randn(100, 2)
true_w = np.array([1.0, -2.0])
logits = X @ true_w + 0.2
y = (sigmoid(logits) > 0.5).astype(int)

model = LogisticRegressionGD(lr=0.1, max_iter=1000, l2=1e-3)
model.fit(X, y)

pred = model.predict(X)
print("accuracy:", (pred == y).mean())
print("weights:", model.w)
```

### 关键追问

- **为什么不直接 `np.log(y_hat)`？** 当概率接近 0 或 1 时会出现 `log(0)`，应使用 logits 形式的稳定 BCE。
- **梯度是什么？** 对 logits 的梯度为 $\hat y-y$，所以参数梯度是 $X^\top(\hat y-y)/n$。
- **MSE 做 Logistic Regression 是凸的吗？** 一般不是。BCE + linear logits 是凸的。
- **为什么 intercept 不正则化？** 截距控制整体基准概率，通常不希望被 L2 收缩。
- **生产中阈值一定是 0.5 吗？** 不一定，阈值由业务成本、Precision/Recall 和校准决定。

---

## 3. Multiple Linear Regression

### 推荐实现：`lstsq`

```python
import numpy as np


class LinearRegressionClosedForm:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.theta = None

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = self._add_intercept(X)

        # More stable than explicitly computing inv(X.T @ X).
        self.theta, residuals, rank, singular_values = np.linalg.lstsq(
            Xb, y, rcond=None
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._add_intercept(X)
        return Xb @ self.theta
```

### Example

```python
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
])
y = np.array([5, 7, 9, 11, 13])

model = LinearRegressionClosedForm()
model.fit(X, y)

X_new = np.array([
    [6, 7],
    [7, 8],
])

print("theta:", model.theta)
print("pred:", model.predict(X_new))
```

### 为什么不要显式求逆？

原始公式是：

$$
\theta=(X^\top X)^{-1}X^\top y
$$

但显式计算逆矩阵数值不稳定，且当 $X^\top X$ 奇异或病态时会失败。更好的做法：

- `np.linalg.lstsq`：基于更稳定的分解求最小二乘。
- `np.linalg.pinv`：使用伪逆。
- Ridge：当共线性强时加入 L2 正则。

### Ridge 版本

```python
def ridge_regression(X, y, alpha=1.0, fit_intercept=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if fit_intercept:
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    else:
        Xb = X

    n_features = Xb.shape[1]
    reg = alpha * np.eye(n_features)
    if fit_intercept:
        reg[0, 0] = 0.0

    theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return theta
```

---

## 4. Softmax

### 稳定实现

```python
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)
```

### Cross-Entropy

```python
def cross_entropy_from_logits(logits, y):
    """
    logits: shape (batch, num_classes)
    y: integer labels, shape (batch,)
    """
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)

    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -log_probs[np.arange(logits.shape[0]), y].mean()
```

---

## 5. Scaled Dot-Product Attention

### 单头 Attention

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: shape (batch, q_len, d_k)
    K: shape (batch, kv_len, d_k)
    V: shape (batch, kv_len, d_v)
    mask: optional, broadcastable to (batch, q_len, kv_len).
          Use True for positions that are allowed.

    Returns:
        output: shape (batch, q_len, d_v)
        weights: shape (batch, q_len, kv_len)
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights
```

### Causal Mask

```python
def causal_mask(q_len, kv_len=None):
    if kv_len is None:
        kv_len = q_len
    # shape (q_len, kv_len), True means visible
    return np.tril(np.ones((q_len, kv_len), dtype=bool))


batch, seq_len, d_model = 2, 4, 8
Q = np.random.randn(batch, seq_len, d_model)
K = np.random.randn(batch, seq_len, d_model)
V = np.random.randn(batch, seq_len, d_model)

mask = causal_mask(seq_len)[None, :, :]
out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

print(out.shape)      # (2, 4, 8)
print(weights.shape)  # (2, 4, 4)
```

### 关键追问

- **为什么除以 $\sqrt{d_k}$？** 防止点积方差随维度增大，导致 softmax 饱和。
- **Mask 用 True 还是 False 表示可见？** 面试时必须说清楚。上面代码使用 True 表示可见。
- **为什么用 `-1e9`？** 让 masked position 的 softmax 权重近似 0。真实框架常用 dtype 对应的最小值。
- **复杂度？** 标准 attention 时间和注意力矩阵显存为 $O(n^2)$。

---

## 6. Multi-Head Attention

### NumPy 实现

```python
import numpy as np


def split_heads(x, num_heads):
    """
    x: shape (batch, seq_len, d_model)
    return: shape (batch, num_heads, seq_len, d_head)
    """
    batch, seq_len, d_model = x.shape
    assert d_model % num_heads == 0
    d_head = d_model // num_heads
    x = x.reshape(batch, seq_len, num_heads, d_head)
    return np.transpose(x, (0, 2, 1, 3))


def combine_heads(x):
    """
    x: shape (batch, num_heads, seq_len, d_head)
    return: shape (batch, seq_len, d_model)
    """
    batch, num_heads, seq_len, d_head = x.shape
    x = np.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq_len, num_heads * d_head)


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    """
    X: shape (batch, seq_len, d_model)
    Wq/Wk/Wv/Wo: shape (d_model, d_model)
    mask: optional, shape broadcastable to (batch, num_heads, seq_len, seq_len)
    """
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    Q = split_heads(Q, num_heads)
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    d_head = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_head)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)
    context = weights @ V
    context = combine_heads(context)
    return context @ Wo, weights
```

### Example

```python
batch, seq_len, d_model, num_heads = 2, 5, 16, 4
X = np.random.randn(batch, seq_len, d_model)

Wq = np.random.randn(d_model, d_model) / np.sqrt(d_model)
Wk = np.random.randn(d_model, d_model) / np.sqrt(d_model)
Wv = np.random.randn(d_model, d_model) / np.sqrt(d_model)
Wo = np.random.randn(d_model, d_model) / np.sqrt(d_model)

mask = causal_mask(seq_len)[None, None, :, :]
out, weights = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=mask)

print(out.shape)      # (2, 5, 16)
print(weights.shape)  # (2, 4, 5, 5)
```

---

## 7. 面试时主动指出的坑

### KMeans

- 空簇处理。
- 初始化敏感，需要多次重启或 k-means++。
- 特征尺度会影响欧氏距离。
- `np.all(centroids == new_centroids)` 对浮点数不稳，应使用 tolerance。

### Logistic Regression

- Sigmoid 和 log loss 数值稳定。
- 阈值不一定为 0.5。
- 类别不平衡时 accuracy 不可靠。
- 截距通常不正则化。

### Linear Regression

- 不要显式求逆。
- 共线性导致 $X^\top X$ 病态。
- 训练前检查 shape，尤其是 $y$ 是 `(n,)` 还是 `(n, 1)`。

### Attention

- Mask 语义必须清楚。
- 注意 softmax 的 axis。
- Q/K/V shape 必须能矩阵乘。
- 标准 attention 有 $O(n^2)$ 注意力矩阵。

---

## 8. 手写与实现题通用检查清单

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

## 9. 常见手写题清单

应能在不调用高级封装的情况下实现并解释：

- 稳定 Softmax、LogSoftmax、Cross-Entropy。
- 单头/多头 Attention 与 causal mask。
- LayerNorm、RMSNorm。
- 线性回归、Logistic Regression。
- K-Means、PCA 的核心步骤。
- LSTM 单步前向和参数量。
- Precision、Recall、F1、ROC-AUC 的计算。

评价手写代码时不只看结果，还要检查 shape、时间/空间复杂度、边界条件、数值稳定性和梯度正确性。
