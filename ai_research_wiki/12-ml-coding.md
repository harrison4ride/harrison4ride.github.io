# 0X. ML Coding

[返回目录](README.md)

本页整理常见手写模型。面试时不只要写出能跑的代码，还要主动说明：

- 输入输出 shape。
- 时间/空间复杂度。
- 数值稳定性。
- 边界条件。
- 和框架实现的差异。

---

## 1. KMeans

原理就两步循环：**分配**（每个点归到最近的中心）+ **更新**（每个簇的均值当新中心），直到中心不再移动。

### 先写核心版（面试建议先写这个）

面试时不要一上来就写 `class` 和一堆 `self`。先用十几行核心逻辑证明你懂原理、NumPy 向量化扎实，再主动说“如果要实际部署，我会封装成 `fit`/`predict`，并处理空簇等边界情况”，然后再补下面的完整版。

```python
import numpy as np


def kmeans(X, k, max_iter=100):
    # 1. 随便选 k 个点当中心（最简单的初始化）
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iter):
        # 2. 算距离，找归属（这步最容易写晕）
        distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. 更新中心
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # 4. 如果中心不动了，就停止
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids
```

> 注意这个极简版有两个已知取舍：空簇会得到 `NaN`（下面完整版用 fallback 处理），且 `np.all(centroids == new_centroids)` 对浮点不稳（完整版用 `tol`）。面试时能主动指出这两点就是加分项。

### 广播机制拆解（算距离那行）

`distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)` 是最容易写晕、也最常考的一行。它一次性算出「$n$ 个点到 $k$ 个中心」的距离矩阵，靠的是 NumPy 广播：

- `X` 形状 $(n, d)$，`X[:, None, :]` 插入一个空维 → $(n, 1, d)$。
- `centroids` 形状 $(k, d)$，`centroids[None, :, :]` → $(1, k, d)$。
- 相减时把大小为 1 的维广播拉伸对齐 → $(n, k, d)$，物理意义是「每个点分别减去每个中心」的坐标差 $(\Delta x, \Delta y, \dots)$。
- `** 2` 后 `.sum(axis=2)` 沿最后一维（坐标维 $d$）求和，即 $\sum(\Delta x)^2$，降维得到 $(n, k)$ 的平方距离矩阵。
- 再 `np.argmin(distances, axis=1)`：`axis=1` 是**第二维**（$k$ 个中心；NumPy 轴从 0 数起），对每一行（每个点）挑距离最小的那一列 → 长度 $n$ 的 label。

这样避免了两层 Python for 循环，是向量化标准写法（把 `np.` 换成 `torch.` 几乎就是 PyTorch）。

### K-Means++ 初始化

纯随机初始化很看运气：若初始中心挤在一起，收敛慢、易陷入差的局部最优。K-Means++ 只改**初始化**（后面 assign/update 完全不变），让初始中心尽量**相互远离**：

1. 从数据里随机选 1 个点作为第一个中心。
2. 对每个点 $x$，算它到**已选中心里最近那个**的距离 $D(x)$。
3. 以正比于 $D(x)^2$ 的概率抽下一个中心（离已有中心越远，被选中概率越大）。
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

名字叫回归，其实是**二分类**算法，三步走：

1. **线性打分**：$z=w^\top x+b$。
2. **压进概率**：用 Sigmoid $\sigma(z)=1/(1+e^{-z})$ 把 $z$ 从 $(-\infty,+\infty)$ 映射到 $(0,1)$，代表属于正类的概率。
3. **算损失并更新**：用二元交叉熵衡量猜得准不准，再用梯度下降更新 $w,b$。

### 先写核心版

```python
import numpy as np


def logistic_regression_train(X, y, lr=0.1, max_iter=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(max_iter):
        z = X @ w + b                  # 1. 前向：线性分数
        y_pred = sigmoid(z)            # 2. 过 Sigmoid 变概率

        # 3. 梯度（求导后非常干净：预测值 - 真实值）
        dw = (1 / n_samples) * (X.T @ (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # 4. 梯度下降更新
        w -= lr * dw
        b -= lr * db

    return w, b
```

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

**多元** = 多个特征（$y=w_1x_1+\dots+w_nx_n+b$），不要和**多项式回归**（引入 $x^2,x^3$ 等高次项）搞混。

和 KMeans 的“盲人摸象、迭代逼近”不同，线性回归有**上帝视角**：对均方误差求导令其为 0，可以直接解出闭式解（Normal Equation），不用写迭代循环。

### 先写核心版

`_add_intercept` 做的事就是：在 $X$ 最左边**拼一列全 1**。这样常数项 $b$ 就变成了 $w_0\times 1$，被吸收进权重向量，公式简化为纯矩阵乘法 $y=X_{\text{new}}\theta$（`theta[0]` 即截距）。

```python
import numpy as np


def linear_regression(X, y):
    # 左边拼一列 1 当截距项：b 变成权重向量的第 0 项
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    # lstsq 基于 SVD，比显式求逆稳定得多
    theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return theta                        # theta[0] 是截距 b，其余是各特征权重
```

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

## 3.1 多项式回归

多项式回归**不是新算法**，只是「特征扩展 + 多元线性回归」：把 $x$ 展开成 $[1, x, x^2, \dots]$ 当作新特征，再直接套用上面的线性回归求解器。

```python
def polynomial_regression_1d(x, y, degree=3):
    # x: (N,) 一维数据；把 x 变成矩阵 [1, x, x^2, x^3]
    # x**0 刚好全是 1，顺便把截距项也搞定了
    X_poly = np.column_stack([x ** i for i in range(degree + 1)])

    # X_poly 形状 (N, degree+1)，完全变成了多元线性回归
    theta, *_ = np.linalg.lstsq(X_poly, y, rcond=None)
    return theta
```

面试常问的坑：

- **过拟合**：degree 越高越容易剧烈震荡去穿过每个训练点（Runge 现象），泛化极差 → 用 Ridge/Lasso 把高次项权重压向 0。
- **维度灾难**：多特征做高次展开会产生大量交叉项（$x_1x_2$、$x_1^2x_2$ …），特征数急剧膨胀。
- 面试极少让从零手写，多作为概念题；关键是能说清「它本质就是特征工程 + 线性回归」。

---

## 4. Softmax

Softmax 把一堆没有约束的原始分数（logits）变成**和为 1** 的概率分布：先取指数 $e^{z}$（保证为正、并放大差距），再除以总和做归一化。

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

### 两个必说的点

- **为什么要减最大值？** 直接算 $e^{z}$，当 $z$ 很大（如 1000）时会溢出成 `inf`。减去每行最大值在数学上**完全等价**（分子分母同乘 $e^{-\max z}$），却把指数压回安全范围。这是手写 Softmax 时的必答加分项。
- **交叉熵吃 logits 还是概率？** 交叉熵本质是 $-\log(\text{真实类别的预测概率})$：猜得准（概率接近 1）惩罚趋近 0，猜得离谱（概率接近 0）惩罚巨大。实现上**直接吃 logits** 更好——用 log-sum-exp 把 softmax 和 log 合并（上面 `cross_entropy_from_logits` 的写法），少一次中间求概率的舍入，数值更稳、速度更快，这也是 PyTorch `CrossEntropyLoss` 的做法；合并后梯度还特别干净：**softmax 概率 $-$ one-hot 标签**。若手上只有概率，必须 `np.clip(p, 1e-15, 1)` 防 `log(0)`。

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

多头 = **三个臭皮匠顶个诸葛亮**：让 $h$ 个「专家」从不同视角同时读这句话（有的头盯指代关系、有的盯动宾搭配……），最后把各自的结论拼起来。

`split_heads` 的本质就是**切蛋糕（reshape）**：总维度 $d_{\text{model}}$ 平分给 $h$ 个头，每头 $d_{\text{head}}=d_{\text{model}}/h$（如 $512/8=64$），形状从 $(B, L, d_{\text{model}})$ 变成 $(B, h, L, d_{\text{head}})$。把 $h$ 独立成一个维度后，这 $h$ 个头就能**并行**算 attention；算完再 `concat` 拼回 $(B, L, d_{\text{model}})$，最后过 $W_o$ 融合。

### 核心版（先切蛋糕，再并行算，最后拼回去）

```python
import numpy as np


def multi_head_attention_core(Q, K, V, num_heads):
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads  # 每个头分到的维度（比如 512 / 8 = 64）

    # 1. split_head：把 [batch, seq, d_model] 变成 [batch, num_heads, seq, d_k]
    # （实际代码中通常先通过线性层 Linear 投影，再 reshape）
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).swapaxes(1, 2)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).swapaxes(1, 2)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).swapaxes(1, 2)

    # 2. 丢进 Attention 核心（num_heads 个头同时并行计算）
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

    out = weights @ V  # 形状是 [batch, num_heads, seq_len, d_k]

    # 3. concat（拼回去）：把 head 和 d_k 重新合体变回 [batch, seq_len, d_model]
    out = out.swapaxes(1, 2).reshape(batch_size, seq_len, d_model)

    return out
```

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

## 6.1 RoPE（旋转位置编码）

### 原理与直觉

传统绝对位置编码把位置向量**加**到词向量上，长文本外推表现不佳。RoPE 换了个思路：**用旋转表达位置**。

把词向量的每两个通道看作平面上的一个箭头，位置 $m$ 就把这个箭头旋转 $m\theta_i$ 的角度（不同通道用不同频率 $\theta_i$，低维转得快、高维转得慢）。

**关键性质**：两个分别处于位置 $m$、$n$ 的向量做点积（算 Attention Score）时，绝对旋转角自动抵消，结果**只依赖相对距离 $m-n$**：

$$
\langle R_m q,\ R_n k\rangle = q^\top R_{n-m} k
$$

这正契合语言里「相对距离比绝对位置更重要」的直觉，也是 LLaMA / Qwen / ChatGLM 等主流开源模型的标配。

### 实现

```python
import numpy as np


def apply_rope(x, freq_base=10000.0):
    """
    x: (batch_size, seq_len, num_heads, d_k)，d_k 必须是偶数（两两配对做二维旋转）
    """
    batch_size, seq_len, num_heads, d_k = x.shape
    assert d_k % 2 == 0, "d_k 必须是偶数才能进行两两旋转"

    # 1. 每个特征对的旋转频率 theta_i = base^(-2i/d_k)
    i = np.arange(0, d_k, 2)
    theta = 1.0 / (freq_base ** (i / d_k))          # (d_k/2,)

    # 2-3. 位置索引 m 与角度 m*theta 的外积
    m = np.arange(seq_len)
    freqs = np.outer(m, theta)                      # (seq_len, d_k/2)

    # 4. cos/sin 各复制一遍，对齐交错排列的 d_k 维
    cos_val = np.repeat(np.cos(freqs), 2, axis=-1)[None, :, None, :]   # (1, seq_len, 1, d_k)
    sin_val = np.repeat(np.sin(freqs), 2, axis=-1)[None, :, None, :]

    # 5. 旋转：对每一对 (x_even, x_odd) 做 x*cos + (-x_odd, x_even)*sin
    swapped = np.stack([-x[..., 1::2], x[..., ::2]], axis=-1)
    swapped = swapped.reshape(batch_size, seq_len, num_heads, d_k)

    return x * cos_val + swapped * sin_val
```

对每一对通道 $(x_{2i},x_{2i+1})$ 展开就是标准的二维旋转：

$$
x'_{2i}=x_{2i}\cos(m\theta_i)-x_{2i+1}\sin(m\theta_i),\qquad
x'_{2i+1}=x_{2i+1}\cos(m\theta_i)+x_{2i}\sin(m\theta_i)
$$

### 关键追问

- **为什么 `np.repeat(..., 2)` 而不是 `np.tile`？** 这份实现用的是**交错**排列（相邻两维为一对），所以每个角度要连续复制两次与 $(x_{2i},x_{2i+1})$ 对齐。若改用「前半 / 后半」配对的实现（HuggingFace 常见的 `rotate_half`），则要用 `np.tile` 并相应改配对方式——两种约定不能混用。
- **验证方式？** 旋转是正交变换，所以 **保范数**；位置 0 不旋转（等于原向量）；把同一对向量放到不同绝对位置、保持相对距离不变，点积应完全相同。
- **怎么扩长？** 位置插值（Position Interpolation）把每步转角调小，让同样的角度范围装下更长序列；另有 NTK-aware、YaRN 等改进。
- **作用在哪？** 只对 **Q 和 K** 施加（因为只有它们进入点积），不作用于 V。

---

## 6.2 LayerNorm

### 原理与直觉

深层网络里每层输出的数值分布会剧烈漂移，导致训练不稳。LayerNorm 对**单个样本内部的所有特征维**做归一化（均值 0、方差 1），再乘可学习的 $\gamma$、加 $\beta$ 恢复表达能力：

$$
y=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

**与 BatchNorm 的区别（面试必考）**：BatchNorm 对「一个 batch 内所有样本的同一维」求统计，受 batch 大小影响极大；LayerNorm 只看单个样本自己，**与 batch 大小、序列长度完全无关**，因此适配变长序列和逐 token 的自回归推理，是 NLP / Transformer 的标配。

### 实现

```python
import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x: (batch_size, seq_len, d_model)
    gamma, beta: (d_model,) 可学习缩放与偏移
    """
    mean = np.mean(x, axis=-1, keepdims=True)       # 沿特征维
    variance = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + eps)   # eps 防止分母为 0
    return gamma * x_norm + beta
```

### 关键追问

- **为什么 `axis=-1`？** 必须沿**特征维** $d_{\text{model}}$，不能沿 batch 或 seq 维；写错轴是最常见的 bug。
- **`eps` 放在根号里还是外面？** 主流实现（含 PyTorch）是 $\sqrt{\sigma^2+\epsilon}$，放在**根号内**。
- **RMSNorm 有什么不同？** 去掉减均值和 $\beta$，只用均方根缩放 $x/\sqrt{\operatorname{mean}(x^2)+\epsilon}\cdot\gamma$，更省算力，现代 decoder-only LLM 常用。
- **放在残差前还是后？** Post-Norm（原论文）深层易梯度消失、需 warmup；**Pre-Norm** 让残差成为无阻碍主干、训练更稳，是现代 LLM 标配。

---

## 6.3 Beam Search

### 原理与直觉

生成文本时每步只取概率最高的词（Greedy）容易陷入局部最优；穷举所有组合又会指数爆炸。Beam Search 是折中：**同时保留当前得分最高的 $k$ 条路径**（$k$ = beam size），每步扩展后再全局剪枝回 $k$ 条，直到遇到结束符，最后挑总分最高的一条。

分数用**累计对数概率**（把连乘变连加，避免概率连乘下溢）：

$$
\text{score}(y_{1:t})=\sum_{i=1}^{t}\log p(y_i\mid y_{<i})
$$

### 实现

```python
import numpy as np


def beam_search(step_fn, start_token, eos_token, beam_size=3, max_len=10,
                length_penalty=0.0):
    """
    step_fn(seq) -> 长度为 vocab 的 log-prob 向量（给定已生成序列，预测下一个 token）
    返回得分最高的 (score, seq)
    """
    beams = [(0.0, [start_token])]      # (累计 log-prob, 序列)
    finished = []

    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_token:                       # 已结束，移入 finished
                finished.append((score, seq))
                continue
            log_probs = step_fn(seq)
            for tok in np.argsort(log_probs)[-beam_size:]:  # 每束只展开 top-k
                candidates.append((score + log_probs[tok], seq + [int(tok)]))

        if not candidates:
            break
        candidates.sort(key=lambda t: t[0], reverse=True)
        beams = candidates[:beam_size]                      # 全局剪枝回 k 条

    finished.extend(beams)

    def norm(item):                                         # 长度惩罚
        s, seq = item
        return s / (len(seq) ** length_penalty) if length_penalty else s

    return max(finished, key=norm)
```

### 关键追问

- **为什么要长度惩罚？** 累计 log-prob 恒为负，越长分越低，不加惩罚会**偏爱短句**。常用 $\text{score}/|y|^{\alpha}$（$\alpha\approx0.6\sim1.0$）。
- **Beam Search 保证最优吗？** **不保证**。每步全局只留 $k$ 条，可能把「当下分低、后面才翻盘」的路径提前剪掉，所以它是启发式近似而非精确搜索。
- **beam 越大越好吗？** 不是。$k$ 增大计算量线性增长，且实践中过大反而更易产生**通用、乏味**的句子（beam search curse）；开放式生成常改用 top-k / top-p 采样。
- **复杂度？** 每步约 $O(k\cdot V)$（$V$ 为词表），整体 $O(k V \cdot L)$。
- **和 Greedy 的关系？** $k=1$ 时 Beam Search 退化为 Greedy Search。

---

## 6.4 Top-K / Top-P 采样

### 原理与直觉

Greedy 每步取最大值，输出死板且容易复读；直接按全词表采样又会偶尔抽到概率极低的垃圾 token（词表 15 万个，长尾加起来的概率不小）。两种截断办法：

- **Top-K**：只留概率最高的 $K$ 个，重新归一化后采样。简单，但 $K$ 是固定的——分布尖锐时放进太多垃圾，分布平坦时又砍掉合理选项。
- **Top-P（Nucleus）**：把 token 按概率从高到低排序，取**累计概率刚好超过 $p$** 的最小集合。候选集大小随分布自动伸缩，是它相对 Top-K 的核心优势。

温度在截断之前作用于 logits：$p_i=\dfrac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$，$T<1$ 让分布更尖，$T>1$ 更平。

### 实现

```python
import numpy as np


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def top_k_sampling(logits, k=50, temperature=1.0, rng=None):
    """logits: shape (vocab,)；返回采样得到的 token id"""
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=float) / max(temperature, 1e-8)

    k = min(k, logits.size)
    idx = np.argpartition(logits, -k)[-k:]      # O(V)，不必全排序
    probs = softmax(logits[idx])                # 只在候选集内归一化
    return int(rng.choice(idx, p=probs))


def top_p_sampling(logits, p=0.9, temperature=1.0, rng=None):
    """核采样：取累计概率刚好超过 p 的最小候选集"""
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=float) / max(temperature, 1e-8)

    probs = softmax(logits)
    order = np.argsort(probs)[::-1]             # 从大到小
    cumsum = np.cumsum(probs[order])

    cutoff = np.searchsorted(cumsum, p) + 1     # 至少保留 1 个
    idx = order[:cutoff]

    kept = probs[idx] / probs[idx].sum()        # 截断后重新归一化
    return int(rng.choice(idx, p=kept))
```

### 关键追问

- **截断后为什么必须重新归一化？** 砍掉长尾后剩下的概率加起来小于 1，不归一化就不是合法分布，`rng.choice` 也会直接报错。
- **Top-K 一定要全排序吗？** 不用。`np.argpartition` 是 $O(V)$ 的选择算法，只保证第 $k$ 位左右分开、不保证组内有序，比 $O(V\log V)$ 的全排序快。Top-P 因为需要累计和，才不得不排序。
- **$T\to0$ 会怎样？** 退化成 greedy（最大值的概率趋于 1）。实现上要防除零，所以写成 `max(temperature, 1e-8)`；工程里通常直接判断 `T == 0` 就走 argmax。
- **两者能一起用吗？** 能，而且常见。工业实现一般先温度缩放，再 top-k 粗筛，再 top-p 细筛，最后采样。
- **为什么不直接在全词表上采样？** 长尾 token 单个概率极低，但数量巨大，累积起来仍会被抽中，一旦抽到就毁掉整句。截断的本质是「砍掉不可能选项，保留合理的随机性」。

---

## 7. 决策树：信息熵与信息增益

决策树每一步要决定「先问哪个特征、在哪切」，标准是：切完之后子集**最纯**。

- **熵（Entropy）**：$H(y)=-\sum_c p_c\log_2 p_c$，衡量混乱程度。全是同一类 → $H=0$（最纯）；各类均匀 → $H$ 最大（最乱）。
- **信息增益（Information Gain）**：$IG=H(\text{父})-\sum_{\text{子}}\frac{n_{\text{子}}}{n}H(\text{子})$，即「切分前的混乱度 $-$ 切分后子集混乱度的加权平均」。谁让混乱度下降最多（$IG$ 最大），就选谁分裂。

```python
import numpy as np


def calculate_entropy(y):
    # y 是标签数组，比如 [0, 0, 1, 1, 1]
    n_samples = len(y)
    if n_samples == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)   # 统计各类别频次
    probabilities = counts / n_samples
    # 核心公式：Entropy = - sum(p * log2(p))；np.unique 保证 p>0
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    child = (len(y_left) / n) * calculate_entropy(y_left) \
          + (len(y_right) / n) * calculate_entropy(y_right)
    return calculate_entropy(y_parent) - child
```

面试最常让手写的就是 `calculate_entropy`；Gini 不纯度 $1-\sum_c p_c^2$ 是另一个常见替代（CART 默认）。

---

## 8. SVM（支持向量机）

面试极少让手写完整的 SMO 优化算法（太长），重点是能讲清几何直觉：

- **最大间隔**：能把两类分开的超平面有无数条，SVM 要找**走在正中间、两边最空旷**的那条——即让「离它最近的点（**支持向量**）到它的间隔」最大化，泛化更好。
- **核技巧（Kernel Trick）**：线性不可分时（比如一类被另一类包围），用核函数（如 RBF 高斯核）把数据隐式映射到高维空间，在高维里一刀切开，而无需显式计算高维坐标。
- 代码通常直接调库；`C` 越大越不容忍误分类（间隔越硬、越容易过拟合）。

```python
from sklearn.svm import SVC

clf = SVC(kernel='rbf', C=1.0)     # RBF（高斯核）支持向量机
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

## 8.1 Cosine Similarity

### 原理与直觉

$$
\cos(a,b)=\frac{a\cdot b}{\|a\|\,\|b\|}
$$

分子是内积，分母把两个向量的长度除掉，所以它**只看方向、不看长度**，取值范围 $[-1,1]$。

这正是它在检索和 RAG 里的价值：一篇长文档的 embedding 模长通常更大，如果直接用内积，长文档会天然占便宜；除掉模长之后，比较的才是「语义方向像不像」。

和欧氏距离的关系：如果向量已做 L2 归一化，则 $\|a-b\|^2=2-2\cos(a,b)$——两者单调等价。所以向量库通常先把所有向量归一化，之后只需算内积，省掉除法。

### 实现

```python
import numpy as np


def cosine_similarity(a, b, eps=1e-8):
    """两个一维向量的余弦相似度"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def cosine_similarity_matrix(A, B, eps=1e-8):
    """
    A: shape (n, d)  B: shape (m, d)
    返回 shape (n, m)，A 中每一行与 B 中每一行的相似度
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return A_norm @ B_norm.T          # 归一化之后，内积就是余弦
```

### 关键追问

- **为什么要 `eps`？** 零向量的模长是 0，直接除会得到 `nan`。检索场景里空文本、全 padding 的行都可能产生零向量。
- **批量版为什么先归一化再矩阵乘？** 归一化是 $O(nd)$，之后一次矩阵乘就能拿到全部 $n\times m$ 对的相似度，交给 BLAS 跑；逐对循环则是 Python 层的 $O(nm)$ 次调用，慢几个数量级。
- **余弦相似度是距离吗？** 不是。它越大越相似，与距离方向相反；常用 $1-\cos$ 当作「余弦距离」，但它不满足三角不等式，不是严格的度量。
- **什么时候不该用它？** 当模长本身有意义时（比如用词频计数向量表示强度、或推荐里的置信度），归一化会把这部分信息扔掉。

---

## 8.2 Precision / Recall / F1

### 原理与直觉

从混淆矩阵出发（以正类为关注对象）：

$$
\text{Precision}=\frac{TP}{TP+FP},\qquad
\text{Recall}=\frac{TP}{TP+FN},\qquad
F_1=\frac{2PR}{P+R}
$$

一句话记法：

- **Precision（查准率）**：我说是正的里面，真的有多少。分母是「我预测的正类」，管的是**别误报**。
- **Recall（查全率）**：真正的正类里面，我抓到了多少。分母是「实际的正类」，管的是**别漏报**。

两者天然对立：把阈值调低，什么都判成正类 → recall 冲到 1、precision 崩掉；阈值调高只报最有把握的 → precision 高、recall 低。**F1 是两者的调和平均**，用调和平均而不是算术平均，是因为它对偏科更狠——0.9 和 0.1 的算术平均是 0.5，F1 只有 0.18。

### 实现

```python
import numpy as np


def precision_recall(y_true, y_pred, eps=1e-12):
    """二分类，标签为 0/1；返回 precision、recall、f1"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def precision_recall_multiclass(y_true, y_pred, num_classes, average='macro'):
    """多分类：对每个类别做一次 one-vs-rest，再平均"""
    scores = []
    weights = []
    for c in range(num_classes):
        p, r, f = precision_recall(y_true == c, y_pred == c)
        scores.append((p, r, f))
        weights.append(np.sum(y_true == c))

    scores = np.array(scores, dtype=float)
    if average == 'macro':                       # 每类等权，小类同样重要
        return tuple(scores.mean(axis=0))
    weights = np.array(weights, dtype=float)     # weighted：按样本数加权
    return tuple(scores.T @ weights / weights.sum())
```

### 关键追问

- **分母为 0 怎么办？** 一个正类都没预测出来时 $TP+FP=0$。加 `eps` 只是防崩，业务上应显式约定返回 0 并告警——这种情况通常意味着阈值或训练本身有问题。sklearn 的做法是 `zero_division` 参数。
- **为什么不用 accuracy？** 类别不平衡时它没有信息量：1% 正例的数据，全预测成负类就有 99% accuracy，而 recall 是 0。
- **macro 和 micro 有什么区别？** macro 对每个类别单独算再取平均，**小类和大类等权**；micro 是把所有类别的 TP/FP/FN 汇总后再算，**被大类主导**（多分类单标签下 micro-F1 等于 accuracy）。关心稀有类就看 macro。
- **precision 和 recall 谁更重要？** 取决于错误的代价：垃圾邮件过滤怕误杀正常邮件 → 重 precision；癌症筛查怕漏诊 → 重 recall。也可以用 $F_\beta$ 调整偏好，$\beta>1$ 偏 recall。
- **和 PR-AUC 的关系？** 上面算的是**某一个阈值**下的一组数；扫遍所有阈值把 (recall, precision) 连成曲线，其下面积就是 PR-AUC，不受阈值选择影响。不平衡数据上它比 ROC-AUC 更敏感（见 [02. 模型评估与指标](02-evaluation.md)）。

---

## 目录

| 章节                                          |
| --------------------------------------------- |
| [00. 机器学习核心概念](00-ml-concepts.md)     |
| [01. 基础与神经网络机制](01-foundations.md)   |
| [02. 模型评估与指标](02-evaluation.md)        |
| [04. 经典机器学习](04-classical-ml.md)        |
| [05. NLP、RNN 与词向量](05-nlp-rnn.md)        |
| [06. LLM 基础](06-llm-foundations.md)         |
| [07. 训练与系统](07-training-and-systems.md)  |
| [08. 对齐与 RLHF](08-alignment-and-rlhf.md)   |
| [09. 推理与部署](09-inference-and-serving.md) |
| [12. ML Coding](12-ml-coding.md)              |
| [参考资料](references.md)                     |