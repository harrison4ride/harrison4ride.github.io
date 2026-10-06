# 0X. ML Coding

[返回目录](README.md)

本页整理常见手写模型，以及 MLE 电面里的高频算法题（Top-K、蓄水池抽样）。面试时不只要写出能跑的代码，还要主动说明：

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

#### Sigmoid：公式与它的坑

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

照着这个式子直接写 `1 / (1 + np.exp(-z))`，当 $z$ 是很大的负数（比如 $-1000$）时，$e^{-z}=e^{1000}$ 直接溢出成 `inf`，得到 `inf` 甚至 `nan`。

解决办法是分子分母同乘 $e^{z}$，得到一个等价形式：

$$
\sigma(z)=\frac{1}{1+e^{-z}}=\frac{e^{z}}{1+e^{z}}
$$

两个写法**数学上完全相同，数值行为却相反**：

- $z\ge0$ 时用左边，指数部分是 $e^{-z}\le1$，安全。
- $z<0$ 时用右边，指数部分是 $e^{z}<1$，也安全。

所以按符号分两支，保证**永远只对负数取指数**。

```python
import numpy as np


def sigmoid(z):
    z = np.asarray(z)
    out = np.empty_like(z, dtype=float)   # 创建一个和 z 维度相同、用来装结果的空数组

    pos = z >= 0    # 找到所有 >= 0 的位置（布尔掩码 Boolean Mask）
    neg = ~pos      # 找到所有 < 0 的位置（按位取反）

    # 1. 正数区域：用 1 / (1 + exp(-z))，此时 exp(-z) <= 1，不会溢出
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    # 2. 负数区域：用 exp(z) / (1 + exp(z))，此时 exp(z) < 1，同样不会溢出
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)

    return out
```

#### Binary Cross-Entropy：同一个坑的另一面

原始公式（记 $p=\sigma(z)$）：

$$
\mathcal L=-\big[y\log p+(1-y)\log(1-p)\big]
$$

问题在于先求出 $p$ 再取对数：$p$ 一旦被浮点舍入成 0 或 1，就撞上 $\log 0=-\infty$。把 $p=\sigma(z)$ 代进去化简，可以得到一个**只用 logits、且不会出现 $\log0$** 的等价式：

$$
\mathcal L=\max(z,0)-zy+\log\!\left(1+e^{-|z|}\right)
$$

$\max(z,0)$ 与 $|z|$ 配合，保证指数项永远是 $e^{\text{负数}}\in(0,1]$，怎么算都不溢出。

```python
def binary_cross_entropy_with_logits(logits, y):
    """稳定形式：max(z, 0) - z*y + log(1 + exp(-|z|))；直接吃 logits，不要先算概率"""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)

    # -np.abs(logits) 保证指数恒为负 → exp 落在 (0, 1]，永不溢出
    # np.log1p(t) 算的是 log(1+t)，t 很小时比 np.log(1+t) 精度高得多
    loss = np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return loss.mean()          # 对 batch 求平均


class LogisticRegressionGD:
    def __init__(self, lr=0.1, max_iter=1000, l2=0.0, fit_intercept=True):
        self.lr = lr                      # 学习率
        self.max_iter = max_iter          # 迭代轮数
        self.l2 = l2                      # L2 正则强度，0 表示不加正则
        self.fit_intercept = fit_intercept
        self.w = None
        self.loss_history = []            # 记录每轮 loss，用来画收敛曲线 / 调试

    def _add_intercept(self, X):
        """在 X 最左边拼一列全 1，把截距 b 吸收成 w[0]，公式简化为纯矩阵乘"""
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)   # 拉平成 (n,)，防止 (n,1) 触发广播 bug
        Xb = self._add_intercept(X)

        n_samples, n_features = Xb.shape
        self.w = np.zeros(n_features)     # BCE + 线性 logit 是凸问题，零初始化就够
        self.loss_history = []

        for _ in range(self.max_iter):
            logits = Xb @ self.w          # 1. 前向只算到 logits，不提前求概率
            probs = sigmoid(logits)       # 2. 转成概率

            # 3. 梯度：对 logits 求导后是 (p - y)，链式回到 w 就是 X^T (p - y) / n
            grad = Xb.T @ (probs - y) / n_samples

            if self.l2 > 0:               # 4. L2 正则的梯度就是 lambda * w
                reg = self.w.copy()
                if self.fit_intercept:
                    reg[0] = 0.0          # 截距不参与正则（它只是整体基准，不该被收缩）
                grad += self.l2 * reg

            self.w -= self.lr * grad      # 5. 梯度下降更新

            # 6. 用更新后的参数记录 loss，曲线才反映当前模型
            logits = Xb @ self.w
            loss = binary_cross_entropy_with_logits(logits, y)
            if self.l2 > 0:               # 正则项也要计入 loss，否则曲线和目标函数对不上
                reg_w = self.w[1:] if self.fit_intercept else self.w
                loss += 0.5 * self.l2 * np.dot(reg_w, reg_w)
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        """输出正类概率，落在 (0, 1)"""
        X = np.asarray(X, dtype=float)
        Xb = self._add_intercept(X)
        return sigmoid(Xb @ self.w)

    def predict(self, X, threshold=0.5):
        """阈值默认 0.5，但生产中应由 precision/recall 的业务成本决定"""
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

### `lstsq` 是什么？要手写它吗？

**不用手写。** `np.linalg.lstsq` 是 LAPACK 提供的最小二乘求解器（内部走 SVD 或 QR 分解），面试里直接调用即可——手撕 SVD 不是这道题的考点。真正要能讲清的是**为什么用它，而不是照抄闭式解**：

$$
\theta=(X^\top X)^{-1}X^\top y
$$

这个式子适合写在纸上，不适合写进代码。三种写法从差到好：

| 写法                                | 问题                                                                        |
| ----------------------------------- | --------------------------------------------------------------------------- |
| `inv(X.T @ X) @ X.T @ y`            | 显式求逆，最不稳；且 $X^\top X$ 的条件数是 $X$ 的**平方**，共线性一强就炸    |
| `solve(X.T @ X, X.T @ y)`           | 不求逆、改解线性方程组，好一些；但仍要构造 $X^\top X$，条件数平方的问题还在  |
| `lstsq(X, y)`                       | **根本不构造 $X^\top X$**，直接对 $X$ 分解求最小二乘；$X$ 秩亏时还给最小范数解 |

所以标准答法是：先写出闭式解证明你懂推导，再说「实际实现我会调 `lstsq`，因为它不显式求逆、也不构造条件数被平方的 $X^\top X$」。

### 面试版实现（封装成类）

```python
import numpy as np


class LinearRegressionClosedForm:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.theta = None                 # 拟合后是长度 (n_features + 1) 的权重向量

    def _add_intercept(self, X):
        """拼一列全 1，让截距 b 变成 theta[0]"""
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xb = self._add_intercept(X)

        # 比显式算 inv(X.T @ X) 稳得多；rcond=None 采用新版默认的奇异值截断规则
        # 返回值依次是：解、残差平方和、X 的秩、奇异值
        self.theta, residuals, rank, singular_values = np.linalg.lstsq(
            Xb, y, rcond=None
        )
        # rank < Xb.shape[1] 就说明特征间存在完全共线，此时解不唯一
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._add_intercept(X)       # 预测时也必须拼同样的一列 1
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

闭式解只需在 $X^\top X$ 上加一个 $\alpha I$：

$$
\theta=(X^\top X+\alpha I)^{-1}X^\top y
$$

这一项的作用不只是「防过拟合」——加在对角线上会把最小的那些奇异值抬起来，$X^\top X$ 即使原本奇异也变得可逆，所以 Ridge 天然能处理共线性。

```python
def ridge_regression(X, y, alpha=1.0, fit_intercept=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if fit_intercept:
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])   # 拼截距列
    else:
        Xb = X

    n_features = Xb.shape[1]
    reg = alpha * np.eye(n_features)      # alpha * I，加在 X^T X 的对角线上
    if fit_intercept:
        reg[0, 0] = 0.0                   # 截距项不正则化

    # 用 solve 而不是 inv：解方程比求逆更稳、更快
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

Softmax 把一堆没有约束的原始分数（logits）变成**和为 1** 的概率分布：

$$
\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_{j}e^{z_j}}
$$

分子取指数保证结果为正、并放大分数差距，分母是所有项之和保证归一化。

### 稳定实现

真正写的是下面这个等价形式（分子分母同乘 $e^{-\max_j z_j}$）：

$$
\operatorname{softmax}(z)_i=\frac{e^{z_i-\max_j z_j}}{\sum_k e^{z_k-\max_j z_j}}
$$

```python
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)

    # 减去每行最大值：数学上完全等价，但把最大的指数压成 e^0 = 1，杜绝上溢
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)

    # keepdims=True 保留被求和的那一维（(batch, 1) 而不是 (batch,)），否则广播会错位
    return exp_x / exp_x.sum(axis=axis, keepdims=True)
```

**为什么不写成更短的那个版本？** 你可能见过这种写法：

```python
def softmax(x):            # 只对「一维向量」正确
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

单条 logits 向量喂进去没问题，但一带 batch 就错了：`np.max(x)` 取的是**整个矩阵**的最大值，`e.sum()` 把**所有样本的所有类别**加成一个标量。结果每行不再各自归一化，而是整个矩阵加起来才等于 1，概率全被稀释成 $1/\text{batch}$ 量级。

所以只要输入可能有 batch 维，就必须写 `axis=-1, keepdims=True` 让每行独立归一化。面试时写带 `axis` 的版本，并主动说明这个区别。

### Cross-Entropy

多分类交叉熵只看**真实类别那一项**的概率：

$$
\mathcal L=-\frac1N\sum_{i=1}^{N}\log p_{i,y_i},
\qquad p_{i,k}=\operatorname{softmax}(z_i)_k
$$

实现上不要先算 softmax 再取 log，而是把两步合并成 **log-softmax**，除法变减法、也不会出现 $\log 0$：

$$
\log\operatorname{softmax}(z)_k
=z_k-\max_j z_j-\log\sum_{m}e^{z_m-\max_j z_j}
$$

```python
def cross_entropy_from_logits(logits, y):
    """
    logits: shape (batch, num_classes)，未经 softmax 的原始分数
    y:      shape (batch,)，整数类别标签（不是 one-hot）
    """
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=int)

    # 1. 同样先减每行最大值做稳定化
    shifted = logits - logits.max(axis=1, keepdims=True)

    # 2. log-softmax：log(e^s / Σe^s) = s - log(Σe^s)，全程没有除法、也不会 log(0)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    # 3. 花式索引取出每个样本真实类别那一项：第 i 行取第 y[i] 列
    #    等价于「和 one-hot 相乘再求和」，但不必真的构造 one-hot 矩阵
    return -log_probs[np.arange(logits.shape[0]), y].mean()
```

### 两个必说的点

- **为什么要减最大值？** 直接算 $e^{z}$，当 $z$ 很大（如 1000）时会溢出成 `inf`。减去每行最大值在数学上**完全等价**（分子分母同乘 $e^{-\max z}$），却把指数压回安全范围。这是手写 Softmax 时的必答加分项。
- **交叉熵吃 logits 还是概率？** 交叉熵本质是 $-\log(\text{真实类别的预测概率})$：猜得准（概率接近 1）惩罚趋近 0，猜得离谱（概率接近 0）惩罚巨大。实现上**直接吃 logits** 更好——用 log-sum-exp 把 softmax 和 log 合并（上面 `cross_entropy_from_logits` 的写法），少一次中间求概率的舍入，数值更稳、速度更快，这也是 PyTorch `CrossEntropyLoss` 的做法；合并后梯度还特别干净：**softmax 概率 $-$ one-hot 标签**。若手上只有概率，必须 `np.clip(p, 1e-15, 1)` 防 `log(0)`。

---

## 5. Scaled Dot-Product Attention

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

四步：$QK^\top$ 算出每个 query 和每个 key 的匹配分 → 除 $\sqrt{d_k}$ 防止 softmax 饱和 → 加 mask 屏蔽不该看的位置 → softmax 归一成权重后去加权 $V$。

### 单头 Attention

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)     # 稳定化：减去每行最大值
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: shape (batch, q_len, d_k)
    K: shape (batch, kv_len, d_k)
    V: shape (batch, kv_len, d_v)
    mask: optional, broadcastable to (batch, q_len, kv_len).
          本实现约定 True = 该位置可见（不同框架约定相反，面试务必说清）

    Returns:
        output: shape (batch, q_len, d_v)
        weights: shape (batch, q_len, kv_len)
    """
    d_k = Q.shape[-1]

    # 1. 打分并缩放：swapaxes(-1,-2) 是转置最后两维，(q_len,d_k)@(d_k,kv_len) -> (q_len,kv_len)
    #    除 sqrt(d_k)：点积是 d_k 项之和、方差随维度线性增长，不缩放会让 softmax 饱和、梯度消失
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)

    # 2. 屏蔽：把不可见位置换成一个极小值，softmax 后权重 ≈ 0
    #    必须在 softmax 之前做，否则归一化分母里仍混进了这些位置
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # 3. 沿 kv_len 维归一化：每个 query 对所有 key 的注意力加起来为 1
    weights = softmax(scores, axis=-1)

    # 4. 用权重加权求和 V，得到每个 query 位置的输出
    output = weights @ V
    return output, weights
```

### Causal Mask

```python
def causal_mask(q_len, kv_len=None):
    if kv_len is None:
        kv_len = q_len
    # np.tril 取下三角（含对角线）：第 i 行只有前 i+1 列为 True
    # 即位置 i 只能看到 <= i 的位置，看不到未来
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
    切蛋糕：把 d_model 平分给 num_heads 个头
    x: shape (batch, seq_len, d_model)
    return: shape (batch, num_heads, seq_len, d_head)
    """
    batch, seq_len, d_model = x.shape
    assert d_model % num_heads == 0          # 必须整除，否则没法平分
    d_head = d_model // num_heads

    # 先把最后一维拆成 (num_heads, d_head)
    x = x.reshape(batch, seq_len, num_heads, d_head)
    # 再把 num_heads 换到前面 —— 让它变成"批次维"，后面 h 个头就能并行算 attention
    return np.transpose(x, (0, 2, 1, 3))


def combine_heads(x):
    """
    拼回去：split_heads 的逆操作
    x: shape (batch, num_heads, seq_len, d_head)
    return: shape (batch, seq_len, d_model)
    """
    batch, num_heads, seq_len, d_head = x.shape
    x = np.transpose(x, (0, 2, 1, 3))        # 先把 seq_len 换回第 1 维
    return x.reshape(batch, seq_len, num_heads * d_head)   # 再把 h 个头首尾相接


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    """
    X: shape (batch, seq_len, d_model)
    Wq/Wk/Wv/Wo: shape (d_model, d_model)
    mask: optional, shape broadcastable to (batch, num_heads, seq_len, seq_len)
    """
    # 1. 三次线性投影，把同一个 X 变成 Q/K/V 三种视角（三组权重互相独立，不共享）
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # 2. 切成多头：注意是先投影再切，等价于每个头有自己的小投影矩阵
    Q = split_heads(Q, num_heads)
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    # 3. 每个头各算各的 attention。这里 d_head = d_model / h，
    #    所以缩放用的是 sqrt(d_head) 而不是 sqrt(d_model)
    d_head = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_head)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    weights = softmax(scores, axis=-1)
    context = weights @ V                    # (batch, num_heads, seq_len, d_head)

    # 4. 拼回 d_model，再过输出投影 Wo 把各头的结论融合起来
    #    没有 Wo 的话各头结果只是简单并排堆着，从未交互
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
    """这里输入是单条一维 logits，所以可以用不带 axis 的简版（见第 4 节的说明）"""
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def top_k_sampling(logits, k=50, temperature=1.0, rng=None):
    """logits: shape (vocab,)；返回采样得到的 token id"""
    rng = rng or np.random.default_rng()

    # 温度缩放要在 softmax 之前作用于 logits；max(..., 1e-8) 防止 T=0 时除零
    logits = np.asarray(logits, dtype=float) / max(temperature, 1e-8)

    k = min(k, logits.size)                     # k 比词表还大时退化成全采样
    # argpartition 只保证第 k 大的元素就位、左右两边各自不排序，O(V) 比全排序 O(VlogV) 快
    idx = np.argpartition(logits, -k)[-k:]      # 取最大的 k 个的下标
    probs = softmax(logits[idx])                # 只在候选集内归一化（截断后必须重新归一）
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
    # 子节点熵的加权平均，权重是落进该子节点的样本占比
    # 必须加权：否则一个只有 2 个样本的纯净子节点会把整个分裂的评分骗高
    child = (len(y_left) / n) * calculate_entropy(y_left) \
          + (len(y_right) / n) * calculate_entropy(y_right)
    # 切分前的混乱度 - 切分后的期望混乱度，越大说明这一刀切得越有效
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

## 9. KNN（K 近邻）

### 原理与直觉

KNN 就是「近朱者赤」：要预测一个新点，就去训练集里找离它最近的 $k$ 个点，让它们投票。

- **分类**：$k$ 个邻居里哪个类别最多，就预测哪个。
- **回归**：取 $k$ 个邻居标签的平均值。

KNN **没有训练过程**，`fit` 只是把训练数据原样存起来（lazy learning），所有计算都推迟到预测时。这和别的模型正好相反：训练几乎免费，推理很贵。概念部分见 [04. 经典机器学习](04-classical-ml.md) 的 KNN 一节。

### 先写核心版

```python
import numpy as np
from collections import Counter


def knn_predict_one(X_train, y_train, x, k=3):
    # 1. x 到每个训练点的距离：(n, d) - (d,) 广播成 (n, d)，沿 d 求和 -> (n,)
    distances = np.sqrt(((X_train - x) ** 2).sum(axis=1))
    # 2. 取距离最小的 k 个下标
    nearest = np.argsort(distances)[:k]
    # 3. 多数投票：most_common(1) 返回 [(类别, 票数)]
    return Counter(y_train[nearest]).most_common(1)[0][0]
```

这个版本一次只预测一个点，预测 $m$ 个点就要在外面套一层 Python 循环。面试官的下一个问题通常是：**能不能不用循环，一次算出所有测试点到所有训练点的距离？**

### 向量化：用展开公式算距离矩阵

KMeans 那节的广播写法 `X[:, None, :] - centroids[None, :, :]` 会生成 $(m, n, d)$ 的中间数组。KMeans 里中心只有 $k$ 个，问题不大；KNN 里训练点有 $n$ 个，$m\times n\times d$ 很容易撑爆内存。更好的办法是把平方距离展开：

$$
\|a-b\|^2=\|a\|^2-2\,a\cdot b+\|b\|^2
$$

中间的 $a\cdot b$ 对所有点对一起算，就是一次矩阵乘法 $AB^\top$，内存只需要 $(m,n)$。

```python
def pairwise_sq_dist(A, B):
    """
    A: (m, d)  B: (n, d)
    返回 (m, n) 的平方距离矩阵：第 i 行第 j 列是 A[i] 和 B[j] 的平方距离
    """
    A_sq = (A ** 2).sum(axis=1, keepdims=True)    # (m, 1)
    B_sq = (B ** 2).sum(axis=1)                   # (n,)，相加时广播成 (1, n)
    d2 = A_sq - 2 * A @ B.T + B_sq                # (m, 1) - (m, n) + (n,) -> (m, n)
    return np.maximum(d2, 0)                      # 浮点误差可能算出 -1e-12，截到 0
```

为什么要 `np.maximum(d2, 0)`：两个点几乎重合时，$\|a\|^2$ 和 $2a\cdot b+\ldots$ 是两个很接近的大数相减，舍入误差会让结果变成一个很小的负数，后面再开根号就得到 `nan`。

### 面试版实现

```python
class KNN:
    def __init__(self, k=3, task="classification"):
        self.k = k
        self.task = task                  # "classification" 或 "regression"

    def fit(self, X, y):
        # 没有训练：只是把数据存起来
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        k = min(self.k, len(self.X_train))
        d2 = pairwise_sq_dist(X, self.X_train)            # (m, n)；找最近邻不需要开根号

        # 1. argpartition：只把每行最小的 k 个挪到前面，O(n)，不用 O(n log n) 全排序
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]   # (m, k)，这 k 个之间无序

        # 2. 把这 k 个按距离排好，平票时 Counter 会选先出现的类别 = 离得最近的那个
        order = np.argsort(np.take_along_axis(d2, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        neighbor_y = self.y_train[idx]                    # (m, k)，每行是一个测试点的 k 个邻居标签

        if self.task == "regression":
            return neighbor_y.mean(axis=1)                # 回归：邻居标签取平均
        # 分类：每行做一次多数投票
        return np.array([Counter(row).most_common(1)[0][0] for row in neighbor_y])
```

### Example

```python
rng = np.random.default_rng(0)
# 两团高斯点：类别 0 围绕 (0, 0)，类别 1 围绕 (3, 3)
X_train = np.vstack([rng.normal(0, 1, (50, 2)), rng.normal(3, 1, (50, 2))])
y_train = np.array([0] * 50 + [1] * 50)
X_test = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(3, 1, (20, 2))])
y_test = np.array([0] * 20 + [1] * 20)

model = KNN(k=5).fit(X_train, y_train)
print("test accuracy:", (model.predict(X_test) == y_test).mean())   # test accuracy: 1.0
```

### 关键追问

- **复杂度？** `fit` 是 $O(1)$，只存数据，但要 $O(nd)$ 内存。预测一个点要算 $n$ 个距离，$O(nd)$，再选 top-k，$O(n)$。$m$ 个测试点合计 $O(mnd)$。
- **$k$ 怎么选？** $k$ 太小对噪声敏感，决策边界碎，容易过拟合（$k=1$ 时一个标错的点就能带偏它周围一片）。$k$ 太大边界过于平滑，容易欠拟合（$k=n$ 时永远预测训练集里最多的类）。用交叉验证选；二分类取奇数 $k$ 可以避免平票。
- **平票怎么办？** 上面的实现把邻居按距离排好，平票时选离得最近的那个邻居的类别（sklearn 的 `KNeighborsClassifier` 平票时选标签值最小的类，所以两者只在平票处可能不同）。另一个常见做法是**距离加权投票**：权重取 $1/(\text{距离}+\epsilon)$，近的邻居说话更有分量。
- **为什么要标准化？** 距离会被尺度最大的特征主导。比如「年龄」在 0 到 100、「年收入」在 0 到 $10^6$，不标准化时年龄几乎不起作用。
- **为什么在训练集上算 KNN 的准确率会虚高？** 每个训练点最近的邻居就是它自己，距离为 0。$k=1$ 时训练准确率恒为 100%（除非有重复点标签不同）。评估一定要用留出的测试集。
- **高维时有什么问题？** 维度灾难：维度很高时，最近点和最远点的距离之比趋近于 1，「最近」失去意义（Beyer et al., 1999, *When Is "Nearest Neighbor" Meaningful?*）。实践中先降维，或者用学出来的 embedding。
- **数据量很大怎么办？** 低维可以用 KD-Tree、Ball Tree 剪枝；高维 embedding 用近似最近邻（ANN）索引，比如 FAISS 的 IVF、HNSW 图索引，用一点召回率换几个数量级的速度。RAG 和推荐召回里的「向量检索」本质上就是 KNN，距离常用 8.1 节的余弦相似度。

---

## 10. 优化器：SGD 与 Momentum

### 原理与直觉

训练就是**下山**：loss 是地形，参数 $\theta$ 是你站的位置，梯度 $g=\nabla_\theta\mathcal L$ 指向上坡最陡的方向。所以每一步往梯度的反方向走一小步：

$$
\theta_{t+1}=\theta_t-\eta\,g_t
$$

$\eta$ 是学习率，也就是步长。SGD 里的 S（Stochastic）指 $g_t$ 只用一个 mini-batch 算出来，是全量梯度的一个带噪声的估计：算得快，但每步方向会抖。

Momentum 给下山的人加上**惯性**，像一个小球滚下山坡：

$$
v_{t+1}=\beta\,v_t+g_t,\qquad
\theta_{t+1}=\theta_t-\eta\,v_{t+1}
$$

$v$ 是速度，$\beta$ 是动量系数（通常取 0.9）。把递推展开就看得很清楚：

$$
v_{t+1}=g_t+\beta\,g_{t-1}+\beta^2 g_{t-2}+\cdots
$$

速度等于过去所有梯度的加权和，越旧的梯度权重越小。

### 先写核心版

```python
import numpy as np


def sgd_step(w, grad, lr=0.01):
    return w - lr * grad                   # 沿梯度反方向走一步


def momentum_step(w, grad, v, lr=0.01, momentum=0.9):
    v = momentum * v + grad                # 1. 新速度 = 衰减后的旧速度 + 当前梯度
    w = w - lr * v                         # 2. 用速度更新参数，梯度只负责改速度
    return w, v                            # v 要返回，下一步接着用
```

面试时边写边讲：`v` 和 `w` 形状相同，初始化为 0。第一步 $v=g$，所以第一步和 SGD 完全一样，从第二步开始才体现惯性。

### 为什么 Momentum 能加速？

最典型的场景是一条**又窄又长的山谷**，即不同方向的曲率差很多（Hessian 的条件数大）：

- 横跨山谷的方向很陡：梯度大，而且每一步正负翻转，SGD 在两侧山壁之间来回弹（zig-zag）。
- 沿着谷底的方向很平：梯度小，但方向始终一致，SGD 走得很慢。

Momentum 的加权和正好对症。来回翻转的分量在求和时互相抵消，震荡被压下去；方向一致的分量不断累加，越滚越快。如果梯度一直是 $g$，速度最终稳定在

$$
v=\frac{g}{1-\beta}
$$

即有效步长放大到 $\eta/(1-\beta)$。$\beta=0.9$ 时放大 10 倍。

理论上，对二次函数，梯度下降需要的迭代次数正比于条件数 $\kappa$，调好参数的 Momentum 只需要正比于 $\sqrt\kappa$（Polyak, 1964）。Distill 的 [Why Momentum Really Works](https://distill.pub/2017/momentum/) 有交互式的可视化。

### 什么情况下会震荡？

拿最简单的一维二次函数 $f(x)=\tfrac12\lambda x^2$ 看，$\lambda$ 是曲率，梯度是 $\lambda x$。SGD 的一步变成：

$$
x_{t+1}=(1-\eta\lambda)\,x_t
$$

- $\eta\lambda<1$：每步按比例缩小，平稳收敛。
- $1<\eta\lambda<2$：$1-\eta\lambda$ 是负数，$x$ **每步换号**，在最低点两侧来回跳，幅度逐步缩小。
- $\eta\lambda>2$：每步幅度放大，**发散**。

所以学习率的安全上限是 $2/\lambda_{\max}$，由最陡的方向决定。山谷里的 zig-zag 就是这么来的：学习率受最陡方向限制不能再大，最陡方向已经 $\eta\lambda>1$ 在来回跳，最平方向的 $\eta\lambda$ 却很小，走得很慢。

Momentum 自己也会震荡。$\beta$ 太大时惯性太强，小球冲过最低点，要来回摆很多次才停。在这个二次函数上可以算出，摆动幅度每步只衰减为原来的 $\sqrt\beta$ 倍：$\beta=0.9$ 时约 90 步衰减到 1%，$\beta=0.99$ 时要约 900 步。

第三种震荡来自 mini-batch 噪声：到了最低点附近，loss 不再下降，而是上下抖。解决办法是学习率衰减（step decay、cosine decay）或者加大 batch。

### 学习率和 β 的直觉

- **学习率 $\eta$ 是步长。** 太小走不动，太大在最陡方向来回跳甚至发散。调参时按对数尺度试（0.1、0.01、0.001），训练中配合 warmup 和衰减。
- **动量 $\beta$ 是记忆长度。** 它大致相当于对最近 $1/(1-\beta)$ 步的梯度做平均：0.9 约 10 步，0.99 约 100 步。
- **两者要一起调。** 有效步长是 $\eta/(1-\beta)$。把 $\beta$ 从 0.9 提到 0.99，有效步长放大 10 倍，$\eta$ 通常要相应调小。

### 面试版实现（仿 PyTorch 接口）

```python
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.params = params                    # 参数数组的列表，step 里会原地修改
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(p) for p in params]   # 每个参数一个速度，形状相同

    def step(self, grads):
        for p, g, v in zip(self.params, grads, self.velocities):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p   # L2 正则的梯度就是 lambda * w
            if self.momentum > 0:
                v *= self.momentum              # 1. 衰减旧速度（原地改，self.velocities 才记得住）
                v += g                          # 2. 加上当前梯度
                # Nesterov 变体（PyTorch 的写法，见下方追问）
                g = g + self.momentum * v if self.nesterov else v
            p -= self.lr * g                    # 3. 原地更新：模型里引用的是同一个数组
```

### Example：在窄山谷里比一比

```python
def ravine_grad(w):
    # f(x, y) = 0.5 * (x^2 + 100 * y^2)：y 方向比 x 方向陡 100 倍
    return np.array([1.0, 100.0]) * w


def steps_to_converge(use_momentum, lr=0.018, momentum=0.9, tol=1e-3, max_steps=5000):
    w = np.array([10.0, 1.0])
    v = np.zeros_like(w)
    for t in range(1, max_steps + 1):
        g = ravine_grad(w)
        if use_momentum:
            w, v = momentum_step(w, g, v, lr, momentum)
        else:
            w = sgd_step(w, g, lr)
        if np.linalg.norm(w) < tol:         # 离最低点 (0, 0) 足够近就停
            return t
    return max_steps


print("GD      :", steps_to_converge(False))   # GD      : 508
print("Momentum:", steps_to_converge(True))    # Momentum: 138
```

这里的梯度是精确的，所以严格说是 GD。$y$ 方向 $\eta\lambda=0.018\times100=1.8$，落在 1 到 2 之间：GD 的 $y$ 坐标依次是 $1,-0.8,0.64,-0.512,\ldots$，每步换号。$x$ 方向 $\eta\lambda=0.018$，每步只缩小 1.8%，所以 GD 需要 508 步。同样的学习率加上 $\beta=0.9$ 的 Momentum，只要 138 步。

### 延伸：Adam

问完 Momentum，面试官常接着问 Adam。Adam 等于 Momentum（一阶矩：梯度的滑动平均）加 RMSProp（二阶矩：梯度平方的滑动平均，给每个参数单独定步长），再加偏差修正：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)\,g_t,\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^2
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

注意这里的 $v$ 是梯度平方的平均，和 Momentum 的速度 $v$ 含义不同。

```python
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]   # 一阶矩：管方向
        self.v = [np.zeros_like(p) for p in params]   # 二阶矩：管每个参数的步长
        self.t = 0                                    # 步数，偏差修正要用

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            # m、v 从 0 起步，前几步被拉向 0，除以 (1 - beta^t) 把偏差修回来
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

偏差修正的直觉：第一步 $m_1=(1-\beta_1)g_1=0.1\,g_1$，明显偏小；除以 $1-\beta_1^1=0.1$ 正好还原成 $g_1$。

### 关键追问

- **Momentum 为什么能加速收敛？** 方向一致的梯度分量累加，有效步长变成 $\eta/(1-\beta)$；来回翻转的分量互相抵消。在病态（条件数大）的问题上效果最明显。
- **什么情况下会震荡？** 三种：学习率超过 $1/\lambda_{\max}$ 开始换号来回跳，超过 $2/\lambda_{\max}$ 发散；$\beta$ 太大，冲过头来回摆；mini-batch 噪声让参数在最低点附近抖，靠学习率衰减解决。
- **为什么 `v *= momentum` 必须原地写？** 写成 `v = momentum * v + g` 会新建一个数组，`self.velocities` 里存的速度永远是 0，Momentum 悄悄退化成 SGD，而且不报错。`p -= ...` 同理，必须原地改模型持有的那个数组。
- **Momentum 有好几种写法，等价吗？** PyTorch 写 $v\leftarrow\beta v+g,\ \theta\leftarrow\theta-\eta v$；CS231n 写 $v\leftarrow\beta v-\eta g,\ \theta\leftarrow\theta+v$。学习率固定时两者完全等价，后者的 $v$ 就是前者的 $-\eta v$。吴恩达课上的写法 $v\leftarrow\beta v+(1-\beta)g$ 多乘了 $(1-\beta)$，等价于把学习率换成 $\eta(1-\beta)$，换写法时学习率要跟着换。
- **Nesterov 有什么不同？** 普通 Momentum 在当前位置算梯度。Nesterov 先按惯性往前看一步，在预估的新位置算梯度，冲过头时能更早刹车。PyTorch 把参数直接存在「往前看一步」的位置上，推导后更新量变成 $g+\beta v$，不用额外算一次梯度。
- **SGD 和 Adam 怎么选？** Adam 给每个参数自适应步长，对学习率不太敏感、收敛快，Transformer 和 LLM 基本都用它的变体 AdamW。SGD + Momentum 在一些 CV 任务上泛化更好（Wilson et al., 2017, *The Marginal Value of Adaptive Gradient Methods in Machine Learning*），而且省显存：Adam 每个参数要多存 $m$、$v$ 两份状态，SGD + Momentum 只多存一份速度。
- **AdamW 改了什么？** Adam 把 L2 正则加进梯度后，正则项也会被 $\sqrt{\hat v}$ 除掉，梯度大的参数被衰减得少。AdamW 把 weight decay 从梯度里拿出来，单独做一步 `p -= lr * wd * p`。

---

## 11. NumPy 手写神经网络层

### 原理：每层只做两件事

反向传播就是链式法则。把网络拆成一层一层，每层只实现两个函数：

- `forward(x)`：算输出，并把 backward 要用的东西**缓存**起来。
- `backward(dout)`：拿到上游传回来的梯度 $\partial\mathcal L/\partial\text{out}$，算出参数梯度，并返回 $\partial\mathcal L/\partial x$ 继续往前传。

写 backward 时最有用的一条规则：**梯度和它对应的变量形状完全相同。** $W$ 是 $(D_{in},D_{out})$，$dW$ 就必须是 $(D_{in},D_{out})$。很多时候不用推导，把形状凑对就能写出正确的矩阵乘法。面试时边写边把每一步的形状念出来。

### Linear 层

前向：

$$
Y=XW+b,\qquad X:(N,D_{in}),\quad W:(D_{in},D_{out}),\quad b:(D_{out},),\quad Y:(N,D_{out})
$$

反向（$dY$ 是上游传来的梯度，形状 $(N,D_{out})$）：

$$
dX=dY\,W^\top,\qquad dW=X^\top dY,\qquad db=\sum_{i=1}^{N}dY_{i,:}
$$

用凑形状的方法检查一遍：

- $dW$ 要 $(D_{in},D_{out})$。手上有 $X:(N,D_{in})$ 和 $dY:(N,D_{out})$，唯一能凑出来的是 $X^\top dY$：$(D_{in},N)\times(N,D_{out})$。
- $dX$ 要 $(N,D_{in})$，只能是 $dY\,W^\top$：$(N,D_{out})\times(D_{out},D_{in})$。
- $db$ 要 $(D_{out},)$。前向时 $b$ 被**广播**加到了 $N$ 行上，相当于用了 $N$ 次，反向就要把 $N$ 行的梯度**加起来**。规则是：前向广播，反向求和。

想严格推导也只要一行。因为 $Y_{ij}=\sum_k X_{ik}W_{kj}+b_j$，所以

$$
\frac{\partial\mathcal L}{\partial W_{kj}}=\sum_i\frac{\partial\mathcal L}{\partial Y_{ij}}\,X_{ik}=(X^\top dY)_{kj}
$$

```python
import numpy as np


def linear_forward(X, W, b):
    out = X @ W + b                # (N, D_in) @ (D_in, D_out) + (D_out,) -> (N, D_out)
    cache = (X, W)                 # backward 要用 X 算 dW，用 W 算 dX
    return out, cache


def linear_backward(dout, cache):
    X, W = cache
    dX = dout @ W.T                # (N, D_out) @ (D_out, D_in) -> (N, D_in)
    dW = X.T @ dout                # (D_in, N) @ (N, D_out) -> (D_in, D_out)
    db = dout.sum(axis=0)          # (N, D_out) -> (D_out,)：前向广播，反向求和
    return dX, dW, db
```

### ReLU

前向是 $\max(0,x)$。反向时，前向被截成 0 的位置梯度也是 0，其余位置原样传回：

```python
def relu_forward(x):
    return np.maximum(0, x), x     # 缓存输入：backward 要知道哪些位置 > 0


def relu_backward(dout, x):
    return dout * (x > 0)          # 形状不变：(N, D) -> (N, D)
```

### Softmax + Cross-Entropy 的反向

第 4 节提过结论，这里补上推导：

$$
\frac{\partial\mathcal L}{\partial z}=\frac{1}{N}\big(p-\text{onehot}(y)\big),\qquad p=\operatorname{softmax}(z)
$$

走 log-softmax 推导最省事。单个样本的 loss 是

$$
\mathcal L=-\log p_y=-z_y+\log\sum_j e^{z_j}
$$

对 $z_k$ 求导：第一项只在 $k=y$ 时贡献 $-1$；第二项的导数是 $e^{z_k}/\sum_j e^{z_j}$，正好是 $p_k$。合起来就是 $p_k-\mathbf{1}[k=y]$。batch 上取了平均，梯度再除以 $N$。

如果面试官要求走 softmax 的 Jacobian：$\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$，乘上交叉熵的 $\partial\mathcal L/\partial p_i=-\mathbf{1}[i=y]/p_i$ 再对 $i$ 求和，结果相同，只是步骤多一些。

```python
def softmax_cross_entropy(logits, y):
    """
    logits: (N, C) 未经 softmax 的分数
    y:      (N,) 整数标签
    返回 loss（标量）和 dlogits（N, C）
    """
    N = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)                      # 稳定化
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))  # log-softmax
    loss = -log_probs[np.arange(N), y].mean()

    dlogits = np.exp(log_probs)            # 1. 拿到概率 p，(N, C)；exp 生成新数组，可以放心原地改
    dlogits[np.arange(N), y] -= 1          # 2. 每行真实类别那一列减 1：p - onehot
    dlogits /= N                           # 3. loss 对 batch 取了平均，梯度也要除以 N
    return loss, dlogits
```

### 梯度检查：怎么证明 backward 写对了

用数值微分算一遍梯度，和解析梯度比。中心差分比单边差分准得多：

$$
\frac{\partial f}{\partial x}\approx\frac{f(x+h)-f(x-h)}{2h}
$$

```python
def numerical_grad(f, x, h=1e-5):
    """f: 无参函数，返回标量 loss；x: 参数数组，会被临时扰动"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        old = x.flat[i]
        x.flat[i] = old + h
        f_plus = f()
        x.flat[i] = old - h
        f_minus = f()
        x.flat[i] = old                    # 一定要还原，否则后面的参数全错
        grad.flat[i] = (f_plus - f_minus) / (2 * h)
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))
```

检查 Linear 层时有个小技巧：令 $\mathcal L=\sum(\text{out}\odot dY)$，则 $\partial\mathcal L/\partial\text{out}$ 恰好等于 $dY$，这样就能检查任意一个上游梯度。

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((4, 3))
W = rng.standard_normal((3, 5))
b = rng.standard_normal(5)
dout = rng.standard_normal((4, 5))           # 随便造一个上游梯度

loss_fn = lambda: np.sum(linear_forward(X, W, b)[0] * dout)

_, cache = linear_forward(X, W, b)
dX, dW, db = linear_backward(dout, cache)
print(rel_error(dX, numerical_grad(loss_fn, X)))   # 三个结果都在 1e-8 以下，说明写对了
print(rel_error(dW, numerical_grad(loss_fn, W)))
print(rel_error(db, numerical_grad(loss_fn, b)))
```

经验值（float64）：相对误差小于 $10^{-7}$ 基本没问题，大于 $10^{-3}$ 几乎一定有 bug。ReLU 在 0 点不可导，扰动刚好跨过 0 时数值梯度会不准，所以检查时用随机输入，不要用整数。

### 拼起来：两层 MLP + Momentum

把上面几层串成 Linear → ReLU → Linear → Softmax CE，用第 10 节的 `SGD` 训练。前向从左往右，反向从右往左，每层把梯度交给前一层。

```python
def train_mlp(X, y, hidden=16, num_classes=2, epochs=100, batch_size=32,
              lr=0.1, momentum=0.9, seed=0):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    # He 初始化：配合 ReLU，让每层输出的方差大致不变
    W1 = rng.standard_normal((D, hidden)) * np.sqrt(2.0 / D)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, num_classes)) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(num_classes)
    opt = SGD([W1, b1, W2, b2], lr=lr, momentum=momentum)    # 第 10 节的优化器

    N = X.shape[0]
    for epoch in range(epochs):
        perm = rng.permutation(N)                    # 每个 epoch 打乱一次
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]     # 取一个 mini-batch
            xb, yb = X[idx], y[idx]

            # 前向：(B, D) -> (B, hidden) -> (B, hidden) -> (B, C)
            h, cache1 = linear_forward(xb, W1, b1)
            a, relu_cache = relu_forward(h)
            logits, cache2 = linear_forward(a, W2, b2)
            loss, dlogits = softmax_cross_entropy(logits, yb)

            # 反向：倒着走一遍
            da, dW2, db2 = linear_backward(dlogits, cache2)
            dh = relu_backward(da, relu_cache)
            _, dW1, db1 = linear_backward(dh, cache1)

            opt.step([dW1, db1, dW2, db2])           # 顺序要和参数列表一一对应
    return W1, b1, W2, b2


rng = np.random.default_rng(42)
X = rng.standard_normal((400, 2))
y = (X[:, 0] * X[:, 1] > 0).astype(int)      # 一三象限为 1，二四象限为 0：一条直线分不开，逻辑回归只有 0.59

W1, b1, W2, b2 = train_mlp(X, y)
logits = np.maximum(0, X @ W1 + b1) @ W2 + b2
print("train accuracy:", (logits.argmax(axis=1) == y).mean())   # train accuracy: 0.9925
```

### 关键追问

- **为什么 `db` 要 `sum(axis=0)`？** 前向时同一个 $b$ 被广播加到 $N$ 个样本上，相当于用了 $N$ 次，每次使用都贡献一份梯度，反向要全部加起来。
- **为什么 forward 要缓存输入？** $dW=X^\top dY$ 需要 $X$；ReLU 的反向要知道哪里大于 0。这也是训练比推理费显存的原因：每层的激活都要留到反向用完。
- **常见 bug 有哪些？** softmax CE 的梯度忘了除以 $N$，等于把学习率放大了 $N$ 倍。直接在 `probs` 上原地改 `probs[range(N), y] -= 1`，把前向算好的概率也改坏了，要先 `.copy()`。参数梯度和参数对不上号，比如把 `db1` 传给了 `W1`。
- **权重为什么不能全初始化为 0？** 同一层所有神经元会算出一样的输出、拿到一样的梯度，永远一样，等于只有一个神经元（对称性问题）。偏置初始化为 0 没关系。
- **为什么用 He 初始化？** ReLU 会把一半的输入截成 0，输出方差减半；权重方差取 $2/D_{in}$ 正好补回来，深层网络的信号才不会逐层衰减。tanh、sigmoid 配 Xavier 初始化（方差 $2/(D_{in}+D_{out})$）。

---

## 11.1 BatchNorm

### 原理与直觉

BatchNorm 对每个特征，用**当前 batch** 的均值和方差做标准化，再乘可学习的 $\gamma$、加 $\beta$：

$$
\mu=\frac1N\sum_{i=1}^{N}x_i,\qquad
\sigma^2=\frac1N\sum_{i=1}^{N}(x_i-\mu)^2,\qquad
\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
y=\gamma\,\hat x+\beta
$$

公式和 6.2 节的 LayerNorm 完全一样，区别只在统计量沿哪个轴算：BatchNorm 沿 **batch 维**（`axis=0`，每个特征跨样本求均值），LayerNorm 沿**特征维**（`axis=-1`，每个样本跨特征求均值）。

面经里最常考的是 **训练和推理的行为不同**：

- **训练**：用当前 batch 的 $\mu,\sigma^2$；同时用滑动平均累积 `running_mean`、`running_var`。
- **推理**：batch 可能只有 1 条，统计量没有意义，所以改用训练时累积好的 `running_mean`、`running_var`。

### 先写核心版（只有训练前向）

```python
import numpy as np


def batchnorm_forward_train(x, gamma, beta, eps=1e-5):
    # x: (N, D)；gamma, beta: (D,)
    mu = x.mean(axis=0)                    # (D,) 每个特征一个均值：沿 batch 维
    var = x.var(axis=0)                    # (D,) 有偏方差，除以 N
    x_hat = (x - mu) / np.sqrt(var + eps)  # (N, D) 标准化
    return gamma * x_hat + beta            # (N, D) 缩放平移，gamma、beta 广播到每一行
```

### 面试版实现（含 running 统计量和 backward）

```python
class BatchNorm1d:
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(dim)                 # 初始化成恒等变换：y = x_hat
        self.beta = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.momentum = momentum                  # PyTorch 约定：新 batch 的统计量占 0.1
        self.eps = eps

    def forward(self, x, training=True):
        if training:
            mu = x.mean(axis=0)                   # (D,)
            var = x.var(axis=0)                   # (D,)
            # 滑动平均，留给推理用
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mu, var = self.running_mean, self.running_var    # 推理：用累积的统计量

        self.inv_std = 1.0 / np.sqrt(var + self.eps)         # (D,)
        self.x_hat = (x - mu) * self.inv_std                 # (N, D)，backward 要用
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        """只适用于 training=True 的前向：这时 mu 和 var 也是 x 的函数"""
        N = dout.shape[0]
        self.dgamma = (dout * self.x_hat).sum(axis=0)        # (D,)
        self.dbeta = dout.sum(axis=0)                        # (D,) 前向广播，反向求和
        dx_hat = dout * self.gamma                           # (N, D)
        # 化简后的结果，三项分别来自：x 直接的路径、经过 mu 的路径、经过 var 的路径
        dx = self.inv_std / N * (
            N * dx_hat - dx_hat.sum(axis=0) - self.x_hat * (dx_hat * self.x_hat).sum(axis=0)
        )
        return dx                                            # (N, D)
```

backward 的完整推导比较长，面试一般只要求 forward。backward 记住结论，并且会用上一节的梯度检查验证：

```python
rng = np.random.default_rng(0)
x = rng.standard_normal((8, 4)) * 3 + 1
dout = rng.standard_normal((8, 4))
bn = BatchNorm1d(4)
bn.gamma = rng.standard_normal(4)
bn.beta = rng.standard_normal(4)

bn.forward(x)
dx = bn.backward(dout)
loss_fn = lambda: np.sum(bn.forward(x) * dout)
print(rel_error(dx, numerical_grad(loss_fn, x)))                # 约 2e-8，低于 1e-7 的经验线
print(rel_error(bn.dgamma, numerical_grad(loss_fn, bn.gamma)))
```

### 关键追问

- **BatchNorm 沿哪个轴？** 全连接层的输入是 $(N,D)$，沿 `axis=0`。卷积层的输入是 $(N,C,H,W)$，每个通道一组统计量，沿 `axis=(0, 2, 3)`。
- **训练和推理有什么不同？** 训练用当前 batch 的统计量，推理用 running 统计量。PyTorch 里靠 `model.train()` 和 `model.eval()` 切换，推理前忘了调 `eval()` 是经典 bug：结果会随 batch 里其他样本变化。
- **为什么要 $\gamma$ 和 $\beta$？** 强行标准化会限制表达能力，比如 sigmoid 前的输入被压到 0 附近，只剩近似线性的一段。有了 $\gamma,\beta$，网络可以学回需要的尺度和偏移；$\gamma=\sqrt{\sigma^2+\epsilon}$、$\beta=\mu$ 时完全还原原始输入。
- **running_var 用有偏还是无偏方差？** 标准化当前 batch 用有偏方差（除以 $N$）。PyTorch 更新 `running_var` 时用的是无偏方差（除以 $N-1$），上面的实现为了简单没有区分。
- **batch 很小时会怎样？** 统计量噪声很大，效果明显变差。batch 为 1 时方差恒为 0，输出全是 $\beta$，PyTorch 在训练模式下会直接报错。这时改用 LayerNorm 或 GroupNorm，它们不依赖 batch。
- **BatchNorm 为什么有效？** 原论文的解释是减少 internal covariate shift。后来的研究（Santurkar et al., 2018, *How Does Batch Normalization Help Optimization?*）认为主要原因是让 loss 地形更平滑，允许用更大的学习率。

---

## 12. Top-K 问题（堆）

这一节是算法题：从 $n$ 个元素里找最大的 $k$ 个。它和 6.4 节解码时的 Top-K 采样是两个不同的问题。

### 原理与直觉

维护一个**大小为 $k$ 的最小堆**。堆顶是当前 top-k 里最小的那个，可以把它看作**入围门槛**：

- 堆没满：新元素直接进堆。
- 堆满了：新元素比门槛大，就把门槛踢出去，新元素进堆；否则直接丢掉。

扫完一遍，堆里剩下的就是最大的 $k$ 个。

为什么找最大的 $k$ 个反而用最小堆？因为每来一个新元素，要和 top-k 里**最弱**的那个比。最小堆把最弱的放在堆顶，查看是 $O(1)$，替换是 $O(\log k)$。

### 先写核心版

```python
import heapq


def top_k(nums, k):
    heap = []                                  # 最小堆，最多 k 个元素；heap[0] 是门槛
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)            # 1. 没满：直接放
        elif x > heap[0]:
            heapq.heapreplace(heap, x)         # 2. 比门槛大：弹出门槛、放进 x，一次完成
    return sorted(heap, reverse=True)          # 3. 最后把 k 个排一下序，O(k log k)
```

`heapq.heapreplace` 等于先 `heappop` 再 `heappush`，但只做一次调整，更快。Python 自带的 `heapq.nlargest(k, nums)` 做的是同一件事，面试时要会手写，也要知道有现成的。

### 复杂度：为什么不全排序

| 方法                     | 时间                                   | 内存里要放多少数据 | 能处理数据流吗       |
| ------------------------ | -------------------------------------- | ------------------ | -------------------- |
| `sorted(nums)[-k:]`      | $O(n\log n)$                           | 全部 $n$ 个        | 不能，要先拿到全部数据 |
| 大小为 $k$ 的最小堆      | $O(n\log k)$                           | $k$ 个             | 能，遍历一次         |
| 快速选择（quickselect）  | 平均 $O(n)$，最坏 $O(n^2)$             | 全部 $n$ 个        | 不能                 |

$n$ 个元素每个最多做一次堆操作，每次 $O(\log k)$，合计 $O(n\log k)$。$k$ 远小于 $n$ 时，$\log k$ 比 $\log n$ 小很多。更关键的是内存：堆只占 $O(k)$，数据可以边读边丢，$n$ 是十亿条日志也没问题。全排序还把我们不关心的 $n-k$ 个元素也排好了，这部分工作全是浪费。

### 变体 1：数据流中的 Top-K

数据一条条到来，随时要能查询当前的 top-k。和上面的逻辑完全一样，只是拆成 `add` 和查询两个方法（LeetCode 703 是它的简化版）：

```python
class StreamTopK:
    def __init__(self, k):
        self.k = k
        self.heap = []

    def add(self, x):                          # 每条 O(log k)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, x)
        elif x > self.heap[0]:
            heapq.heapreplace(self.heap, x)

    def kth_largest(self):                     # 第 k 大就是门槛，O(1)；调用前要已有 k 个元素
        return self.heap[0]

    def top(self):                             # 有序的 top-k，O(k log k)
        return sorted(self.heap, reverse=True)
```

### 变体 2：带权重（按分数）的 Top-K

ML 场景里的元素通常带一个分数：召回出的 item 有模型打分，商品有点击次数，词有词频。做法是把 `(score, ...)` 元组放进堆，元组按第一项比较。

```python
def top_k_by_score(items, k):
    """items: 可迭代的 (item, score)；返回分数最高的 k 个 (item, score)"""
    heap = []
    for i, (item, score) in enumerate(items):
        entry = (score, i, item)               # 中间放序号 i，平局时比 i，永远比不到 item
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [(item, score) for score, _, item in sorted(heap, reverse=True)]
```

为什么要塞一个序号 `i`：元组比较时，第一项相同会接着比第二项。如果第二项是 dict 这种不支持 `<` 的对象，`heapq` 会直接抛 `TypeError`。中间放一个唯一的序号，平局时比序号就分出了大小。

高频元素 Top-K（LeetCode 347）就是先计数、再按次数取 top-k：

```python
from collections import Counter


def top_k_frequent(words, k):
    counts = Counter(words)                     # O(n) 计数
    return top_k_by_score(counts.items(), k)    # O(m log k)，m 是不同元素的个数
```

### Example

```python
print(top_k([5, 1, 9, 3, 7, 2, 8], k=3))                # [9, 8, 7]

stream = StreamTopK(k=2)
for x in [4, 1, 7, 3, 9]:
    stream.add(x)
print(stream.top(), stream.kth_largest())               # [9, 7] 7

print(top_k_frequent(["a", "b", "a", "c", "b", "a"], k=2))   # [('a', 3), ('b', 2)]
```

### 关键追问

- **为什么不用全排序？** 时间 $O(n\log n)$ 比 $O(n\log k)$ 多；更重要的是全排序要把 $n$ 个元素全放进内存，没法处理数据流，堆只要 $O(k)$。
- **找最小的 $k$ 个怎么办？** Python 的 `heapq` 只有最小堆。把元素取负再存进去，就变成了最大堆，门槛变成当前 $k$ 个里最大的那个。
- **数据分在多台机器上怎么办？** 每台机器先算本地 top-k，再把这些结果合并，求一次 top-k。这样做是对的：全局 top-k 里的任何一个元素，在它所在的机器上一定也排进了本地前 $k$。网络传输量只有「机器数 × $k$」。
- **数据随机顺序到来时，堆真的要调整 $n$ 次吗？** 不用。第 $i$ 个元素进入当前 top-k 的概率是 $k/i$，总替换次数的期望约为 $k\ln(n/k)$。绝大多数元素和门槛比一次（$O(1)$）就被丢掉了。
- **数据全在内存里，能更快吗？** 能。快速选择平均 $O(n)$，NumPy 的 `np.argpartition` 就是这么做的（6.4 节和第 9 节 KNN 都用了它）。
- **`heapreplace` 和 `heappushpop` 有什么区别？** `heapreplace` 先弹再压，新元素一定进堆，所以前面要先判断 `x > heap[0]`。`heappushpop` 先压再弹，新元素如果不比堆顶大会被直接弹回来，可以省掉判断：堆满后每条只写 `heapq.heappushpop(heap, x)`。

---

## 13. 蓄水池抽样（Reservoir Sampling）

### 问题

数据流一条条到来，总长度 $n$ 事先**不知道**（可能大到存不下），只能遍历一次，只有 $O(k)$ 内存。要求最后留下 $k$ 个元素，并且**每个元素被留下的概率都是 $k/n$**。

推荐和广告组爱考它，因为日志天然就是这样的流。比如从一天的全量曝光日志里均匀抽 1 万条做人工标注：日志有多少条，要等当天结束才知道。

### 算法与直觉

想象一个容量为 $k$ 的水池：

1. 前 $k$ 个元素直接放进水池。
2. 第 $i$ 个元素（$i>k$）到来时，以 $k/i$ 的概率让它进池；如果进池，就从池里**等概率**挑一个踢出去。

直觉：越往后的元素，进池概率 $k/i$ 越小；越早进池的元素，要熬过的被踢轮次越多。这两个效应恰好抵消，所有元素最终的概率相同。

实现时用一个随机数同时决定「进不进」和「踢谁」：在 $[1,i]$ 里均匀随机挑一个 $j$，如果 $j\le k$，就用新元素替换第 $j$ 个位置。$j\le k$ 的概率正好是 $k/i$，而且 $j$ 在 $1..k$ 上是等概率的。

### 实现

```python
import random


def reservoir_sample(stream, k):
    reservoir = []
    for i, x in enumerate(stream, start=1):     # i 从 1 开始：x 是第 i 个元素
        if i <= k:
            reservoir.append(x)                 # 1. 前 k 个直接进池
        else:
            j = random.randint(1, i)            # 2. 在 [1, i] 里等概率挑一个（两端都包含）
            if j <= k:                          # 3. 概率 k/i 进池
                reservoir[j - 1] = x            #    同时踢掉第 j 个：池里每个位置被踢的概率相同
    return reservoir                            # 流比 k 短时，返回全部元素
```

$k=1$ 是最常见的特例（LeetCode 382、398）：第 $i$ 个元素以 $1/i$ 的概率替换当前选中的元素。

```python
def reservoir_sample_one(stream):
    chosen = None
    for i, x in enumerate(stream, start=1):
        if random.randint(1, i) == 1:           # 概率 1/i 换成当前元素
            chosen = x
    return chosen
```

### 证明（数学归纳法）

很多人代码能写，证明讲不清。下面这个归纳证明要能完整讲出来。

**命题**：处理完前 $i$ 个元素后（$i\ge k$），这 $i$ 个元素中的每一个在池里的概率都是 $k/i$。

**基础**：$i=k$ 时，前 $k$ 个元素全在池里，概率为 $1=k/k$。

**归纳**：假设处理完前 $i$ 个元素时命题成立，现在来了第 $i+1$ 个。

- **新元素**：按算法，进池概率就是 $\dfrac{k}{i+1}$。
- **老元素**（前 $i$ 个中的任意一个）：它最后在池里，需要两件事同时发生。
  1. 处理完前 $i$ 个时它在池里：概率 $\dfrac{k}{i}$（归纳假设）。
  2. 这一轮没被踢掉。被踢需要新元素进池（$\dfrac{k}{i+1}$），而且恰好挑中它的位置（$\dfrac1k$），所以被踢的概率是 $\dfrac{k}{i+1}\cdot\dfrac1k=\dfrac{1}{i+1}$，不被踢的概率是 $\dfrac{i}{i+1}$。

  这一轮的随机数和之前的过程无关，两个概率直接相乘：

$$
\frac{k}{i}\cdot\frac{i}{i+1}=\frac{k}{i+1}
$$

新老元素的概率都是 $k/(i+1)$，命题对 $i+1$ 也成立。归纳到 $i=n$，每个元素被留下的概率都是 $k/n$。$\blacksquare$

**也可以不用归纳，直接连乘。** 第 $m$ 个元素（$m>k$）进池的概率是 $k/m$；之后第 $t$ 个元素到来时，它不被踢的概率是 $1-\frac1t=\frac{t-1}{t}$。所以

$$
P(\text{第 }m\text{ 个被留下})=\frac{k}{m}\cdot\frac{m}{m+1}\cdot\frac{m+1}{m+2}\cdots\frac{n-1}{n}=\frac{k}{n}
$$

中间项全部约掉。前 $k$ 个元素（$m\le k$）进池概率是 1，连乘从 $t=k+1$ 开始，结果同样是 $\frac{k}{k+1}\cdots\frac{n-1}{n}=\frac kn$。

### Example：用模拟验证

```python
from collections import Counter

random.seed(0)
n, k, trials = 10, 3, 100_000
counts = Counter()
for _ in range(trials):
    counts.update(reservoir_sample(range(n), k))

print({x: round(counts[x] / trials, 3) for x in range(n)})   # 每个值都接近 k/n = 0.3
```

写完算法后跑一遍这样的模拟，是发现差一错误（off-by-one）最快的办法。

### 变体：带权重的蓄水池抽样

每个元素带一个权重 $w>0$，希望权重越大越容易被抽中。Efraimidis 和 Spirakis（2006）的 A-Res 算法：给每个元素算一个随机 key：

$$
\text{key}=u^{1/w},\qquad u\sim\text{Uniform}(0,1)
$$

然后留下 key 最大的 $k$ 个。这一步正好是上一节的数据流 Top-K。

```python
import heapq


def weighted_reservoir_sample(stream, k):
    """stream 产出 (item, weight)，weight > 0"""
    heap = []                                   # 最小堆，存 (key, item)；堆顶是门槛
    for item, w in stream:
        key = random.random() ** (1.0 / w)      # u^(1/w)：权重越大，key 越接近 1
        if len(heap) < k:
            heapq.heappush(heap, (key, item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, item))
    return [item for _, item in heap]
```

$k=1$ 时，每个元素被选中的概率恰好是 $w_i/\sum_j w_j$。$k>1$ 时，结果等价于按权重**不放回**地依次抽 $k$ 次。

### 关键追问

- **为什么不先数出 $n$ 再随机抽？** 流的长度事先未知，数据也可能大到存不下。蓄水池只要遍历一次、$O(k)$ 内存。如果数据已经在内存里、$n$ 已知，直接 `random.sample(data, k)` 就行。
- **复杂度？** 时间 $O(n)$，每个元素一个随机数；空间 $O(k)$。$n$ 远大于 $k$ 时，绝大多数随机数都没用上。Li（1994）的 Algorithm L 用几何分布直接算出下一次替换要跳过多少个元素，复杂度降到 $O\big(k(1+\log(n/k))\big)$。
- **常见写错的地方？** 把 `randint(1, i)` 写成 `randint(1, i - 1)`，或者 0-based、1-based 下标混用，新元素的进池概率就不再是 $k/i$，结果会偏向流的某一段。
- **数据分在多台机器上怎么办？** 给每个元素分配一个随机 key $u\sim U(0,1)$，每台机器保留 key 最大的 $k$ 个，汇总后再取全局最大的 $k$ 个。所有 key 独立同分布，所以哪 $k$ 个 key 最大是等概率的。这和上一节分布式 Top-K 是同一个套路。
- **和 Top-K 是什么关系？** 蓄水池抽样可以看成「给每个元素一个随机分数，取分数最高的 $k$ 个」：不带权重时分数是 $u$，带权重时是 $u^{1/w}$。

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