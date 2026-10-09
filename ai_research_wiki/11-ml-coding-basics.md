# 11. ML Coding 基础

[返回目录](README.md)

本页是 ML Coding 的基础篇：先讲 CodeSignal 的题型和代码规范，再过一遍 NumPy、Pandas、PyTorch 的基本语法，最后是不许 import 时的纯 Python 矩阵运算。手写模型和算法题在 [12. ML Coding 专题](12-ml-coding.md)，正文里写作「专题第 N 节」。本页的节号是 0 到 0.4，专题页从第 1 节开始，两页的节号不重复。

---

## 0. CodeSignal 题型与代码规范

CodeSignal 上的 ML 面试有三种形态，代码写法跟着题型走，拿到题先判断是哪一种。下面的信息来自 CodeSignal 官方的测评白皮书和帮助文档（链接在本节末尾）。

| 题型 | 长什么样 | 代码怎么写 |
| --- | --- | --- |
| 单函数题（GCA、ML Core 测评） | 一个固定的入口函数，题面给样例 | 实现 `solution(...)`，用 `return` 返回答案 |
| 多文件工程题（Industry ML、ICA） | 有 `solution.py`、`data/`、`tests/` 等文件 | 按给定的函数和类签名填实现，跑 `unittest` |
| 现场面试（CodeSignal Interview） | 面试官在线看你写，可能是上面任意一种，也可能是共享的 Jupyter Notebook | 面试官定规则；能跑、讲清楚最重要 |

### 单函数题的规则

- 入口函数叫 `solution`，**不能改名**：测试框架按名字调用它，改名会直接报错。辅助函数写在模块顶层，和 `solution` 并列。
- 答案用 `return` 返回。`print` 只用来调试，输出显示在每个测试用例下面，不算答案。
- 能看到的只有 sample tests，点 Submit 才跑全部测试（含 hidden tests，往往更严格），按通过的测试数给分。空输入、只有一个元素、全部相同的值、超时，都是常见的失分点。
- 每道题有执行时间限制，写在题面的 Input/Output 部分，复杂度要心里有数。
- 可以反复 Submit，每道题按得分最高的那次算。官方建议早交、常交。

### ML Core 测评：纯 Python、不许调库

白皮书描述的 Machine Learning Engineering Core 测评是 70 分钟、三个模块：6 道考 ML 基础概念的场景题；1 道数据处理编程题，约 15~20 行；2 道 ML 算法实现题，每题约 20~25 分钟、15~25 行。候选人反馈的实际题量会有出入。算法题的范围包括 kNN、k-Means、决策树、GMM、矩阵归一化、Bagging、前向传播，白皮书明确写了**不考需要 sklearn、PyTorch、TensorFlow 的内容**，官方 kNN 样题更是要求「不 import 任何库」。输入是 Python 的 list，返回值也是 list，基本功见 0.4 节。

官方样题给的是骨架代码：辅助函数只有 `# implement this` 和 `pass`，`solution` 和其余胶水代码已经写好，**不要改动**，只填空。示意如下（以真实题面为准；填好的 kNN 见专题第 3 节）：

```python
def euc_dist(value1, value2):
    # implement this
    pass

def solution(train_data, test_data, k):
    final_labels = list()
    ...                          # 题目给好的胶水代码：调用辅助函数、多数投票，不要改
    return final_labels
```

填空前先读懂胶水代码怎么调用辅助函数，比如传进来的一行含不含标签。没读懂给定代码的接口约定，是填空题最容易丢分的地方。

### 多文件工程题与现场面试

- 多文件题的可见测试故意不完整，评分用更严格的 hidden tests，所以要按题面的 Acceptance Criteria 写。类和方法的签名不要改，ICA 题面原话是 *do not change the existing method signatures*。样题参考答案用了类型标注、小而单一职责的函数，以及 `main()` 加 `if __name__ == "__main__": main()` 的入口。
- 现场面试可能打开共享的 Jupyter Notebook，官方说预装了「几百个包」，缺的在终端里 `pip3 install --user 包名`，装完重启 kernel。另有一个 Ubuntu 终端可以跑代码。官方文档没有提到 GPU，按 CPU、小数据写。
- 官方没有公布 Python 和各个库的版本。开场先跑 `import sys, numpy as np; print(sys.version.split()[0], np.__version__)`，再试 `import torch`、`import pandas`，失败就问面试官能否 pip 安装。

### 两页代码统一的规范

- 兼容 Python 3.8：类型标注用 `typing` 里的 `List`、`Optional`、`Tuple`、`Dict`，不写 `list[int]`、`int | None`。
- 函数带类型标注和简短 docstring，docstring 写清输入输出的形状，例如 `x: (B, T, C)`。
- 纯 Python 题用 `solution(...)` 做入口，不 import 任何库。
- PyTorch 模块继承 `nn.Module`，构造函数第一行写 `super().__init__()`（CodeSignal Learn 课程里的 `super(ClassName, self).__init__()` 是 Python 2 时代的写法，效果相同）；注意力里的线性层叫 `w_q`、`w_k`、`w_v`、`w_o`；mask 用 `masked_fill(mask == 0, -1e9)`，1 表示可见。
- 训练循环按 `model.train()`、`optimizer.zero_grad()`、前向、`loss.backward()`、`optimizer.step()` 的顺序写；验证时 `model.eval()` 加 `with torch.no_grad():`，日志用 f-string。
- 每节有自测：固定随机种子，`assert` 形状和数值，能和库函数对拍就对拍，最后打印 `all tests passed`，放在 `if __name__ == "__main__":` 下。

### 考前清单

- 在 `app.codesignal.com/assessments/practice` 用真实 IDE 练各种题型。练习题有可续期的 1 小时计时，公司看不到练习记录。
- 问清楚 recruiter 是哪种题型：ML Core 测评、多文件工程题，还是现场面试。三者的代码写法差别很大。
- 认证测评默认不允许用 AI，除非公司开启了 CodeSignal 的 AI 助手 Cosmo。能不能搜语法，以开考页面上的规则为准。
- 每道单函数题先看 README 或 Info 标签页里的 Language 一节，那里写着解释器版本和自动 import 的库。

### 关键追问

- **现场面试可以查 API 吗？** 先问面试官。有 ML coding 面试官公开说过不要求记住 API，可以申请查 torch 或 numpy 的文档；看重的是代码能跑、形状讲得清楚。

### 参考

- [Machine Learning Engineering Core 测评白皮书](https://discover.codesignal.com/rs/659-AFH-023/images/Machine-Learning-Engineering-Core-Skills-Evaluation-Framework-CodeSignal-Skills-Evaluation-Lab.pdf)
- [Machine Learning Engineering Industry Framework 2023](https://discover.codesignal.com/rs/659-AFH-023/images/Machine-Learning-Engineering-Industry-Framework-2023-Technical-Brief.pdf)
- [Using the README during your CodeSignal assessment](https://support.codesignal.com/hc/en-us/articles/1500000922402-Using-the-README-during-your-CodeSignal-assessment)
- [What is the CodeSignal Cloud IDE coding environment](https://support.codesignal.com/hc/en-us/articles/360039872914-What-is-the-CodeSignal-Cloud-IDE-Coding-Environment)
- [Using Jupyter Notebook for live data science interviews](https://support.codesignal.com/hc/en-us/articles/360050875793-Using-Jupyter-Notebook-for-live-data-science-interviews)
- [How do I practice coding questions on CodeSignal](https://support.codesignal.com/hc/en-us/articles/21025134150423-How-do-I-practice-coding-questions-on-CodeSignal)

---

## 0.1 NumPy 速查

代码块按顺序在同一个会话里运行，只在第一块 import。注释里的输出都是真实运行结果；面试时每一步都在注释里标出 shape。

### 创建、dtype 与随机数

一个数组只有一种 dtype，混进一个小数整个数组就变成 float64（PyTorch 默认 float32，见 0.3 节）；标签要存成整数才能当下标。随机数用 `default_rng`：它是独立的生成器对象，可以当参数传，不碰全局状态。

```python
import numpy as np

a, b = np.array([1, 2, 3]), np.array([1, 2.7, 3])   # (3,)、(3,)
print(a.dtype, b.dtype, b.astype(int))       # int64 float64 [1 2 3]，astype 向 0 截断，不四舍五入
print(np.zeros((2, 3)).shape, np.eye(2).tolist())   # (2, 3) [[1.0, 0.0], [0.0, 1.0]]，同类有 np.ones、np.full
print(np.arange(0, 1, 0.25), np.linspace(0, 1, 3))  # [0.   0.25 0.5  0.75] [0.  0.5 1. ]，arange 不含终点
probs, y = np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([1.0, 0.0])   # (N, C)、(N,) 的 float 标签
print(probs[np.arange(2), y.astype(int)])    # [0.9 0.8]，不转 int 直接当下标会报 IndexError
rng = np.random.default_rng(0)
print(rng.integers(0, 10, size=5), rng.normal(0.0, 1.0, size=(2, 3)).shape)   # [8 6 5 2 3] (2, 3)
print(rng.choice(5, size=3, replace=False), rng.permutation(5))   # [4 2 0] [2 1 3 4 0]，同一个 perm 打乱 X 和 y
print(rng.choice(3, size=5, p=[0.1, 0.2, 0.7]))   # [0 2 1 2 2]，按概率抽，K-Means++ 和采样解码都用它
```

### shape、reshape、新轴与拼接

```python
X = np.arange(6).reshape(2, 3)               # (2, 3)，按行填入：[[0 1 2] [3 4 5]]
print(X.reshape(3, -1).tolist(), X.T.tolist())   # [[0, 1], [2, 3], [4, 5]] [[0, 3], [1, 4], [2, 5]]，reshape 不是转置
x = np.arange(6)                             # (6,)
print(x[:, None].shape, x[None, :].shape, x[:, None].squeeze().shape)   # (6, 1) (1, 6) (6,)，None 插入长度 1 的轴
print(np.zeros((4, 2, 3)).transpose(0, 2, 1).shape)   # (4, 3, 2)，相当于 torch 的 transpose(-2, -1)
print(np.concatenate([X, X], axis=0).shape, np.concatenate([X, X], axis=1).shape)   # (4, 3) (2, 6)，沿已有的轴拼
print(np.stack([X, X]).shape, np.hstack([np.ones((2, 1)), X]).shape)   # (2, 2, 3) (2, 4)，后者是加一列 1 当截距
```

### 索引：切片、布尔掩码、花式索引

切片返回 view，和原数组共用内存。多个条件用 `&`、`|`、`~`，每个条件都加括号（`&` 的优先级比 `>` 高，`and` 不能用在数组上）。

```python
X = np.arange(12).reshape(3, 4)              # (3, 4)：[[0 1 2 3] [4 5 6 7] [8 9 10 11]]
print(X[1:, ::2].tolist())                   # [[4, 6], [8, 10]]
print(X[:, 0].shape, X[:, [0]].shape)        # (3,) (3, 1)，整数下标去掉这一维，列表保留
mask = (X[:, 0] > 3) & ~(X[:, 1] == 9)       # (3,)
print(mask, X[mask].shape, X[X % 5 == 0])    # [False  True False] (1, 4) [ 0  5 10]，二维掩码取出来是一维
y = np.array([2, 0, 3])                      # (3,)，每行要取的列号
print(X[np.arange(3), y])                    # [ 2  4 11]，第 i 行取第 y[i] 列，交叉熵取真实类别就这么写
print(X[:, y].shape)                         # (3, 3)，常见错写法：每行都取这 3 列，不报错
row = X[0, :2]                               # 切片是 view
row[0] = -1
print(X[0])                                  # [-1  1  2  3]，原数组也被改了
```

### axis、keepdims 与广播

`axis=k` 就是把第 k 维压掉，`keepdims=True` 把它留成长度 1。广播：两个 shape 右对齐，每一维要么相等要么有一个是 1，否则报错（`(2, 3) + (2,)` 就报错）。

```python
X = np.array([[1, 2, 3], [4, 5, 6]])         # (2, 3)
print(X.sum(axis=0), X.sum(axis=1), X.sum(axis=1, keepdims=True).shape)   # [5 7 9] [ 6 15] (2, 1)
print((X + np.array([10, 20, 30])).shape)    # (2, 3)，(2, 3) + (3,)：每行加同一个向量
y = np.array([1.0, 2.0, 3.0])                # (3,)
y_pred = np.array([[1.1], [1.9], [3.2]])     # (3, 1)，模型输出常见的形状
print((y_pred - y).shape)                    # (3, 3)，静默变成两两相减
print(np.mean((y_pred - y) ** 2).round(4), np.mean((y_pred.reshape(-1) - y) ** 2).round(4))   # 1.42 0.02
S = np.arange(1.0, 10.0).reshape(3, 3)       # (3, 3) 方阵，比如 (T, T) 的注意力分数
print((S / S.sum(axis=1)).sum(axis=1))       # [0.425 1.25  2.075]，漏了 keepdims：不报错，行和却不是 1
print((S / S.sum(axis=1, keepdims=True)).sum(axis=1))   # [1. 1. 1.]
```

### 归约、排序与计数

`argsort` 只有升序，降序对 `-scores` 排。只要前 k 大时用 `argpartition`（O(n)，这 k 个之间无序）。

```python
x = np.array([3, 1, 4, 1, 5])                # (5,)
print(x.sum(), x.mean(), x.max(), x.argmax())   # 14 2.8 5 4，argmax 并列时取第一个
print(x.std(), x.std(ddof=1), np.cumsum(x))  # 1.6 1.7888543819998317 [ 3  4  8  9 14]
logits = np.array([[0.1, 2.0, -1.0], [1.5, 0.3, 0.2]])   # (N, C)
print((logits.argmax(axis=1) == np.array([1, 2])).mean())   # 0.5，准确率
scores = np.array([0.2, 0.9, 0.1, 0.7, 0.5])   # (5,)
top = np.argpartition(scores, -2)[-2:]       # (2,)，最大的 2 个，顺序不保证
print(np.argsort(-scores), top[np.argsort(-scores[top])])   # [1 3 4 0 2] [1 3]
D = np.array([[3, 1, 2], [0, 5, 4]])         # (2, 3)，2 个查询点到 3 个样本的距离
idx = np.argsort(D, axis=1)[:, :2]           # (2, 2)，每行最近的 2 个
print(np.take_along_axis(D, idx, axis=1).tolist())   # [[1, 2], [0, 4]]，逐行按下标取值
values, counts = np.unique(np.array([2, 0, 2, 1, 2]), return_counts=True)
print(values, counts, values[counts.argmax()])   # [0 1 2] [1 1 3] 2，KNN 多数投票
```

### where、clip、maximum、矩阵乘

`np.maximum` 逐元素比大小（ReLU），`np.max` 是归约，`np.max(x, 0)` 的 0 是 axis。`*` 逐元素乘，`@` 矩阵乘。

```python
x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])   # (5,)
print(np.where(x > 0, x, 0.0), np.where(x > 0)[0])   # [0.  0.  0.  1.5 3. ] [3 4]，只传条件时返回下标
print(np.clip(x, -1.0, 2.0))                 # [-1.  -0.5  0.   1.5  2. ]，取 log 前写 np.clip(p, 1e-15, 1 - 1e-15)
print(np.maximum(x, 0), np.max(x))           # [0.  0.  0.  1.5 3. ] 3.0
A, B = np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])   # (2, 2)、(2, 2)
print((A * B).tolist(), (A @ B).tolist())    # [[5, 12], [21, 32]] [[19, 22], [43, 50]]
print(np.linalg.norm(np.array([[3.0, 4.0], [6.0, 8.0]]), axis=1))   # [ 5. 10.]，每行的 L2 范数
print(np.einsum("bqd,bkd->bqk", np.ones((2, 4, 8)), np.ones((2, 5, 8))).shape)   # (2, 4, 5)，即 Q @ K.transpose(0, 2, 1)
```

### 练习：按列 z-score

ML Core 的「矩阵归一化」：每列减均值、除以总体标准差（`ddof=0`），常数列输出 0。不许 import 时的纯 Python 版见 0.4 节。

```python
from typing import List


def solution(matrix: List[List[float]]) -> List[List[float]]:
    """按列 z-score，常数列输出 0。matrix: (n, d) -> (n, d)"""
    if not matrix:
        return []
    X = np.asarray(matrix, dtype=float)                  # 1. (n, d)，转 float 防整型截断
    const = X.max(axis=0) == X.min(axis=0)               # 2. (d,)，常数列（不用 std == 0，见 0.4 节）
    std = np.where(const, 1.0, X.std(axis=0))            # 3. (d,)，常数列分母换成 1
    return np.where(const, 0.0, (X - X.mean(axis=0)) / std).tolist()   # 4. (n, d)，转回 Python 类型


if __name__ == "__main__":
    out = solution([[1, 10, 5], [2, 20, 5], [3, 30, 5]])
    assert np.allclose(out, [[-1.2247, -1.2247, 0], [0, 0, 0], [1.2247, 1.2247, 0]], atol=1e-4)
    assert solution([[0.1], [0.1], [0.1]]) == [[0.0], [0.0], [0.0]] and solution([]) == []
    print("all tests passed")
```

### 常见坑

- **切片是 view，掩码和花式索引是副本。** 改切片会改原数组；链式的 `X[[0, 1]][:, 0] = 99` 改的是中间副本，`X` 不变。要独立的一份写 `.copy()`。
- **重复下标只生效一次。** `a[[0, 0, 1]] += 1` 里下标 0 只加了 1，按次数累加用 `np.add.at(a, idx, 1)` 或 `np.bincount`。
- **整型数组装不下小数。** 往 int 数组赋 2.7 会被静默截成 2，`c -= 0.5` 直接报错；输入先 `np.asarray(x, dtype=float)`。
- **数组不能直接 `if a == b:`，浮点也不能用 `==`。** 写 `np.array_equal`、`np.allclose`；`solution()` 返回前用 `.tolist()`、`float(x)` 转回 Python 类型。
- **标准差的默认值不同。** `np.std` 默认 `ddof=0`（除以 n），pandas 的 `.std()` 和 `torch.std` 默认除以 n-1；题目没说清就先问，或者用样例反推。

---

## 0.2 Pandas 速查

`DataFrame` 是带列名的二维表，每一列是一个 `Series`。按整列向量化，少写 for 循环：`apply(axis=1)` 逐行调 Python 函数，比向量化慢几个数量级。下面的块共用同一个 `df`，改数据前先 `.copy()`。

### 造数据、读 CSV、先看一眼

```python
import io

import numpy as np
import pandas as pd

df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "city": ["NY", "SF", "NY", "LA", "SF", "NY", "NY", "SF"],
    "age": [25, 32, np.nan, np.nan, 28, 35, 45, np.nan],
    "income": [50.0, 120.0, 65.0, np.nan, 95.0, 70.0, 80.0, 110.0],
    "bought": [0, 1, 0, 1, 1, 0, 1, 0],
})
print(df.shape, df.dtypes["age"], df.dtypes["city"])   # (8, 5) float64 object，有 NaN 的整数列变成 float
print(df.describe().loc["mean"].round(1).to_dict())    # {'user_id': 4.5, 'age': 33.0, 'income': 84.3, 'bought': 0.5}
print(df["city"].value_counts().to_dict())             # {'NY': 4, 'SF': 3, 'LA': 1}，再看 df.head()、df.info()
print(pd.read_csv(io.StringIO("a,b\n1,\n2,x\n")).isna().sum().to_dict())   # {'a': 0, 'b': 1}，空字段读成 NaN
```

### 选择：loc、iloc、布尔过滤

```python
age, sub = df["age"], df[["city", "age"]]       # 单括号是 Series (8,)，双括号是 DataFrame (8, 2)
print(df.loc[2, "city"], df.iloc[2, 1], len(df.loc[1:3]), len(df.iloc[1:3]))   # NY NY 3 2，loc 按标签且含右端
mask = (df["city"] == "NY") & (df["age"] > 30)  # 每个条件加括号，用 & | ~；和 NaN 比较一律 False
print(df.loc[mask, "user_id"].tolist())         # [6, 7]，一次 .loc 同时过滤行、选列
print(df.loc[~df["city"].isin(["SF", "LA"]), "user_id"].tolist())   # [1, 3, 6, 7]
print(df.loc[df["income"].between(60, 95), "user_id"].tolist())     # [3, 5, 6, 7]，两端都含
```

### 新增列与缺失值

```python
d = df.copy()
d["income_k"] = d["income"] / 1000                          # 整列运算，NaN 自动传播
d["is_senior"] = np.where(d["age"] >= 30, 1, 0)             # 向量化 if-else；NaN >= 30 是 False
d["region"] = d["city"].map({"NY": "east", "SF": "west"})   # 字典里没有的键变成 NaN
print(d["is_senior"].tolist(), d["region"].tolist()[:4])    # [0, 1, 0, 0, 0, 1, 1, 0] ['east', 'west', 'east', nan]
print(df.isna().sum().to_dict())   # {'user_id': 0, 'city': 0, 'age': 3, 'income': 1, 'bought': 0}
d["income"] = d["income"].fillna(d["income"].median())      # 整列填中位数 80.0
d["age"] = d["age"].fillna(d.groupby("city")["age"].transform("mean"))   # 组内填充：所在城市的平均年龄
d["age"] = d["age"].fillna(df["age"].median())              # LA 整组缺失，再用全局中位数兜底
print(d["age"].tolist())           # [25.0, 32.0, 35.0, 32.0, 28.0, 35.0, 45.0, 30.0]
print(df.dropna().shape, df.dropna(subset=["income"]).shape)   # (5, 5) (7, 5)
```

### groupby、排序与每组 Top-N

`agg` 每组压成一行；`transform` 把组统计量广播回每一行，长度和原表相同。分组键会进 index，要变回普通列就接 `.reset_index()`。

```python
stats = df.groupby("city").agg(n_users=("user_id", "count"), avg_income=("income", "mean"),
                               buy_rate=("bought", "mean"))      # 新列名=(原列名, 聚合函数字符串)
print(stats.round(2))
#       n_users  avg_income  buy_rate
# city
# LA          1         NaN      1.00
# NY          4       66.25      0.25
# SF          3      108.33      0.67
diff = df["income"] - df.groupby("city")["income"].transform("mean")   # (8,)，减所在城市的均值
print(diff.round(2).tolist())      # [-16.25, 11.67, -1.25, nan, -13.33, 3.75, 13.75, 1.67]
by_income = df.sort_values("income", ascending=False)                  # NaN 默认排最后
print(by_income["user_id"].tolist(), df.nlargest(3, "income")["user_id"].tolist())   # [2, 8, 5, 7, 6, 3, 1, 4] [2, 8, 5]
print(by_income.groupby("city").head(2)["user_id"].tolist())           # [2, 8, 7, 6, 4]，每个城市收入前 2
```

### merge、concat、pivot_table

「先 groupby 聚合，再 left merge 回主表，没记录的填 0」是最常见的特征套路；`validate` 在键意外重复时直接报错。

```python
users = df[["user_id", "city"]]
orders = pd.DataFrame({"user_id": [1, 1, 2, 5, 9], "amount": [30.0, 20.0, 50.0, 15.0, 99.0]})
print([users.merge(orders, on="user_id", how=h).shape for h in ["inner", "left", "outer"]])   # [(4, 3), (9, 3), (10, 3)]
spend = orders.groupby("user_id", as_index=False).agg(total=("amount", "sum"))
feat = users.merge(spend, on="user_id", how="left", validate="one_to_one")
print(feat["total"].fillna(0.0).tolist())      # [50.0, 50.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0]
print(pd.concat([users, users], ignore_index=True).shape, pd.concat([users, df[["age"]]], axis=1).shape)   # (16, 2) (8, 3)
table = df.pivot_table(index="city", columns="bought", values="user_id", aggfunc="count", fill_value=0)
print(table.values.tolist())                   # [[0, 1], [3, 1], [1, 2]]，行是 LA/NY/SF，列是 bought=0/1
X = df[["age", "income"]].fillna(0.0).to_numpy(dtype=np.float32)   # (8, 2)，接着 torch.from_numpy(X)
```

### Industry ML 小题：train 和 test 一起预处理

两个高频 bug：用 test 自己的统计量做填充和标准化（数据泄漏，同一个值在两边被映射成不同的数）；对 train、test 分别 `get_dummies`（列数和列顺序对不上）。题面给了函数签名（比如拆成 `fill_missing_values`、`one_hot_encode`、`standardize`）就照签名拆开写，不要改签名。

```python
from typing import Tuple

NUM, CAT = ["age", "income"], ["city"]
TRAIN_CSV = "age,income,city,bought\n25,50,NY,0\n32,120,SF,1\n,65,NY,0\n41,,LA,1\n28,95,SF,1\n35,70,NY,0\n"
TEST_CSV = "age,income,city,bought\n30,,NY,0\n,100,SF,1\n52,85,Austin,1\n"


def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """所有统计量只用 train。(n_train, d)、(n_test, d) -> 列名和顺序相同的两张表"""
    fill = train[NUM].median().to_dict()                     # 1. 数值列填 train 的中位数
    fill.update({c: "unknown" for c in CAT})                 #    类别列填 "unknown"
    train, test = train.fillna(fill), test.fillna(fill)
    train = pd.get_dummies(train, columns=CAT, dtype=int)    # 2. one-hot，test 缺的列补 0、多的列丢掉
    test = pd.get_dummies(test, columns=CAT, dtype=int).reindex(columns=train.columns, fill_value=0)
    mean, std = train[NUM].mean(), train[NUM].std(ddof=0)    # 3. z-score，ddof=0 和 StandardScaler 一致
    std = std.where(std > 1e-8, 1.0)                         #    常数列除以 1
    train[NUM], test[NUM] = (train[NUM] - mean) / std, (test[NUM] - mean) / std
    return train, test


if __name__ == "__main__":
    train, test = pd.read_csv(io.StringIO(TRAIN_CSV)), pd.read_csv(io.StringIO(TEST_CSV))   # 考试里读 data/ 下的文件
    X_tr, X_te = preprocess(train.drop(columns=["bought"]), test.drop(columns=["bought"]))
    assert list(X_tr.columns) == list(X_te.columns) and X_te.notna().all().all()
    assert np.allclose(X_tr[NUM].mean(), 0.0) and np.allclose(X_tr[NUM].std(ddof=0), 1.0)
    print(X_te.round(2))
    print("all tests passed")
#     age  income  city_LA  city_NY  city_SF
# 0 -0.43   -0.36        0        1        0
# 1 -0.03    0.95        0        0        1
# 2  3.90    0.29        0        0        0      <- Austin 没见过，三个 city 列都是 0
# all tests passed
```

### 常见坑

- **链式赋值。** `d[mask]["income"] = 0` 改的是临时子表，原表不变；改原表写一次 `d.loc[mask, "income"] = 0`，要独立子表就 `.copy()`。
- **`and` / `or` 不能连接 Series 条件。** 会报 `The truth value of a Series is ambiguous`；用 `&`、`|`、`~`，每个条件加括号。
- **过滤或排序后 index 不连续。** 这时 `loc[0]` 和 `iloc[0]` 取到的行不同；按位置一律写 `iloc`，需要时 `reset_index(drop=True)`。
- **有 NaN 的列不能直接 `astype(int)`。** 先 `fillna` 再转，或者转成可空整数 `"Int64"`。
- **`df.append` 在 pandas 2.0 删除了，`groupby().mean()` 遇到字符串列会报错。** 循环里把行攒成 dict 的 list，最后 `pd.DataFrame(rows)`；聚合前先选数值列。

---

## 0.3 PyTorch 速查

张量操作和 NumPy 几乎一一对应（`axis` 换成 `dim`）。代码块按顺序在同一个会话里运行，后面的块直接用前面定义的类。

### 创建：dtype 与 device

```python
import numpy as np
import torch

a, b = torch.tensor([1, 2, 3]), torch.tensor([1.0, 2.0])    # 小写 tensor：从数据建，自动推断 dtype
c = torch.from_numpy(np.array([1.0, 2.0]))                  # 和 NumPy 共享内存，还是 float64
print(a.dtype, b.dtype, c.dtype, torch.zeros(2, 3).shape)   # torch.int64 torch.float32 torch.float64 torch.Size([2, 3])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x32 = c.float().to(device)                    # x.to() 返回新张量要接住；model.to(device) 原地移动
print(x32.dtype, x32.device, b.long().dtype)  # torch.float32 cpu torch.int64，标签和 token id 一律 long
```

### 形状操作

```python
x = torch.arange(6).view(2, 3)                # (2, 3)，view 要求内存连续
print(x.t().is_contiguous(), x.t().reshape(6))    # False tensor([0, 3, 1, 4, 2, 5])，转置后 view(6) 会报错，用 reshape
B, T, H, D_h = 2, 5, 4, 8
q = torch.randn(B, T, H * D_h).view(B, T, H, D_h).transpose(1, 2)   # (B, H, T, D_h)：拆多头
merged = q.transpose(1, 2).contiguous().view(B, T, H * D_h)        # (B, T, C)：拼回多头，先 contiguous 再 view
print(q.permute(0, 2, 1, 3).shape, merged.shape)   # torch.Size([2, 5, 4, 8]) torch.Size([2, 5, 32])
print(x[0].unsqueeze(0).shape, x[0].unsqueeze(1).shape)   # torch.Size([1, 3]) torch.Size([3, 1])
print(torch.zeros(1, 3, 1).squeeze(-1).shape) # torch.Size([1, 3])，squeeze 总带 dim，免得 batch=1 也被挤掉
print(torch.cat([x, x], dim=0).shape, torch.stack([x, x], dim=0).shape)   # torch.Size([4, 3]) torch.Size([2, 2, 3])
```

### 索引、矩阵乘与归约

```python
logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])   # (N, C)
y = torch.tensor([0, 1])                      # (N,)，long；gather 的 index 和输出同形状，先 unsqueeze 成 (N, 1)
print(logits[torch.arange(2), y], logits.gather(1, y.unsqueeze(1)).squeeze(1))   # 两个都是 tensor([2.0000, 2.5000])
print(logits.topk(2, dim=-1).indices.tolist(), logits.argmax(dim=-1))   # [[0, 1], [1, 0]] tensor([0, 1])
print(logits.sum(dim=1, keepdim=True).shape, logits.max(dim=1).values)  # torch.Size([2, 1]) tensor([2.0000, 2.5000])
causal = torch.tril(torch.ones(3, 3))         # (T, T) 下三角：1 = 可见，0 = 屏蔽；masked_fill 返回新张量
print(torch.zeros(3, 3).masked_fill(causal == 0, -1e9).softmax(dim=-1)[1])   # tensor([0.5000, 0.5000, 0.0000])
A = torch.randn(2, 5, 8)                      # (B, T, C)
print((A @ torch.randn(8, 16)).shape, torch.bmm(A, torch.randn(2, 8, 7)).shape)   # torch.Size([2, 5, 16]) torch.Size([2, 5, 7])，bmm 只收 3 维
```

### autograd

`.grad` 默认累加，每步都要先清零；手动更新参数要包在 `no_grad` 里。累计 loss 写 `total += loss.item()`，直接加张量会把每个 batch 的计算图留在内存里；转 NumPy 写 `.detach().cpu().numpy()`。

```python
w = torch.tensor([1.0, 2.0], requires_grad=True)   # 叶子张量
(w ** 2).sum().backward()                     # backward 只能对标量调用
(w ** 2).sum().backward()
print(w.grad)                                 # tensor([4., 8.])，两次累加：2w + 2w
with torch.no_grad():
    w -= 0.1 * w.grad                         # 手动 SGD 一步；不包 no_grad 会报错
w.grad.zero_()                                # optimizer.zero_grad() 做的就是这个
print(w, (w * 3).detach().requires_grad, round((w ** 2).sum().item(), 4))   # tensor([0.6000, 1.2000], requires_grad=True) False 1.8
```

### nn.Module

```python
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()                             # 1. 先调父类构造：漏写的话，下一行赋值子模块就报 AttributeError
        self.inp = nn.Linear(in_dim, hidden_dim)       # 2. 有参数的层写成属性才会注册
        layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        self.hidden = nn.ModuleList(layers)            # 3. 必须包 ModuleList：普通 list 不注册，优化器看不到，也不报错
        self.dropout = nn.Dropout(0.1)                 # 自己会看 train() / eval()
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> logits: (B, out_dim)，不过 softmax"""
        h = F.relu(self.inp(x))                        # (B, hidden_dim)
        for layer in self.hidden:
            h = self.dropout(F.relu(layer(h)))         # (B, hidden_dim)
        return self.out(h)                             # (B, out_dim)


model = MLP(4, 8, 3)
print(model(torch.randn(5, 4)).shape, sum(p.numel() for p in model.parameters()))   # torch.Size([5, 3]) 211
```

### Dataset、DataLoader、训练骨架与保存

```python
import io
from typing import Tuple

from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    """dataset[i] = (X[i], y[i])，和现成的 TensorDataset(X, y) 等价"""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X, self.y = X, y                         # (N, D)、(N,)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


torch.manual_seed(42)
X = torch.randn(64, 4)                                # (N, D)
y = (X[:, 0] + X[:, 1] > 0).long()                    # (N,)，long 标签
loader = DataLoader(ToyDataset(X, y), batch_size=16, shuffle=True)
model, criterion = MLP(4, 16, 2), nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(20):                               # 完整的训练、验证循环和早停见专题 10.1 节
    model.train()                                     # 1. Dropout 生效
    for xb, yb in loader:                             # xb: (16, 4)，yb: (16,)
        optimizer.zero_grad()                         # 2. 清梯度（2.x 默认把 grad 设成 None）
        loss = criterion(model(xb), yb)               # 3. 前向，logits (16, 2)
        loss.backward()                               # 4. 反向
        optimizer.step()                              # 5. 更新
model.eval()                                          # 6. 关掉 Dropout
with torch.no_grad():                                 # 7. 不建计算图
    print((model(X).argmax(dim=1) == y).float().mean().item())   # 1.0，train acc
buffer = io.BytesIO()                                 # 真实场景写 "model.pt"
torch.save(model.state_dict(), buffer)                # 只存「名字 -> 张量」字典
buffer.seek(0)
print(MLP(4, 16, 2).load_state_dict(torch.load(buffer, weights_only=True)))   # <All keys matched successfully>
```

### 常见坑

- **loss 的 target。** `CrossEntropyLoss` 要原始 logits 和 `(N,)` 的 long 标签；`BCEWithLogitsLoss` 要 float、形状和 logits 相同。细节见专题第 9 节。
- **`(N, 1)` 和 `(N,)` 悄悄广播。** `nn.Linear(D, 1)` 输出 `(N, 1)`，`pred - target` 变成 `(N, N)`，loss 照样算得出来；先 `pred.squeeze(1)` 或 `assert pred.shape == target.shape`。
- **评估忘了 `model.eval()`。** `no_grad` 只管不建图，不会关掉 Dropout 和 BatchNorm 的训练行为。
- **原地操作破坏 autograd。** `y = x.exp()` 后再 `y.add_(1)`，backward 会报错，因为反向要用原来的 `y`；带下划线的方法都是原地操作。
- **float64 的 NumPy 数据直接喂 `nn.Linear`。** 报 `mat1 and mat2 must have the same dtype`，先 `.float()`；模型和数据也要在同一个 device。

---

## 0.4 手撸矩阵计算（纯 Python）

ML Core 的算法题不许 import 任何库，矩阵只能用 list of lists 表示：`A[i][j]` 是第 i 行第 j 列，形状 $(n, m)$ 就是 `(len(A), len(A[0]))`。本节约定：**题解代码一个 import 都没有**，`math`、`typing` 也不用，类型标注只写 `list`、`dict` 这些内置类型；空矩阵返回 `[]`，形状对不上就 `raise ValueError`。每道题的入口叫 `solution`，只转发给干活的函数（写法见矩阵乘法一小节，其余小节省略），节末的自测直接测这些函数；**NumPy 只出现在自测里**，提交时只交纯 Python 的题解。

### 形状检查与转置

list of lists 不会检查「每行一样长」，后面的函数都先过一遍 `get_shape`。按列算统计量时，先转置再按行处理最顺手。

```python
def get_shape(A: list) -> tuple:
    """返回 (行数, 列数)；空矩阵记为 (0, 0)；每一行必须一样长"""
    n_cols = len(A[0]) if A else 0
    if any(len(row) != n_cols for row in A):
        raise ValueError(f"ragged matrix: every row must have {n_cols} columns, like row 0")
    return len(A), n_cols

def transpose(A: list) -> list:
    """A: (n, m) -> (m, n)。一行写法 [list(col) for col in zip(*A)]：zip 产出 tuple，要转回 list"""
    n, m = get_shape(A)
    return [[A[i][j] for i in range(n)] for j in range(m)]   # 外层走列 j，内层走行 i；非方阵别都写成 range(n)
```

### 矩阵乘法

$C_{ij}=\sum_k A_{ik}B_{kj}$，形状 $(n,m)\times(m,p)\to(n,p)$。时间 $O(nmp)$，方阵就是 $O(n^3)$，额外空间是结果的 $O(np)$。循环用 i-k-j 顺序：最内层 j 沿着 B 和 C 的同一行走，`A[i][k]`、`B[k]` 提到内层循环外面，乘加次数和照公式写的 i-j-k 一样。

```python
def matmul(A: list, B: list) -> list:
    """A: (n, m), B: (m, p) -> A @ B: (n, p)"""
    (n, m), (m_b, p) = get_shape(A), get_shape(B)
    if n > 0 and m != m_b:
        raise ValueError(f"shape mismatch: A is ({n}, {m}), B is ({m_b}, {p}); need A cols == B rows")
    C = [[0.0] * p for _ in range(n)]       # (n, p)；不能写 [[0.0] * p] * n，那样 n 行是同一个 list
    for i in range(n):
        for k in range(m):
            a_ik, row_b = A[i][k], B[k]     # 整个内层循环都不变
            for j in range(p):
                C[i][j] += a_ik * row_b[j]
    return C

def solution(A: list, B: list) -> list:
    return matmul(A, B)
```

### 按列归一化：min-max 与 z-score

每一列是一个特征，各自缩放到可比的范围，统计量按第 j 列算。min-max 是 $x'_{ij}=(x_{ij}-\min_j)/(\max_j-\min_j)$，z-score 是 $x'_{ij}=(x_{ij}-\mu_j)/\sigma_j$。

σ 用总体标准差（除以 N），和 `np.std` 的默认值、sklearn 的 `StandardScaler` 一致；pandas 的 `.std()` 默认除以 N-1，题目写明用哪种就按题目来。常数列的分母是 0，约定输出全 0，并且要用原始数据的 `max == min` 判断：`[0.1, 0.1, 0.1]` 算出的 σ 是 1.3877787807814457e-17，用 `std == 0` 判断会一除全变成 -1.0。

```python
def normalize(X: list, method: str = "z-score") -> list:
    """X: (N, D) -> (N, D)，每列做 min-max 或 z-score（总体标准差）；常数列输出 0"""
    if method not in ("z-score", "min-max"):
        raise ValueError(f"unknown method: {method!r}")
    N, cols = len(X), transpose(X)                                  # cols: (D, N)；空矩阵时为 []
    const = [max(c) == min(c) for c in cols]                        # (D,)：用原始数据判断常数列
    if method == "min-max":
        shift = [min(c) for c in cols]                              # (D,)
        scale = [max(c) - lo for c, lo in zip(cols, shift)]         # (D,)
    else:
        shift = [sum(c) / N for c in cols]                          # (D,)：均值
        scale = [(sum((v - mu) ** 2 for v in c) / N) ** 0.5 for c, mu in zip(cols, shift)]   # 两遍法
    return [[0.0 if is_c else (v - s) / d for v, s, d, is_c in zip(row, shift, scale, const)]
            for row in X]                                           # (N, D)
```

### 前向传播：一个隐藏层的全连接网络

给定权重算输出：$H=\phi(XW_1+b_1)$，$P=\operatorname{softmax}(HW_2+b_2)$，P 是 $(N,C)$，每行是一个样本在 C 个类别上的概率。W 存成 (in, out)，和专题第 11 节的 NumPy 层同一个约定；PyTorch 的 `nn.Linear` 存成 (out, in)，题目给的权重要先看清。纯 Python 没有广播，偏置要自己加到每一行；不能 `import math`，就用 `E ** z` 代替 `math.exp(z)`。纯 Python 的浮点溢出会直接抛 `OverflowError`（NumPy 是返回 `inf` 加警告），所以 softmax 先减最大值，sigmoid 按正负分两支。

```python
E = 2.718281828459045                              # 自然常数 e，等于 math.e

def add_bias(Y: list, b: list) -> list:
    """Y: (N, D), b: (D,) -> (N, D)。手写广播：同一个 b 加到每一行"""
    N, D = get_shape(Y)
    if N > 0 and D != len(b):
        raise ValueError(f"shape mismatch: Y is ({N}, {D}), b has length {len(b)}")
    return [[y + bj for y, bj in zip(row, b)] for row in Y]

def sigmoid(z: float) -> float:
    # 只对非正数取指数，E ** 大正数会抛 OverflowError
    return 1.0 / (1.0 + E ** (-z)) if z >= 0 else E ** z / (1.0 + E ** z)

def softmax_row(z: list) -> list:
    """z: (C,) -> (C,)，和为 1；先减最大值，最大的指数项是 E ** 0 = 1"""
    z_max = max(z)
    exps = [E ** (v - z_max) for v in z]
    total = sum(exps)
    return [e / total for e in exps]

ACTIVATIONS = {"relu": lambda z: max(z, 0.0), "sigmoid": sigmoid}

def forward(X: list, W1: list, b1: list, W2: list, b2: list, activation: str = "relu") -> list:
    """X: (N, D_in), W1: (D_in, D_h), b1: (D_h,), W2: (D_h, C), b2: (C,) -> (N, C)，每行和为 1"""
    act = ACTIVATIONS[activation]                                        # 未知的激活函数名会抛 KeyError
    H = [[act(v) for v in row] for row in add_bias(matmul(X, W1), b1)]   # (N, D_h)
    logits = add_bias(matmul(H, W2), b2)                                 # (N, C)
    return [softmax_row(row) for row in logits]                          # (N, C)
```

### 稀疏矩阵乘法

LeetCode 311。矩阵大部分是 0 时（one-hot 特征、词袋向量、邻接矩阵），只存非零元素 `{行号: {列号: 值}}`。沿用 i-k-j 的思路，只走 A 第 i 行的非零元素 k，再只走 B 第 k 行的非零元素。乘法次数是 $\sum_{A_{ik}\ne0}\text{nnz}(B_{k,:})$，稠密输入转字典另要 $O(nm+mp)$；矩阵不稀疏时，字典的开销会让它比三重循环更慢。

```python
def to_sparse(A: list) -> dict:
    """稠密 (n, m) -> {i: {j: A[i][j]}}，只存非零元素，全零的行整行不存"""
    return {i: {j: v for j, v in enumerate(row) if v != 0} for i, row in enumerate(A) if any(row)}

def sparse_matmul(A: list, B: list) -> list:
    """A: (n, m), B: (m, p)，大部分元素是 0 -> A @ B: (n, p)"""
    (n, m), (m_b, p) = get_shape(A), get_shape(B)
    if n > 0 and m != m_b:
        raise ValueError(f"shape mismatch: A is ({n}, {m}), B is ({m_b}, {p}); need A cols == B rows")
    A_sp, B_sp = to_sparse(A), to_sparse(B)
    C = [[0.0] * p for _ in range(n)]                  # (n, p)
    for i, row_a in A_sp.items():
        for k, a in row_a.items():                     # A 第 i 行的非零元素
            for j, b in B_sp.get(k, {}).items():       # B 第 k 行的非零元素
                C[i][j] += a * b
    return C
```

### 自测

```python
import numpy as np

def close(mine: list, ref: np.ndarray) -> bool:
    return np.shape(mine) == ref.shape and np.allclose(mine, ref)        # 形状和数值都要对

def test_matrix() -> None:
    rng = np.random.default_rng(0)
    for n, m, p in [(1, 1, 1), (2, 3, 4), (5, 1, 3), (1, 4, 1), (4, 4, 4)]:
        A, B = rng.standard_normal((n, m)), rng.standard_normal((m, p))
        A[rng.random((n, m)) < 0.5] = 0.0                                   # 约一半置 0，给稀疏版用
        assert close(matmul(A.tolist(), B.tolist()), A @ B)
        assert close(sparse_matmul(A.tolist(), B.tolist()), A @ B)
        assert transpose(A.tolist()) == A.T.tolist()                        # 必须是 list，不能是 tuple
    X = rng.standard_normal((6, 4))
    assert close(normalize(X.tolist(), "min-max"), (X - X.min(0)) / (X.max(0) - X.min(0)))
    assert close(normalize(X.tolist()), (X - X.mean(0)) / X.std(0))        # np.std 默认除以 N
    assert normalize([[0.1]] * 3) == normalize([[0.1]] * 3, "min-max") == [[0.0], [0.0], [0.0]]
    X, W1, b1, W2, b2 = args = [rng.standard_normal(s) for s in [(5, 3), (3, 4), (4,), (4, 2), (2,)]]
    for name, act in [("relu", lambda z: np.maximum(z, 0.0)), ("sigmoid", lambda z: 1 / (1 + np.exp(-z)))]:
        logits = act(X @ W1 + b1) @ W2 + b2                                  # (5, 2)
        P = np.exp(logits - logits.max(1, keepdims=True))
        assert close(forward(*[a.tolist() for a in args], activation=name), P / P.sum(1, keepdims=True))
    assert softmax_row([1000.0, 0.0]) == [1.0, 0.0] and sigmoid(-1000.0) == 0.0 and sigmoid(1000.0) == 1.0
    assert matmul([], [[1.0]]) == [] and sparse_matmul([], [[1.0]]) == [] and normalize([]) == []
    try:
        matmul([[1.0, 2.0]], [[1.0, 2.0], [3.0]])                           # B 行长不一致
        raise AssertionError("matmul should raise ValueError")
    except ValueError:
        pass
    print("all tests passed")

if __name__ == "__main__":
    test_matrix()
```

### 关键追问

- **纯 Python 和 NumPy 的矩阵乘法差在哪？** 复杂度都是 $O(n^3)$，差在常数。NumPy 调用 BLAS（C 实现、分块让数据留在缓存里、SIMD、多线程），纯 Python 每次乘加都要过解释器、新建 float 对象。
- **归一化的统计量用哪部分数据算？** 只用训练集算 min、max 或 μ、σ，再用同一组数变换验证集和测试集，否则就是数据泄漏。测试集变换后超出 $[0,1]$ 是正常的。
- **min-max 和 z-score 怎么选？** min-max 对异常值敏感，z-score 好一些，更稳的是中位数加四分位距（sklearn 的 `RobustScaler`）。KNN、K-Means、SVM 和用梯度下降训练的模型需要归一化，决策树按阈值切分，不需要。
- **工程里稀疏矩阵怎么存？** 常用 CSR：非零值数组、列号数组、每行起点数组，`scipy.sparse.csr_matrix` 和 `torch.sparse_csr_tensor` 都支持。它和本节的「行号到字典」一样按行组织，用连续数组代替字典，更省内存、遍历更快。

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
| [11. ML Coding 基础](11-ml-coding-basics.md)              |
| [12. ML Coding 专题](12-ml-coding.md)              |
| [参考资料](references.md)                     |