# 11. ML Coding 基础

[返回目录](README.md)

本页是 ML Coding 的基础篇：先讲 CodeSignal 的题型和代码规范，再过一遍 NumPy、Pandas、PyTorch 的基本语法，最后是不许 import 时的纯 Python 矩阵运算。手写模型和算法题在 [12. ML Coding 专题](12-ml-coding.md)，正文里写作「专题第 N 节」。本页的节号是 0 到 0.4，专题页从第 1 节开始，两页的节号不重复。

---

## 0. CodeSignal 环境与代码规范

CodeSignal 上的 ML 面试有好几种形态，代码的写法跟着题型走。拿到题先判断是哪一种，再动手。下面的信息来自 CodeSignal 官方的测评白皮书和帮助文档（链接在本节末尾）。

### 三种题型

| 题型 | 长什么样 | 代码怎么写 |
| --- | --- | --- |
| 单函数题（GCA、ML Core 测评） | 一个固定的入口函数，题面给样例 | 实现 `solution(...)`，用 `return` 返回答案 |
| 多文件工程题（Industry ML、ICA） | 有 `solution.py`、`data/`、`tests/` 等文件 | 按给定的函数和类签名填实现，跑 `unittest` |
| 现场面试（CodeSignal Interview） | 面试官在线看你写，可能是上面任意一种，也可能是共享的 Jupyter Notebook | 面试官定规则；能跑、讲清楚最重要 |

### 单函数题的规则

- 入口函数叫 `solution`，**不能改名**，改名会直接报语法错误。辅助函数写在模块顶层，和 `solution` 并列。
- 答案用 `return` 返回。`print` 只用来调试，输出会显示在每个测试用例下面，不算答案。
- 题面里的例子写成 `For x = ..., the output should be solution(x) = y` 的形式。
- 能看到的只有 sample tests。点 Submit 才会跑全部测试，其中包括 hidden tests，得分取决于通过的测试数。
- 每道题有执行时间限制，写在题面的 Input/Output 部分。超时也算错，所以复杂度要心里有数。
- 可以反复 Submit，每道题按得分最高的那次提交算。官方建议早交、常交。

### ML Core 测评：纯 Python、不许调库

一些公司的 MLE 在线测评（OA）用的是 CodeSignal 的 Machine Learning Engineering Core 测评。白皮书描述的结构是 70 分钟、三个模块：

1. 6 道场景题，考 ML 基础概念。
2. 1 道数据处理编程题，约 15~20 行。
3. 2 道 ML 算法实现题，每题约 20~25 分钟、15~25 行。

候选人反馈的实际题量会有出入，比如多一道需要算出数值的计算题，或者多一道普通算法题。

第 3 模块的范围包括 kNN、k-Means、决策树、GMM、矩阵归一化、Bagging、前向传播。白皮书明确写了**不考需要 sklearn、PyTorch、TensorFlow 的内容**，官方 kNN 样题更是要求「不 import 任何库」。输入是 Python 的 list，返回值也是 list。所以 0.4 节的纯 Python 矩阵运算是这类题的基本功。

官方样题给的是**骨架代码**：几个辅助函数只有 `# implement this` 和 `pass`，入口 `solution` 和其余胶水代码已经写好，**不要改动**，只填空。下面按官方 kNN 样题的结构写一个示意版（胶水部分是简化后的示意，以真实题面为准；样题里每一行数据的最后一个元素是标签）：

```python
# ======== 题目给的骨架（示意）：只填 # implement this 的部分 ========

def euc_dist(value1, value2):
    # implement this
    pass


def k_neighbors(train_data, test_case, k):
    # implement this
    pass


def solution(train_data, test_data, k):
    final_labels = list()
    for test_case in test_data:
        neighbors = k_neighbors(train_data, test_case, k)
        labels = [row[-1] for row in neighbors]               # 每行最后一个元素是标签
        final_labels.append(max(set(labels), key=labels.count))   # 多数投票
    return final_labels
```

填空后的样子：

```python
def euc_dist(value1, value2):
    # 两个特征向量的欧氏距离；只用到特征部分，所以调用方要先去掉标签
    total = 0.0
    for a, b in zip(value1, value2):
        total += (a - b) ** 2
    return total ** 0.5


def k_neighbors(train_data, test_case, k):
    # 1. 算 test_case 到每个训练样本的距离（row[:-1] 去掉标签）
    distances = []
    for row in train_data:
        distances.append((euc_dist(row[:-1], test_case[:-1]), row))
    # 2. 按距离排序，取前 k 个样本
    distances.sort(key=lambda pair: pair[0])
    return [row for _, row in distances[:k]]


if __name__ == "__main__":
    train = [[1.0, 1.0, 0.0], [1.2, 0.8, 0.0], [5.0, 5.0, 1.0], [5.2, 4.9, 1.0], [4.8, 5.1, 1.0]]
    test = [[1.1, 0.9, -1.0], [5.1, 5.0, -1.0]]                # 测试样本的标签位置先占位
    assert solution(train, test, 3) == [0.0, 1.0]
    print(solution(train, test, 3))                          # [0.0, 1.0]
```

注意两个细节：骨架里 `euc_dist` 接收的参数含不含标签，要看胶水代码怎么调用；`k_neighbors` 返回的是整行数据，因为 `solution` 要从 `row[-1]` 取标签。填空题最容易丢分的地方，就是没读懂给定代码的接口约定。

### 多文件工程题

Industry ML 测评的样题是一个小工程：

```
solution.py               # 你要改的文件：load_data、fill_missing_values、train_models 等函数
data/data.csv             # 输入数据
tests/tests_basic.py      # 可见测试（unittest），故意不完整
tests/tests_data/expected.csv
```

样题的参考答案用了类型标注（`def load_data(file_name: str) -> DataFrame`）、小而单一职责的函数，以及 `main()` 加 `if __name__ == "__main__": main()` 的入口。可见测试只是最低要求，比如模型题的可见测试准确率门槛设得很低，评分用的是更严格的 hidden tests，所以要按题面的 Acceptance Criteria 写，不能只满足可见测试。类和方法的签名不要改，ICA 的题面原话是 *do not change the existing method signatures*。0.2 节的 Pandas 小任务就是按这个格式写的。

### 现场面试

- 面试官可以在会话里打开共享的 Jupyter Notebook，双方实时编辑。官方说预装了「几百个包」，缺的可以在终端里 `pip3 install --user 包名`，装完重启 kernel。
- 有一个 Ubuntu 终端，可以跑代码、装软件。
- 官方文档没有提到 GPU。按 CPU、小数据写，示例都控制在几秒内跑完。
- 官方没有公布 Python 和各个库的版本。开场先跑一行确认：

```python
import sys
import numpy as np

print(sys.version.split()[0], np.__version__)
# 再试 import torch / import pandas；import 失败就说明没装，问面试官能否 pip 安装
```

### 新代码遵守的规范

本页 0.1 到 0.4 节，以及专题页的 1.1、2.5、2.6、3、6.1、8.3、10.1、11.2 到 11.4、14、15 节，代码统一按下面的规范写：

- **兼容 Python 3.8**：类型标注用 `typing` 里的 `List`、`Optional`、`Tuple`、`Dict`，不写 `list[int]`、`int | None`。
- **函数带类型标注和 docstring**，docstring 写清输入输出的形状：

  ```python
  from typing import Optional

  import torch


  def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
      """
      Args:
          x: (B, T, C)
          mask: (B, T)，1 表示有效位置，0 表示 padding
      Returns:
          (B, C)
      """
      if mask is None:
          return x.mean(dim=1)
      mask = mask.unsqueeze(-1).to(x.dtype)                        # (B, T, 1)
      return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, C)
  ```

- **单函数题风格**的内容用 `solution(...)` 做入口，纯 Python 题不 import 任何库。
- **PyTorch 模块**按 CodeSignal Learn 课程的写法：继承 `nn.Module`，写 `super(ClassName, self).__init__()`；注意力里的线性层叫 `w_q`、`w_k`、`w_v`、`w_o`；mask 用 `masked_fill(mask == 0, -1e9)`，1 表示可见。
- **训练循环**按 `model.train()`、`optimizer.zero_grad()`、前向、`loss.backward()`、`optimizer.step()` 的顺序写，验证时 `model.eval()` 加 `with torch.no_grad():`，日志用 f-string。
- **每节结尾有自测**：固定随机种子，用 `assert` 检查形状和数值，能和库函数对拍的都对拍，最后打印 `all tests passed`。这和 CodeSignal 用测试用例判分的方式一致：在本地先把自己当成 hidden tests 跑一遍。

### 考前清单

- 在 `app.codesignal.com/assessments/practice` 用真实 IDE 练各种题型。练习题有可续期的 1 小时计时，公司看不到练习记录。
- 问清楚 recruiter 是哪种题型：ML Core 测评、多文件工程题，还是现场面试。三者的代码写法差别很大。
- 认证测评默认不允许用 AI，除非公司开启了 CodeSignal 的 AI 助手 Cosmo。能不能搜语法，不同测评的说法不一致，以开考页面上的规则为准。
- 每道单函数题先看 README 或 Info 标签页里的 Language 一节，那里写着解释器版本和自动 import 的库。

### 关键追问

- **为什么 `solution` 不能改名？** 测试框架按名字调用它，而且会把你的函数包在一个闭包里执行，改名后框架找不到入口，直接报错。
- **sample tests 全过了，为什么分数还不满？** 得分看的是全部测试，hidden tests 只在 Submit 时运行，而且往往更严格（Industry ML 白皮书直接说可见测试是故意不完整的）。空输入、只有一个元素、全部相同的值、超时，都是常见的失分点。
- **ML Core 测评不能用 NumPy，矩阵运算怎么写？** 用 list of lists 加循环，见 0.4 节。复杂度和 NumPy 版本相同，只是常数大；15~25 行就能写完。
- **现场面试可以查 API 吗？** 先问面试官。有 ML coding 面试官公开说过，不要求记住 API，可以申请查 torch 或 numpy 的文档；看重的是代码能跑、形状讲得清楚。

### 参考

- [Machine Learning Engineering Core 测评白皮书](https://discover.codesignal.com/rs/659-AFH-023/images/Machine-Learning-Engineering-Core-Skills-Evaluation-Framework-CodeSignal-Skills-Evaluation-Lab.pdf)
- [Machine Learning Engineering Industry Framework 2023](https://discover.codesignal.com/rs/659-AFH-023/images/Machine-Learning-Engineering-Industry-Framework-2023-Technical-Brief.pdf)
- [Using the README during your CodeSignal assessment](https://support.codesignal.com/hc/en-us/articles/1500000922402-Using-the-README-during-your-CodeSignal-assessment)
- [What is the CodeSignal Cloud IDE coding environment](https://support.codesignal.com/hc/en-us/articles/360039872914-What-is-the-CodeSignal-Cloud-IDE-Coding-Environment)
- [Using Jupyter Notebook for live data science interviews](https://support.codesignal.com/hc/en-us/articles/360050875793-Using-Jupyter-Notebook-for-live-data-science-interviews)
- [How do I practice coding questions on CodeSignal](https://support.codesignal.com/hc/en-us/articles/21025134150423-How-do-I-practice-coding-questions-on-CodeSignal)

---

## 0.1 NumPy 基本语法

手写 KMeans、Attention 时，最容易卡住的往往是 shape、axis 和广播这些基本功。这一节按主题速查：每个小节用一两句话讲规则，配一段能直接运行的代码，注释里的输出都是真实运行结果，numpy 1.24 和 2.0 下完全相同。养成一个习惯：每一步在注释里标出 shape，面试时边写边念出来。

### 创建数组与 dtype

一个数组里所有元素是同一种类型（dtype）。全是整数就是 int64（Windows 上的 NumPy 1.x 是 int32），混进一个小数，整个数组就变成 float64；PyTorch 的浮点默认是 float32，数组交给模型前要转换，见 0.3 节。标签 `y` 要存成整数，因为它要当下标去取 `probs[i, y[i]]`，float 数组不能当下标。

```python
import numpy as np

a = np.array([1, 2, 3])                      # (3,)，全是整数
b = np.array([1, 2.7, 3])                    # (3,)，混入一个小数，整个数组升成 float
print(a.dtype, b.dtype)                      # int64 float64
print(b.astype(int))                         # [1 2 3]，向 0 截断：2.7 变成 2，-2.7 会变成 -2，不做四舍五入
print(np.zeros((2, 3)).shape)                # (2, 3)，同类还有 np.ones、np.full((2, 3), 7.0)、np.eye(3)
print(np.arange(0, 1, 0.25))                 # [0.   0.25 0.5  0.75]，左闭右开，不含 1
print(np.linspace(0, 1, 5))                  # [0.   0.25 0.5  0.75 1.  ]，含终点，一共 5 个

probs = np.array([[0.1, 0.9], [0.8, 0.2]])   # (N, C) = (2, 2)
y = np.array([1.0, 0.0])                     # (N,)，标签被存成了 float
try:
    probs[np.arange(2), y]
except IndexError as e:
    print(e)                                 # arrays used as indices must be of integer (or boolean) type
print(probs[np.arange(2), y.astype(int)])    # [0.9 0.8]，转成 int 后取到每个样本真实类别的概率
```

### shape、reshape 与拼接

`reshape` 只改「怎么切行」，元素的先后顺序不变；`.T` 是行列互换，两者结果不同。`-1` 表示这一维让 NumPy 自己算，`None` 在指定位置插入一个长度为 1 的新轴。拼接时，`concatenate` 沿已有的轴拼，那一维的长度相加；`stack` 先新建一个轴再拼，要求所有输入形状完全相同。

```python
import numpy as np

x = np.arange(6)                             # (6,)：[0 1 2 3 4 5]
X = x.reshape(2, 3)                          # (2, 3)，按行依次填入
print(X.reshape(-1).shape, X.reshape(3, -1).shape)   # (6,) (3, 2)
print(X.reshape(3, 2).tolist())              # [[0, 1], [2, 3], [4, 5]]，顺序不变，重新切成 3 行
print(X.T.tolist())                          # [[0, 3], [1, 4], [2, 5]]，转置：第 i 列变成第 i 行
print(x[:, None].shape, x[None, :].shape)    # (6, 1) (1, 6)
print(x[:, None].squeeze().shape)            # (6,)，squeeze 去掉长度为 1 的轴
A = np.zeros((4, 2, 3))                      # (B, T, C)
print(A.transpose(0, 2, 1).shape)            # (4, 3, 2)，交换最后两维，相当于 torch 的 transpose(-2, -1)

Y = np.ones((2, 3))                          # (2, 3)
print(np.concatenate([X, Y], axis=0).shape)  # (4, 3)，行数相加；二维时等于 np.vstack
print(np.concatenate([X, Y], axis=1).shape)  # (2, 6)，列数相加；二维时等于 np.hstack
print(np.stack([X, Y]).shape)                # (2, 2, 3)，最前面多出一个长度 2 的轴
rows = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]   # 3 个 (2,)
print(np.stack(rows).shape)                  # (3, 2)，一组向量拼成矩阵
print(np.hstack([np.ones((2, 1)), X]).shape) # (2, 4)，最左边加一列 1 当截距，专题第 4 节用过
```

### 索引：切片、布尔掩码、花式索引

切片 `start:stop:step` 不含 stop，返回的是 view（和原数组共用内存）。布尔掩码按条件筛选；多个条件用 `&`、`|`、`~` 组合，每个条件都要加括号，因为 `&` 的优先级比 `>` 高，Python 的 `and` 也不能用在数组上。花式索引用整数数组当下标，`X[np.arange(n), y]` 表示第 i 行取第 `y[i]` 列，专题第 6 节交叉熵取真实类别那一项就是这么写的。

```python
import numpy as np

X = np.arange(12).reshape(3, 4)              # (3, 4)：[[0 1 2 3] [4 5 6 7] [8 9 10 11]]
print(X[1:, ::2].tolist())                   # [[4, 6], [8, 10]]，从第 1 行起，每隔一列取一个
print(X[:, 0].shape, X[:, [0]].shape)        # (3,) (3, 1)，单个整数会去掉这一维，列表或切片会保留
print(X[[0, 2]].shape)                       # (2, 4)，取第 0 行和第 2 行

mask = X[:, 0] > 3                           # (3,)：[False  True  True]
print(X[mask].shape)                         # (2, 4)，只留第 0 列大于 3 的行
both = (X[:, 0] > 3) & (X[:, 1] < 9)         # (3,)，两个条件都满足；去掉括号会报 ValueError
print(both)                                  # [False  True False]
print(X[X % 5 == 0])                         # [ 0  5 10]，二维掩码取出来的结果是一维

y = np.array([2, 0, 3])                      # (3,)，每行要取的列号
print(X[np.arange(3), y])                    # [ 2  4 11]，(3,)：行号和列号一一配对
print(X[:, y].shape)                         # (3, 3)，常见错写法：每一行都取了这 3 列，不报错

row = X[0, :2]                               # 切片是 view
row[0] = -1
print(X[0])                                  # [-1  1  2  3]，改 row 把 X 也改了
```

### 广播

两个形状不同的数组做运算时，NumPy 按三条规则自动对齐：

1. 把两个 shape 右对齐，左边缺的维当成 1。
2. 每一维要么相等，要么其中一个是 1，是 1 的那个被拉伸成另一个的长度。
3. 有一维两边都不是 1 又不相等，就报错。

```python
import numpy as np

X = np.ones((2, 3))                          # (2, 3)
print((X + np.array([10, 20, 30])).shape)    # (2, 3)，(2, 3) + (3,)：每行加同一个向量
print((X + np.array([[10], [20]])).shape)    # (2, 3)，(2, 3) + (2, 1)：每行加不同的数
try:
    X + np.array([10, 20])                   # (2, 3) + (2,)：右对齐后 3 对 2
except ValueError as e:
    print(e)                                 # operands could not be broadcast together with shapes (2,3) (2,)

y = np.array([1.0, 2.0, 3.0])                # (3,)
y_pred = np.array([[1.1], [1.9], [3.2]])     # (3, 1)，模型输出常见的形状
print((y_pred - y).shape)                    # (3, 3)，不报错，悄悄变成了两两相减
wrong = np.mean((y_pred - y) ** 2)
right = np.mean((y_pred.reshape(-1) - y) ** 2)   # 先拉平成 (3,) 再相减
print(f"{wrong:.4f} {right:.4f}")            # 1.4200 0.0200
```

`(n,)` 和 `(n, 1)` 混用是最常见的静默 bug：MSE 从 0.02 变成了 1.42，程序却照常运行。算 loss 之前先把两边都 `reshape(-1)`。专题第 3 节算距离矩阵的 `X[:, None, :] - centroids[None, :, :]` 则是故意利用广播。

### axis 与 keepdims

口诀：`axis=k` 就是沿第 k 维压扁，这一维从 shape 里消失。`keepdims=True` 把被压扁的那一维留成长度 1，结果还能按上面的规则和原数组广播。`axis=-1` 指最后一维。

```python
import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6]])         # (2, 3)
print(X.sum(axis=0))                         # [5 7 9]，压掉 axis 0（2 行）→ (3,)，每列一个和
print(X.sum(axis=1))                         # [ 6 15]，压掉 axis 1（3 列）→ (2,)，每行一个和
print(X.sum(axis=1, keepdims=True).shape)    # (2, 1)
P = X / X.sum(axis=1, keepdims=True)         # (2, 3) / (2, 1)：每行除以自己的和
print(P.sum(axis=1))                         # [1. 1.]
print(X.sum())                               # 21，不写 axis 就对全部元素求和

S = np.arange(1.0, 10.0).reshape(3, 3)       # (3, 3)，方阵，比如 self-attention 的 (T, T) 分数
print((S / S.sum(axis=1)).sum(axis=1))       # [0.425 1.25  2.075]，漏了 keepdims：不报错，每行和却不是 1
```

漏写 `keepdims=True` 时，`(2, 3) / (2,)` 右对齐后是 3 对 2，会报错，反而容易发现。方阵就危险了：`(3, 3) / (3,)` 能广播，第 j 列被除以第 j 行的和，静默算错。self-attention 的分数正好是 `(T, T)`，所以按行归一化一律写 `keepdims=True`，专题第 6 节的 softmax 就是这么写的。

### 归约：sum、mean、std、argmax、cumsum

归约（reduction）把一组数压成一个数。下面这些函数都接受 `axis` 和 `keepdims`，规则同上；不写 `axis` 就是对全部元素。

```python
import numpy as np

x = np.array([3, 1, 4, 1, 5])                # (5,)
print(x.sum(), x.mean(), x.max())            # 14 2.8 5
print(x.std(), x.std(ddof=1))                # 1.6 1.7888543819998317，默认除以 n，ddof=1 除以 n-1
print(x.argmax(), x.argmin())                # 4 1，有并列时返回第一次出现的位置
print(np.cumsum(x))                          # [ 3  4  8  9 14]，前缀和
logits = np.array([[0.1, 2.0, -1.0], [1.5, 0.3, 0.2]])   # (N, C) = (2, 3)
pred = logits.argmax(axis=1)                 # (N,)，每行最大值的下标就是预测类别
print(pred)                                  # [1 0]
print((pred == np.array([1, 2])).mean())     # 0.5，准确率：布尔数组的均值就是 True 的比例
```

### 排序与计数：sort、argsort、argpartition、unique

`argsort` 只有升序。要降序就对 `-scores` 排，或者把升序结果倒过来写 `np.argsort(scores)[::-1]`（专题 2.4 节的写法）。只要前 k 大时，`argpartition` 用 O(n) 把最大的 k 个挪到末尾，但它们之间没有排序，需要的话再对这 k 个排一次，合计 O(n + k log k)；数据一条条到来时改用专题第 12 节的堆。每一行各取 top-k 时加 `axis=1`，再用 `np.take_along_axis` 按下标逐行取值。

```python
import numpy as np

scores = np.array([0.2, 0.9, 0.1, 0.7, 0.5])   # (5,)
print(np.sort(scores))                       # [0.1 0.2 0.5 0.7 0.9]，返回新数组，升序
print(np.argsort(-scores))                   # [1 3 4 0 2]，从大到小排列的下标
k = 2
top = np.argpartition(scores, -k)[-k:]       # (k,)，最大的 k 个的下标，顺序不保证
top = top[np.argsort(-scores[top])]          # 只对这 k 个从大到小排
print(top)                                   # [1 3]

D = np.array([[3, 1, 2], [0, 5, 4]])         # (2, 3)，2 个查询点到 3 个样本的距离
idx = np.argsort(D, axis=1)[:, :2]           # (2, 2)，每行最近的 2 个样本的列号
print(np.take_along_axis(D, idx, axis=1).tolist())   # [[1, 2], [0, 4]]，逐行按列号取出距离，专题第 9 节 KNN 用过

labels = np.array([2, 0, 2, 1, 2])           # (5,)
values, counts = np.unique(labels, return_counts=True)   # values 已经升序
print(values, counts)                        # [0 1 2] [1 1 3]
print(values[counts.argmax()])               # 2，出现最多的类别，KNN 投票就这么写
```

### 条件与截断：where、clip、maximum

`np.where(cond, a, b)` 逐元素二选一。`np.clip` 把数截到区间里，对概率取 log 前写 `np.clip(p, 1e-15, 1 - 1e-15)` 防 `log(0)`。`np.maximum` 是两个数组（或数组和标量）逐元素比大小，`np.max` 是一个数组内部求最大值的归约。别把 ReLU 写成 `np.max(x, 0)`：它的第二个参数是 axis。

```python
import numpy as np

x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])   # (5,)
print(np.where(x > 0, x, 0.0))               # [0.  0.  0.  1.5 3. ]，条件为真取 x，否则取 0
print(np.where(x > 0))                       # (array([3, 4]),)，只传条件时返回下标，结果是 tuple，一维时取 [0]
print(np.clip(x, -1.0, 2.0))                 # [-1.  -0.5  0.   1.5  2. ]，截到 [-1, 2] 之内
print(np.maximum(x, 0))                      # [0.  0.  0.  1.5 3. ]，逐元素取大，就是 ReLU
print(np.max(x))                             # 3.0，整个数组的最大值，是归约
```

### 矩阵运算：乘法、范数、einsum

`*` 是逐元素乘，`@` 是矩阵乘。`einsum` 用下标字符串描述运算，输出里没出现的下标（下面的 `j` 和 `d`）会被求和，写 batch 矩阵乘很直观。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])               # (2, 2)
B = np.array([[5, 6], [7, 8]])               # (2, 2)
print((A * B).tolist())                      # [[5, 12], [21, 32]]，逐元素乘
print((A @ B).tolist())                      # [[19, 22], [43, 50]]，矩阵乘
v = np.array([1, 2])                         # (2,)
print(A @ v, v @ v, np.dot(v, v))            # [ 5 11] 5 5，(2, 2) @ (2,) → (2,)；两个向量做 @ 是内积
X = np.array([[3.0, 4.0], [6.0, 8.0]])       # (2, 2)
print(np.linalg.norm(X, axis=1))             # [ 5. 10.]，每行一个 L2 范数；不写 axis 就把所有元素当成一个向量算
print(np.einsum("ij,jk->ik", A, B).tolist()) # [[19, 22], [43, 50]]，和 A @ B 相同
Q = np.ones((2, 4, 8))                       # (B, T_q, d)
K = np.ones((2, 5, 8))                       # (B, T_k, d)
print(np.einsum("bqd,bkd->bqk", Q, K).shape) # (2, 4, 5)，等价于 Q @ K.transpose(0, 2, 1)
```

### 随机数：default_rng

新代码推荐 `rng = np.random.default_rng(seed)`，再用 `rng.xxx` 生成随机数。`rng` 是一个独立的生成器对象，可以当参数传给函数，不同模块互不干扰。旧写法 `np.random.seed(42)` 加 `np.random.rand(...)`（专题第 3 节 Example 用的）设的是全局状态，任何地方调用一次 `np.random.xxx` 都会推进它；两种写法的随机序列也不同。

```python
import numpy as np

rng = np.random.default_rng(0)               # 固定种子，结果可复现
print(rng.integers(0, 10, size=5))           # [8 6 5 2 3]，[0, 10) 里的整数，不含 10
print(rng.normal(0.0, 1.0, size=(2, 3)).shape)   # (2, 3)，均值 0、标准差 1 的正态分布
print(rng.choice(5, size=3, replace=False))  # [4 2 0]，从 0..4 里不放回抽 3 个
X, y = np.arange(10).reshape(5, 2), np.arange(5)   # (5, 2)、(5,)：第 i 行是 [2i, 2i+1]，标签是 i
perm = rng.permutation(5)                    # (5,)，0..4 的一个随机排列
print(perm)                                  # [2 1 3 4 0]
X_shuf, y_shuf = X[perm], y[perm]            # 同一个 perm 打乱 X 和 y，样本和标签不会错位
print((X_shuf[:, 0] // 2 == y_shuf).all())   # True
print(rng.choice(3, size=5, p=[0.1, 0.2, 0.7]))   # [0 2 1 2 2]，按概率 p 抽，p 之和必须是 1；K-Means++ 和专题 2.4 节采样都用它
```

### 常见坑

- **切片是 view，花式索引和布尔掩码取出来的是副本。** `X[0]`、`X[:, 1:3]` 和 `X` 共用内存，改它就改了原数组；`X[[0, 2]]`、`X[mask]` 是一份新数据。要独立的一份就 `.copy()`。
- **赋值会写回原数组。** `X[mask] = 0`、`X[np.arange(N), y] -= 1`（专题第 11 节算 softmax 梯度就这么写）直接改 `X`。链式的 `X[idx][:, 0] = 99` 改的是中间副本，`X` 不变。
- **重复下标只生效一次。** `a[[0, 0, 1]] += 1` 里下标 0 只加了 1。要按出现次数累加，用 `np.add.at(a, idx, 1)` 或 `np.bincount(idx, minlength=len(a))`。
- **浮点数不要用 `==` 比较。** `0.1 + 0.2 == 0.3` 是 False。用 `np.isclose`（逐元素）或 `np.allclose`（整个数组都接近才是 True），和参考答案对拍也用 `np.allclose`。
- **整型数组装不下小数。** 往 int 数组里赋值 2.7 会被静默截成 2；`c -= 0.5` 这种原地运算直接报 `UFuncTypeError`。输入先 `np.asarray(x, dtype=float)`。
- **除零、溢出、`log(0)` 不抛异常。** NumPy 只给一个 RuntimeWarning，结果里的 inf、nan 会一路传下去。可疑时用 `np.isfinite(x).all()` 检查，数值稳定的写法见专题第 4 节和第 6 节。
- **`if a == b:` 会报 ValueError。** 元素多于一个时，数组的真假有歧义。要写 `(a == b).all()`，或者直接用 `np.array_equal(a, b)`、`np.allclose(a, b)`。两边形状对不上时，numpy 1.24 的 `==` 只给一个 DeprecationWarning 并返回 False，numpy 2.0 直接报错。
- **`solution()` 返回前转回 Python 类型。** 数组用 `.tolist()`，标量用 `float(x)`、`int(i)`。返回 ndarray 的话，`assert solution(x) == expected` 会因为上一条的歧义报错；`np.int64` 也不能 `json.dumps`；numpy 2 下标量还会显示成 `np.float64(2.8)`。

```python
import numpy as np

X = np.arange(6).reshape(2, 3)               # (2, 3)：[[0 1 2] [3 4 5]]
X[X > 3] = 0                                 # 掩码赋值：直接写回 X
X[[0, 1]][:, 0] = 99                         # 链式：X[[0, 1]] 先取出副本，改的是副本
print(X.tolist())                            # [[0, 1, 2], [3, 0, 0]]
idx = [0, 0, 1]                              # 下标 0 出现两次
a, b = np.zeros(3), np.zeros(3)              # (3,)、(3,)
a[idx] += 1                                  # 每个位置只加一次
np.add.at(b, idx, 1)                         # 按出现次数累加
print(a, b)                                  # [1. 1. 0.] [2. 1. 0.]

c = np.array([1, 2, 3])                      # (3,)，int64
c[0] = 2.7                                   # 小数被直接截掉，不报错
print(c)                                     # [2 2 3]
try:
    c -= 0.5                                 # 结果是 float，放不回 int 数组
except TypeError as e:                       # UFuncTypeError 是 TypeError 的子类
    print(type(e).__name__)                  # UFuncTypeError
```

### 练习：按列 z-score 标准化

CodeSignal ML Core 考纲里有「矩阵归一化」，题面大致是：

- 输入 `matrix`（n 行 d 列的 list of lists），返回同形状的 list of lists。
- 每一列减去该列均值，再除以该列的总体标准差（`ddof=0`）；常数列输出全 0。
- For `matrix = [[1, 10, 5], [2, 20, 5], [3, 30, 5]]`, the output should be `solution(matrix) = [[-1.2247, -1.2247, 0.0], [0.0, 0.0, 0.0], [1.2247, 1.2247, 0.0]]`（保留 4 位小数展示）。

NumPy 里 z-score 本身就是一行 `(X - X.mean(axis=0)) / X.std(axis=0)`，min-max 是 `(X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))`，`(n, d)` 和 `(d,)` 靠广播对齐。麻烦在常数列：分母是 0，结果是 nan 加一个 RuntimeWarning。所以先用 `max == min` 找出常数列，分母换成 1，再把这些列置 0。为什么不能用 `std == 0` 判断常数列，以及不许 import 时的纯 Python 写法，见 0.4 节。

```python
import statistics
from typing import List

import numpy as np


def solution(matrix: List[List[float]]) -> List[List[float]]:
    """按列做 z-score 标准化，常数列输出 0。

    Args:
        matrix: (n, d) 的二维列表
    Returns:
        (n, d) 的二维列表，元素是 Python float
    """
    if len(matrix) == 0:
        return []
    X = np.asarray(matrix, dtype=float)                       # 1. (n, d)，转 float，避免整型截断
    const = X.max(axis=0) == X.min(axis=0)                    # 2. (d,)，常数列
    std = np.where(const, 1.0, X.std(axis=0))                 # 3. (d,)，常数列的分母换成 1，免得 0/0
    Z = np.where(const, 0.0, (X - X.mean(axis=0)) / std)      # 4. (n, d)，(d,) 广播到每一行，常数列置 0
    return Z.tolist()                                         # 5. 转回 Python 的 list 和 float


def test_solution() -> None:
    expected = [[-1.2247, -1.2247, 0.0], [0.0, 0.0, 0.0], [1.2247, 1.2247, 0.0]]
    assert np.allclose(solution([[1, 10, 5], [2, 20, 5], [3, 30, 5]]), expected, atol=1e-4)   # 题面样例
    assert solution([[0.1], [0.1], [0.1]]) == [[0.0], [0.0], [0.0]]    # 浮点常数列，见 0.4 节
    assert solution([[4.0, 7.0]]) == [[0.0, 0.0]] and solution([]) == []   # 只有一行；空输入
    X = np.random.default_rng(42).normal(5.0, 3.0, size=(50, 4)).tolist()   # (50, 4)
    stats = [(statistics.mean(c), statistics.pstdev(c)) for c in zip(*X)]   # 每列 (均值, 总体标准差)
    ref = [[(v - m) / s for v, (m, s) in zip(row, stats)] for row in X]     # 用标准库逐列算参考答案
    assert np.allclose(solution(X), ref)
    print("all tests passed")


if __name__ == "__main__":
    test_solution()
```

### 关键追问

- **`np.std`、pandas、PyTorch 的标准差默认一样吗？** 不一样。`np.std` 默认 `ddof=0`，除以 n；pandas 的 `.std()` 和 `torch.std` 默认除以 n-1。对 `[3, 1, 4, 1, 5]`，NumPy 是 1.6，pandas 是 1.7888543819998317，`torch.tensor([3., 1, 4, 1, 5]).std()` 是 `tensor(1.7889)`（float32；整数张量会直接报错）。题目没写清时先问，或者用样例反推。
- **`@`、`np.dot`、`*` 有什么区别？** `*` 逐元素乘（会广播）。`@` 即 `np.matmul`，把最后两维当矩阵、前面的维当 batch。`np.dot` 在一维、二维时和 `@` 一样，三维以上规则不同：`(2, 4, 3)` 和 `(2, 3, 5)` 做 `np.dot` 得到 `(2, 4, 2, 5)`，做 `@` 得到 `(2, 4, 5)`。batch 矩阵乘用 `@` 或 `einsum`。
- **为什么要向量化？** NumPy 的运算在 C 里循环，Python 的 for 循环每个元素都要经过解释器。距离矩阵、softmax 这种能用广播一行写完的，面试里不要写双重循环（专题第 3 节、第 9 节）。
- **广播会不会把内存撑爆？** 会。`X[:, None, :] - C[None, :, :]` 先生成 `(n, k, d)` 的中间数组，n=1000、k=10、d=64 的 float64 就是 5120000 字节（约 5 MB）。KMeans 的 k 小，问题不大；KNN 要算 m 个测试点到 n 个训练点的距离，中间数组是 `(m, n, d)`，就要改用专题第 9 节的展开公式，内存只要 `(m, n)`。
- **怎么判断一个结果是 view 还是副本？** 用 `np.shares_memory(a, b)`。切片、`.T` 是 view；`reshape` 能不复制就不复制，所以 `X.reshape(...)` 通常是 view，`X.T.reshape(-1)` 却会复制；花式索引、布尔掩码、`.copy()` 一定是副本。
- **`np.isclose` 有什么坑？** 默认 `atol=1e-8`，比较很小的数时几乎总返回 True：`np.isclose(1e-10, 2e-10)` 是 True。比较这种量级时设 `atol=0`，只靠相对误差 `rtol`。

---

## 0.2 Pandas 基本语法

CodeSignal 的 Industry ML 工程题和现场面试里的数据题，第一步通常是用 Pandas 读表、清洗、做特征。这一节是速查表：每段给最常用的写法和容易踩的坑，最后用一个 `solution.py` 形状的小题把它们串起来。

先记两个概念：`DataFrame` 是带列名的二维表，它的每一列是一个 `Series`（带索引的一维数组）。写 Pandas 的心法和 NumPy 一样：**按整列向量化运算，少写 Python for 循环。** 下面的代码块共用同一个 `df`，按顺序运行；会改数据的地方都先 `.copy()`，所以 `df` 本身始终不变。所有代码在 pandas 1.5、2.0、2.3 上都跑过，注释里的输出三个版本相同，版本差异在关键追问里单独列出。

### 造数据、先看一眼

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "city": ["NY", "SF", "NY", "LA", "SF", "NY", "NY", "SF"],
    "age": [25, 32, np.nan, np.nan, 28, 35, 45, np.nan],
    "income": [50.0, 120.0, 65.0, np.nan, 95.0, 70.0, 80.0, 110.0],
    "signup": ["2024-01-05", "2024-01-20", "2024-02-03", "2024-02-14",
               "2024-02-28", "2024-03-09", "2024-03-17", "2024-03-30"],
    "bought": [0, 1, 0, 1, 1, 0, 1, 0],
})

print(df.shape)                        # (8, 6)：8 行 6 列
print(df.head(3))                      # 前 3 行；df.tail(3) 看最后 3 行
#    user_id city   age  income      signup  bought
# 0        1   NY  25.0    50.0  2024-01-05       0
# 1        2   SF  32.0   120.0  2024-01-20       1
# 2        3   NY   NaN    65.0  2024-02-03       0
print(df.dtypes["age"], df.dtypes["city"])   # float64 object：每列的类型，df.dtypes 一次看全部
df.info()                              # 每列非空个数 + 类型：age 只有 5 non-null，一眼看出缺失
print(df.describe().loc["mean"].round(1).to_dict())   # describe 只统计数值列：count/mean/std/分位数
# {'user_id': 4.5, 'age': 33.0, 'income': 84.3, 'bought': 0.5}
print(df["city"].value_counts().to_dict())                  # {'NY': 4, 'SF': 3, 'LA': 1}
print(df["city"].value_counts(normalize=True).to_dict())    # {'NY': 0.5, 'SF': 0.375, 'LA': 0.125}
```

考试里读文件写 `pd.read_csv("data/train.csv")`，CSV 里的空字段会读成 NaN；本节末尾的小题用 `io.StringIO` 把字符串当文件读。`age` 本来是整数，因为有 NaN 被存成 `float64`（NumPy 的整数类型装不下 NaN）；字符串列的类型是 `object`。

### 选列、选行、过滤

```python
age = df["age"]                        # 单括号：一列，Series，形状 (8,)
sub = df[["city", "age"]]              # 双括号：多列，DataFrame，形状 (8, 2)

# loc 按「标签」取，iloc 按「位置」取
print(df.loc[2, "city"], df.iloc[2, 1])        # NY NY：行标签 2 的 city 列 / 第 3 行第 2 列
print(len(df.loc[1:3]), len(df.iloc[1:3]))     # 3 2：loc 切片包含右端，iloc 不包含

# 布尔过滤：每个条件都加括号，用 & | ~ 组合
mask = (df["city"] == "NY") & (df["age"] > 30)                          # (8,) 的 True/False
print(df.loc[mask, "user_id"].tolist())                                 # [6, 7]
print(df.loc[(df["age"] < 30) | (df["city"] == "LA"), "user_id"].tolist())   # [1, 4, 5]
print(df.loc[df["city"].isin(["SF", "LA"]), "user_id"].tolist())       # [2, 4, 5, 8]
print(df.loc[~df["city"].isin(["SF", "LA"]), "user_id"].tolist())      # [1, 3, 6, 7]
print(df.loc[df["income"].between(60, 95), "user_id"].tolist())        # [3, 5, 6, 7]：两端都包含
```

`df.loc[mask, "user_id"]` 一步完成「过滤行 + 选列」，比 `df[mask]["user_id"]` 好：后者是两次索引，用来赋值时会出问题（见下面的高频坑 1）。和 NaN 比较的结果都是 `False`，所以 `df["age"] > 30` 不会选中缺失年龄的行。

### 新增与修改列

```python
d = df.copy()                                                  # 先复制，不动原始 df
d["income_k"] = d["income"] / 1000                             # 整列运算，NaN 自动传播
d["is_senior"] = np.where(d["age"] >= 30, 1, 0)                # 向量化 if-else
d["region"] = d["city"].map({"NY": "east", "SF": "west", "LA": "west"})   # 字典映射
print(d["is_senior"].tolist())    # [0, 1, 0, 0, 0, 1, 1, 0]：NaN >= 30 是 False，缺失年龄被当成 0
print(d["region"].tolist())       # ['east', 'west', 'east', 'west', 'west', 'east', 'east', 'west']

slow = d.apply(lambda row: row["income"] / row["age"], axis=1)   # 逐行调用 Python 函数
fast = d["income"] / d["age"]                                    # 一次向量化运算
print(np.allclose(slow, fast, equal_nan=True))                   # True
```

`map` 遇到字典里没有的键会得到 NaN，上线后来了新城市要记得处理。`apply(axis=1)` 每一行都要构造一个 Series 再调一次 Python 函数，所以很慢。在写这页用的机器（Apple M1 Pro）上，10 万行的 `apply` 要 0.3~0.4 秒，向量化只要约 0.05 毫秒，差了三个数量级以上；具体倍数随硬件和版本变化。能用列运算、`np.where`、`map` 写的，就不要用 `apply`。

### 缺失值

```python
print(df.isna().sum().to_dict())
# {'user_id': 0, 'city': 0, 'age': 3, 'income': 1, 'signup': 0, 'bought': 0}

d = df.copy()
# 1. 整列用中位数填（也可以用 mean；中位数不怕极端值）
d["income"] = d["income"].fillna(d["income"].median())            # 中位数是 80.0
# 2. 按组填：每个人用自己城市的平均年龄
city_mean = d.groupby("city")["age"].transform("mean")             # (8,)，和 d 逐行对齐
d["age"] = d["age"].fillna(city_mean)
# 3. LA 整组都缺失，transform 得到的也是 NaN，再用全局中位数兜底
d["age"] = d["age"].fillna(df["age"].median())
print(d["age"].tolist())     # [25.0, 32.0, 35.0, 32.0, 28.0, 35.0, 45.0, 30.0]
print(df.dropna().shape, df.dropna(subset=["income"]).shape)   # (5, 6) (7, 6)：任一列缺失就删 / 只看 income
```

不要写 `d["age"].fillna(0, inplace=True)`：这是先取出一列再原地改（链式赋值）。它在 pandas 1.5、2.0、2.3 上都还能改到 `d`，但 2.3 会报 FutureWarning，提示从 pandas 3.0 起这种写法不再修改原表。统一写成 `d["age"] = d["age"].fillna(...)`。

### 类型转换：astype、日期、category

```python
d = df.copy()
d["signup"] = pd.to_datetime(d["signup"], format="%Y-%m-%d")    # object -> datetime64[ns]
d["month"] = d["signup"].dt.month                               # .dt 取日期的各个部分
d["weekday"] = d["signup"].dt.day_name()
d["days_since_first"] = (d["signup"] - d["signup"].min()).dt.days
d["city"] = d["city"].astype("category")                        # 取值很少的字符串列，省内存
d["city_code"] = d["city"].cat.codes                            # 按类别字母序编号：LA=0, NY=1, SF=2
row = d.loc[1]                                                  # 取一行，得到一个 Series
print(row["month"], row["weekday"], row["days_since_first"], row["city_code"])   # 1 Saturday 15 2
print(df["age"].fillna(0).astype(int).tolist())   # [25, 32, 0, 0, 28, 35, 45, 0]：先填再转
print(df["age"].astype("Int64").tolist())         # [25, 32, <NA>, <NA>, 28, 35, 45, <NA>]
```

有 NaN 的列直接 `astype(int)` 会报 `IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer`（它是 `ValueError` 的子类）。两种办法：先填再转；或者转成 pandas 的可空整数类型 `"Int64"`（大写 I），缺失值显示为 `<NA>`。

### groupby 聚合

```python
stats = df.groupby("city").agg(
    n_users=("user_id", "count"),          # 新列名=(原列名, 聚合函数)
    avg_income=("income", "mean"),
    buy_rate=("bought", "mean"),
)
print(stats.round(2))
#       n_users  avg_income  buy_rate
# city
# LA          1         NaN      1.00
# NY          4       66.25      0.25
# SF          3      108.33      0.67

print(df.groupby("city").size().to_dict())            # {'LA': 1, 'NY': 4, 'SF': 3}：每组行数
print(df.groupby("city")["age"].count().to_dict())    # {'LA': 0, 'NY': 3, 'SF': 2}：非空个数

# agg：每组压成一行（3 行）；transform：结果广播回原来每一行（8 行），可以直接当新列
diff = df["income"] - df.groupby("city")["income"].transform("mean")   # (8,)：收入减所在城市均值
print(diff.round(2).tolist())    # [-16.25, 11.67, -1.25, nan, -13.33, 3.75, 13.75, 1.67]
```

`groupby` 的结果把分组键放进了 index，想让 `city` 变回普通列，就接 `.reset_index()`，或者一开始写 `groupby("city", as_index=False)`。聚合函数写字符串 `"mean"`、`"sum"`；写成 `np.mean`、`np.sum` 在 pandas 2.3 上会报 FutureWarning（1.5、2.0 不报）。

### 排序与 Top-N

```python
by_income = df.sort_values("income", ascending=False)
print(by_income["user_id"].tolist())                    # [2, 8, 5, 7, 6, 3, 1, 4]：NaN 默认排最后
print(df.nlargest(3, "income")["user_id"].tolist())     # [2, 8, 5]：只要前几名时比全表排序快
print(df.sort_values(["city", "income"], ascending=[True, False])["user_id"].tolist())
# [4, 7, 6, 3, 1, 2, 8, 5]：先按 city 升序，同一个 city 内按 income 降序

# 每个城市收入最高的 2 人：先全表排序，再每组取前 2 行
top2 = by_income.groupby("city").head(2)
print(top2["user_id"].tolist())                         # [2, 8, 7, 6, 4]：LA 只有用户 4，收入缺失也会留下
```

### merge 与 concat

```python
users = df[["user_id", "city"]]
orders = pd.DataFrame({"user_id": [1, 1, 2, 5, 9], "amount": [30.0, 20.0, 50.0, 15.0, 99.0]})

for how in ["inner", "left", "outer"]:
    print(how, users.merge(orders, on="user_id", how=how).shape)
# inner (4, 3)    只留两边都有的 user_id；用户 1 有两单，变成两行
# left (9, 3)     保留全部 8 个用户，没下单的 amount 是 NaN
# outer (10, 3)   两边的并集；用户 9 只在 orders 里，city 是 NaN

spend = orders.groupby("user_id", as_index=False).agg(total=("amount", "sum"))   # 先按用户聚合
feat = users.merge(spend, on="user_id", how="left", validate="one_to_one")      # 再 left merge 回用户表
feat["total"] = feat["total"].fillna(0.0)
print(feat["total"].tolist())    # [50.0, 50.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0]

print(pd.concat([users, users], ignore_index=True).shape)      # (16, 2)：上下拼，index 重新编号为 0~15
print(pd.concat([users, df[["age"]]], axis=1).shape)           # (8, 3)：左右拼，按 index 对齐
```

「先 groupby 聚合、再 left merge 回主表、没有记录的填 0」是 Industry ML 题里很常见的特征工程套路。`validate` 是一个很便宜的保险：键意外重复时，merge 会悄悄把行数变多；写了 `validate="one_to_one"` 而 `orders` 的键有重复，就会直接报 `MergeError: Merge keys are not unique in right dataset; not a one-to-one merge`。

### pivot_table、去重、重置索引、转 NumPy

```python
d = df.assign(month=pd.to_datetime(df["signup"], format="%Y-%m-%d").dt.month)
table = d.pivot_table(index="city", columns="month", values="bought", aggfunc="sum", fill_value=0)
print(table)
# month  1  2  3
# city
# LA     0  1  0
# NY     0  0  1
# SF     1  1  0

logs = pd.DataFrame({"user_id": [1, 1, 2, 2, 3], "page": ["home", "home", "home", "cart", "home"]})
print(len(logs.drop_duplicates()))                                   # 4：完全相同的行只留一行
print(logs.drop_duplicates(subset=["user_id"], keep="last")["page"].tolist())   # ['home', 'cart', 'home']

ny = df[df["city"] == "NY"]
print(ny.index.tolist())                             # [0, 2, 5, 6]：过滤后 index 不连续
print(ny.reset_index(drop=True).index.tolist())      # [0, 1, 2, 3]；不写 drop=True 会多出一列 index

X = df[["age", "income"]].fillna(0.0).to_numpy(dtype=np.float32)   # (8, 2)，之后 torch.from_numpy(X) 就能喂给模型
y = df["bought"].to_numpy()                                         # (8,)
print(X.shape, X.dtype, y.shape)                                    # (8, 2) float32 (8,)
```

### 四个高频坑

**1. 链式赋值（SettingWithCopyWarning）。** `d[mask]["income"] = 0` 先取出一个临时子表，再改这个子表，原表不变，还会报 SettingWithCopyWarning。改原表就用一次 `.loc[行条件, 列名]`；想要一个独立的子表就显式 `.copy()`。

```python
d = df.copy()
d.loc[d["age"] > 40, "income"] = 0.0       # 一次 .loc 同时定位行和列；别写 d[d["age"] > 40]["income"] = 0.0
print(d.loc[6, "income"])                  # 0.0：用户 7 的年龄是 45
ny = d[d["city"] == "NY"].copy()           # 要独立子表就 .copy()
ny["flag"] = 1                             # 不报警告，也不影响 d
```

**2. `and` / `or` 不能连接 Series 条件。** `df[(df["age"] > 30) and (df["city"] == "NY")]` 会报 `ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().` 原因是 Python 的 `and` 要问「整个 Series 是真还是假」，Pandas 拒绝回答。逐元素的与、或、非要用 `&`、`|`、`~`。另外 `&` 的优先级比 `>`、`==` 高：`df["age"] > 30 & df["city"] == "NY"` 会被解析成 `df["age"] > (30 & df["city"]) == "NY"`，直接报 TypeError。所以每个条件都要加括号。

**3. `df.append` 在 pandas 2.0 删除了。** 在 2.x 上调用会报 `AttributeError: 'DataFrame' object has no attribute 'append'`（1.5 上是 FutureWarning）。在循环里逐行追加也很慢：每次都复制整张表，总共是 $O(n^2)$。正确写法是循环里先 `rows.append({"epoch": e, "loss": loss})` 把每行攒成 dict 放进 Python list，循环结束后 `pd.DataFrame(rows)` 建一次表；要拼已有的表就用上面的 `pd.concat`。

**4. pandas 2 的 groupby 均值遇到字符串列直接报错。** `df.groupby("city").mean()` 在 1.5 上会报 FutureWarning，并悄悄丢掉字符串列 `signup`；在 2.x 上直接报 TypeError，报错文字随版本不同（2.0 是 `Could not convert ... to numeric`，2.3 是 `agg function failed [how->mean,dtype->object]`）。先选出数值列再聚合，写成 `df.groupby("city")[["age", "income"]].mean()`；或者显式传 `df.groupby("city").mean(numeric_only=True)`。

### CodeSignal Industry ML 风格小题

题面（仿写）：`data/train.csv` 和 `data/test.csv` 有数值列 `age`、`income`，类别列 `city`，标签 `bought`。在 `solution.py` 里实现 `fill_missing_values`、`one_hot_encode`、`standardize`，让 `main()` 输出能直接喂给模型的特征。下面两个是这类题里很常见的预处理 bug。CodeSignal 在 GitHub 公开过一道自己招 ML 工程师用的 notebook 题（InternalMLHiring 仓库），题面只要求修好 notebook 里的问题（原话 *Most of the problems are associated with data preprocessing and model training*），它的预处理代码里就同时有这两个：

1. **用测试集自己的统计量做标准化。** 写成 `(test - test.mean()) / test.std()`，用到了整个测试集的信息，是一种数据泄漏；同一个原始值在 train 和 test 里还会被映射成不同的数，模型学到的尺度对不上。上线时一次只来一条样本，也根本算不出「测试集的均值」。正确做法：mean、std 只在 train 上算，train 和 test 都用这一组。缺失值的中位数同理。
2. **对 train 和 test 分别调用 `pd.get_dummies`。** 每张表只为自己出现过的类别建列：test 里没有 `LA` 就少一列 `city_LA`，test 独有的 `Austin` 又多出一列，列数和列顺序都和 train 对不上，模型要么报错，要么把特征喂错位置。修复：`test.reindex(columns=train.columns, fill_value=0)`，缺的列补 0，多的列丢掉，顺序和 train 一致。

```python
import io
from typing import Tuple

import numpy as np
import pandas as pd

NUMERIC_COLS = ["age", "income"]
CATEGORICAL_COLS = ["city"]
TARGET = "bought"

# 真题在 load_data(file_name) 里调 pd.read_csv；这里用字符串模拟 data/ 下的两个 CSV
TRAIN_CSV = "age,income,city,bought\n25,50,NY,0\n32,120,SF,1\n,65,NY,0\n41,,LA,1\n28,95,SF,1\n35,70,NY,0\n"
TEST_CSV = "age,income,city,bought\n30,,NY,0\n,100,SF,1\n52,85,Austin,1\n"


def fill_missing_values(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """数值列填 train 的中位数，类别列填 "unknown"。

    Args:
        train: (n_train, d) 特征表，不含标签列
        test: (n_test, d) 特征表，不参与计算统计量
    Returns:
        填好缺失值的 (train, test)，形状不变
    """
    fill_values = train[NUMERIC_COLS].median().to_dict()       # {'age': 32.0, 'income': 70.0}
    fill_values.update({col: "unknown" for col in CATEGORICAL_COLS})
    return train.fillna(fill_values), test.fillna(fill_values)  # fillna 返回新表，不改输入


def one_hot_encode(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """类别列做 one-hot，test 的列按 train 对齐。

    Args:
        train: (n_train, d) 特征表，没有缺失值
        test: (n_test, d) 特征表，没有缺失值
    Returns:
        (n_train, d2) 和 (n_test, d2)，列名和列顺序完全相同
    """
    train_enc = pd.get_dummies(train, columns=CATEGORICAL_COLS, dtype=int)
    test_enc = pd.get_dummies(test, columns=CATEGORICAL_COLS, dtype=int)
    test_enc = test_enc.reindex(columns=train_enc.columns, fill_value=0)   # 修复 bug 2
    return train_enc, test_enc


def standardize(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """数值列做 z-score，mean 和 std 只用 train 算。

    Args:
        train: (n_train, d2) 特征表
        test: (n_test, d2) 特征表
    Returns:
        标准化后的 (train, test)，形状不变
    """
    mean = train[NUMERIC_COLS].mean()                  # (n_num,)
    std = train[NUMERIC_COLS].std(ddof=0)              # (n_num,)，ddof=0 和 sklearn 的 StandardScaler 一致
    std = std.where(std > 1e-8, 1.0)                   # 常数列的 std 是 0 或 1e-17 这种浮点误差（见 0.1 节），换成 1
    train, test = train.copy(), test.copy()
    train[NUMERIC_COLS] = (train[NUMERIC_COLS] - mean) / std
    test[NUMERIC_COLS] = (test[NUMERIC_COLS] - mean) / std    # 修复 bug 1：用 train 的 mean/std
    return train, test


def main() -> None:
    """读数据 -> 填缺失 -> one-hot -> 标准化，最后自测。"""
    train = pd.read_csv(io.StringIO(TRAIN_CSV))                         # 真题：load_data("data/train.csv")
    test = pd.read_csv(io.StringIO(TEST_CSV))
    y_train = train[TARGET].to_numpy()                                   # (6,)
    X_train, X_test = train.drop(columns=[TARGET]), test.drop(columns=[TARGET])
    X_train, X_test = fill_missing_values(X_train, X_test)
    X_train, X_test = one_hot_encode(X_train, X_test)
    X_train, X_test = standardize(X_train, X_test)

    # 自测：hidden tests 常查的几件事
    assert list(X_train.columns) == list(X_test.columns)               # 列完全对齐
    assert X_train.notna().all().all() and X_test.notna().all().all()   # 没有残留 NaN
    assert np.allclose(X_train[NUMERIC_COLS].mean(), 0.0)              # train 均值为 0
    assert np.allclose(X_train[NUMERIC_COLS].std(ddof=0), 1.0)         # train 标准差为 1
    print(X_train.shape, y_train.shape, X_test.shape)
    print(X_test.round(2))
    print("all tests passed")


if __name__ == "__main__":
    main()
# (6, 5) (6,) (3, 5)
#     age  income  city_LA  city_NY  city_SF
# 0 -0.43   -0.36        0        1        0
# 1 -0.03    0.95        0        0        1
# 2  3.90    0.29        0        0        0
# all tests passed
```

看输出里的三处：第 2 行的 `Austin` 是 train 没见过的类别，三个 city 列都是 0；test 里没有 LA，`city_LA` 也补上了全 0 列；52 岁按 train 的尺度是 3.90，如果犯 bug 1 用 test 自己的 mean/std（`ddof=0`），会被算成 1.41，模型看到的就不是同一个特征了。这三个函数已经和 sklearn 的 `SimpleImputer(strategy="median")`、`OneHotEncoder(handle_unknown="ignore")`、`StandardScaler` 对拍过，150 多组随机输入（包括常数列、test 只有一行、test 的类别 train 全没见过）结果都一致。

### 关键追问

- **`loc` 和 `iloc` 什么时候结果不一样？** 过滤或排序之后 index 不再是 0, 1, 2, ...，这时 `loc[0]` 找标签为 0 的行，`iloc[0]` 找第一行。拿不准就先 `reset_index(drop=True)`。Series 的 `s[0]` 更乱：整数 index 上按标签取，字符串 index 上退回按位置取（pandas 2.3 会为此报 FutureWarning，1.5、2.0 不报）。所以一律写 `s.iloc[0]` 或 `s.loc[标签]`。
- **`transform` 和 `agg` 怎么选？** 要「每组一个数」做汇总表用 `agg`；要把组统计量放回每一行（组内填缺失、减组均值、算组内占比）用 `transform`，它返回的长度和原表相同。
- **pandas 1.5 和 2.x 还有哪些不一样？** `value_counts()` 返回的 Series 在 2.x 里名字叫 `count`（`normalize=True` 时叫 `proportion`），1.5 里沿用原列名，所以上面都转成 dict 再打印。`get_dummies` 默认的 dtype 在 1.5 是 `uint8`，2.x 是 `bool`，所以小题里显式传了 `dtype=int`。`.dt.month` 在 2.x 返回 `int32`，1.5 是 `int64`。混合类型的一行（如 `d.loc[1]`）是 object 类型，取出来的是 NumPy 标量，NumPy 2 下 `.tolist()` 会打印成 `np.int32(1)` 这种样子，所以上面逐个 print。
- **题目给的签名是 `fill_missing_values(df)`，只有一张表，怎么办？** 不要改签名，hidden tests 按给定签名调用，就在这张表上算统计量。可以口头补一句：生产中应该保存 train 的统计量，再拿去填测试集。
- **为什么不把 train 和 test 拼起来再 `get_dummies`？** 列能对齐，但列集合里混进了只在 test 出现的类别，等于用了测试集的信息；上线后新数据还会出现新类别。用 train 的列做 `reindex` 更稳，sklearn 里对应的是 `OneHotEncoder(handle_unknown="ignore")`。
- **为什么 `std(ddof=0)`？** pandas 的 `.std()` 默认 `ddof=1`（除以 n-1），sklearn 的 `StandardScaler` 用 `ddof=0`（除以 n），题目要和 sklearn 对拍时就得写 `ddof=0`。三个库的默认值对比见 0.1 节。
- **还有哪种泄漏常被埋在这类题里？** 标签泄漏（target leakage）：特征里留着由标签算出来的列。InternalMLHiring 那道题的标签 `CODE_QUALITY` 就是 `REVIEWER_1`、`REVIEWER_2` 两列的均值，这两列却还留在特征里。训练前要删掉这类列，以及预测时拿不到的列（比如事后才产生的字段）。

---

## 0.3 PyTorch 基本语法

专题页 2.5 节的 Transformer、2.6 节的解码、6.1 节的 loss、10.1 节的训练循环都用 PyTorch 写，这一节把它们用到的语法集中过一遍。PyTorch 的张量操作和 NumPy 几乎一一对应（`axis` 换成 `dim`），新东西主要是三块：**dtype 规则**、**autograd**、**nn.Module**。下面的代码块按顺序在同一个会话里运行，后面的块直接用前面导入的模块和定义的类。注释里的输出在 torch 1.13、2.4、2.8 上完全一样；版本之间行为不同的地方，正文写的是这三个版本的实测结果。

### 创建张量：dtype 与 device

```python
import numpy as np
import torch

a = torch.tensor([1, 2, 3])                   # 小写 tensor：从数据建，自动推断 dtype
b = torch.tensor([1.0, 2.0])
c = torch.Tensor([1, 2, 3])                   # 大写 Tensor：老式构造器，按默认浮点类型建，即 float32
d = torch.tensor(np.array([1.0, 2.0]))        # NumPy 默认 float64，转过来还是 float64
print(a.dtype, b.dtype, c.dtype, d.dtype)     # torch.int64 torch.float32 torch.float32 torch.float64

print(torch.zeros(2, 3).shape, torch.arange(5))   # torch.Size([2, 3]) tensor([0, 1, 2, 3, 4])

y_float, y_long = a.float(), b.long()         # float32 给 BCE / MSE 的 target；int64 给类别标签、token id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x32 = d.float().to(device)                    # float64 先转 float32；tensor.to() 不改原张量，要接住返回值
print(device, x32.device)                     # cpu cpu（没有 GPU 的机器上）
```

- 统一用小写 `torch.tensor`。大写 `torch.Tensor(3)` 会建一个长度为 3、**没有初始化**的张量，很容易误用。
- 浮点默认 float32（NumPy 是 float64），整数默认 int64，也就是 `torch.long`。类别标签、token id、下标一律用 long。int32 标签喂给 `CrossEntropyLoss`，1.13、2.4、2.8 都报 `expected scalar type Long but found Int`；int32 下标做 `gather`，1.13 和 2.4 报错，2.8 不报。在新版本上碰巧能跑的 int32 代码，换到旧环境就挂。
- float64 的数据喂给 float32 的 `nn.Linear` 会报 `RuntimeError: mat1 and mat2 must have the same dtype`。从 NumPy 来的数据记得 `.float()`。
- 模型和数据必须在同一个 device。`model.to(device)` 原地移动参数（返回的还是这个模型），`x.to(device)` 返回新张量。

### 形状操作

```python
x = torch.arange(6).view(2, 3)                # (2, 3)：[[0, 1, 2], [3, 4, 5]]
xt = x.t()                                    # (3, 2)：转置只改 stride，不搬数据
print(xt.is_contiguous())                     # False
print(xt.reshape(6))                          # tensor([0, 3, 1, 4, 2, 5])
print(xt.contiguous().view(6))                # tensor([0, 3, 1, 4, 2, 5])

B, T, H, D_h = 2, 5, 4, 8
q = torch.randn(B, T, H * D_h)                # (B, T, C)，C = H * D_h
q = q.view(B, T, H, D_h).transpose(1, 2)      # (B, T, H, D_h) -> (B, H, T, D_h)：拆多头
print(q.shape)                                # torch.Size([2, 4, 5, 8])
print(q.permute(0, 2, 1, 3).shape)            # torch.Size([2, 5, 4, 8])：permute 一次排好所有维

v = torch.zeros(3)                            # (3,)
print(v.unsqueeze(0).shape, v.unsqueeze(1).shape)   # torch.Size([1, 3]) torch.Size([3, 1])
print(torch.zeros(1, 3, 1).squeeze(-1).shape)       # torch.Size([1, 3])：只挤掉指定那一维

row = torch.tensor([[1, 2]])                  # (1, 2)
print(row.expand(3, 2).stride())              # (0, 1)：expand 不复制，3 行共享同一份内存
print(row.repeat(3, 1).stride())              # (2, 1)：repeat 真的复制了 3 份

p0, p1 = torch.zeros(2, 3), torch.ones(2, 3)
print(torch.cat([p0, p1], dim=0).shape)       # torch.Size([4, 3])：沿已有的维拼接
print(torch.stack([p0, p1], dim=0).shape)     # torch.Size([2, 2, 3])：新建一维再叠起来
```

- **view 和 reshape**：`view` 要求内存连续，返回共享内存的视图；`reshape` 能 view 就 view，不能就复制一份。`t()`、`transpose`、`permute`、`expand` 之后张量不再连续，这时 `xt.view(6)` 会报 `RuntimeError: view size is not compatible with input tensor's size and stride ... Use .reshape(...) instead.`，要么改用 `reshape`，要么先 `.contiguous()`。专题 2.5 节拼回多头时写 `x.transpose(1, 2).contiguous().view(...)`，就是这个原因。
- **squeeze 总是带 dim**：`squeeze()` 会挤掉所有大小为 1 的维，batch size 恰好为 1 时 `(1, C)` 被挤成 `(C,)`，后面的代码全错位。
- **expand 和 repeat**：`expand` 只能把大小为 1 的维扩出去，不占新内存，扩完之后不能原地写它（会报错，要先 `clone()`）；`repeat` 按倍数复制，任何维都能用。

### 索引：gather、topk 与 masked_fill

```python
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [0.5, 2.5, 0.3]])      # (N, C) = (2, 3)
y = torch.tensor([0, 1])                      # (N,)，long
print(logits[torch.arange(2), y])             # tensor([2.0000, 2.5000])：第 i 行取第 y[i] 列
print(logits.gather(1, y.unsqueeze(1)).squeeze(1))   # tensor([2.0000, 2.5000])
print(logits[logits > 1.0])                   # tensor([2.0000, 2.5000])：布尔掩码，结果拉成一维
top_vals, top_idx = logits.topk(2, dim=-1)    # 每行最大的 2 个，降序；值和下标都是 (N, 2)
print(top_idx.tolist())                       # [[0, 1], [1, 0]]：tolist() 转成 Python 列表

T = 3
causal = torch.tril(torch.ones(T, T))         # (T, T) 下三角：1 = 可见，0 = 屏蔽
scores = torch.zeros(T, T).masked_fill(causal == 0, -1e9)   # 屏蔽位置填 -1e9
print(torch.softmax(scores, dim=-1)[1])       # tensor([0.5000, 0.5000, 0.0000])
```

`gather(dim, index)` 的规则是 `out[i][j] = input[i][index[i][j]]`（dim=1 时），`index` 和输出同形状，所以要先 `unsqueeze(1)` 成 `(N, 1)`。取「每个样本真实类别的 logit」时它和花式索引等价，专题 6.1 节写 loss 会用到。`topk` 是专题 2.6 节 top-k 采样和 beam search 的基础。`masked_fill(mask, value)` 把 `mask` 为 True 的位置换成 `value`，返回新张量，原张量不变；专题 2.5 节的 causal mask 和 padding mask 都这么写。

### 广播、矩阵乘、归约

```python
x = torch.randn(4, 3)                         # (N, D)
mu = x.mean(dim=0, keepdim=True)              # (1, D)
print((x - mu).shape)                         # torch.Size([4, 3])：(4, 3) - (1, 3) 广播

A, W = torch.randn(2, 5, 8), torch.randn(8, 16)    # (B, T, C), (C, D)
print((A @ W).shape)                          # torch.Size([2, 5, 16])：@ 就是 matmul，前面的维当 batch
print(torch.bmm(A, torch.randn(2, 8, 7)).shape)     # torch.Size([2, 5, 7])：bmm 只收 3 维，不广播
q, k = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 5, 8)   # (B, H, T, D_h)
s1 = q @ k.transpose(-2, -1)                  # (B, H, T, T)
s2 = torch.einsum("bhqd,bhkd->bhqk", q, k)    # 同一件事，用下标写：d 在输出里消失，就是对 d 求和
print(torch.allclose(s1, s2, atol=1e-6))      # True

m = torch.tensor([[1.0, 5.0, 3.0],
                  [4.0, 2.0, 6.0]])           # (2, 3)
print(m.sum(dim=1))                           # tensor([ 9., 12.])：(2,)
print(m.sum(dim=1, keepdim=True).shape)       # torch.Size([2, 1])
values, indices = m.max(dim=1)                # 带 dim 的 max 同时返回最大值和下标；只要下标就用 argmax
print(values, indices, m.argmax(dim=-1))      # tensor([5., 6.]) tensor([1, 2]) tensor([1, 2])
```

广播规则和 NumPy 完全一样（0.1 节）。`keepdim=True` 对应 NumPy 的 `keepdims`，归约后保留那一维，方便接着广播（专题第 6 节 softmax 减最大值就靠它）。

### autograd：requires_grad、backward、no_grad

```python
w = torch.tensor([1.0, 2.0], requires_grad=True)    # 叶子张量，需要梯度
loss = (w ** 2).sum()                         # 标量：backward() 只能直接对标量调用
loss.backward()                               # 算 d loss / d w = 2w，结果存进 w.grad
print(w.grad)                                 # tensor([2., 4.])
loss = (w ** 2).sum()
loss.backward()
print(w.grad)                                 # tensor([4., 8.])：没清零，梯度累加了

with torch.no_grad():                         # 手动 SGD 一步：更新参数这一步不能进计算图
    w -= 0.1 * w.grad
w.grad.zero_()                                # 手动清零，optimizer.zero_grad() 起的就是这个作用
print(w)                                      # tensor([0.6000, 1.2000], requires_grad=True)
h = (w * 3).detach()                          # 从计算图上摘下来，h 不再追踪梯度
print(h.requires_grad, loss.item())           # False 5.0
```

- `.grad` 默认**累加**，这样可以把一个大 batch 拆成几次 backward 再更新（梯度累积，专题 10.1 节）。代价是每一步都要先 `optimizer.zero_grad()`。它的默认行为随版本变过：1.13 把 `.grad` 填成 0；2.0 起默认 `set_to_none=True`，直接把 `.grad` 设成 None（2.4、2.8 实测）。所以清零之后不要假设 `.grad` 还是张量，要读它先判断 `is None`。
- `torch.no_grad()` 用在验证、推理和手动更新参数，不建计算图，省内存也省时间。对需要梯度的叶子张量直接写 `w -= lr * w.grad` 会报 `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.`，必须包在 `no_grad` 里。
- `.item()` 把单元素张量变成 Python 数。累计 loss 要写 `total += loss.item()`，原因见专题 10.1 节。

### nn.Module

```python
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super(MLP, self).__init__()           # 1. 先调父类构造函数，否则给 self.fc1 赋值时报 AttributeError
        self.fc1 = nn.Linear(in_dim, hidden_dim)   # 2. 有参数的层写成属性，自动注册
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim)
        Returns:
            logits: (B, out_dim)，没有过 softmax
        """
        h = F.relu(self.fc1(x))               # (B, hidden_dim)
        h = self.dropout(h)                   # 训练时随机置零，eval 时原样通过
        return self.fc2(h)                    # (B, out_dim)


model = MLP(4, 8, 3)
print(model(torch.randn(5, 4)).shape)         # torch.Size([5, 3])：调 model(x)，不要直接调 forward
print(sum(p.numel() for p in model.parameters()))   # 67 = (4*8 + 8) + (8*3 + 3)
print([(name, tuple(p.shape)) for name, p in model.named_parameters()][:2])
# [('fc1.weight', (8, 4)), ('fc1.bias', (8,))]：nn.Linear 的 weight 是 (out, in)
```

模型里还有两类张量要区分：要训练的参数，和「跟着模型走但不训练」的张量（比如 causal mask、BatchNorm 的 running mean）。另外，普通 Python list 里的层不会被注册：`parameters()` 看不到，优化器不更新，`.to(device)` 不搬，`state_dict` 不存，而且不报任何错。层的列表要用 `nn.ModuleList`，字典用 `nn.ModuleDict`。

```python
class Stack(nn.Module):
    def __init__(self, num_layers: int = 3, dim: int = 4):
        super(Stack, self).__init__()
        self.scale = nn.Parameter(torch.ones(dim))                        # 参数：会被优化器更新
        self.register_buffer("mask", torch.tril(torch.ones(dim, dim)))    # buffer：不训练，但会保存、会跟着 .to()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.plain_layers = [nn.Linear(dim, dim) for _ in range(num_layers)]   # 错误示范：普通 list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, dim) -> (B, dim)"""
        for layer in self.layers:             # ModuleList 可以像 list 一样遍历
            x = layer(x)                      # (B, dim)
        return x * self.scale                 # (B, dim)


stack = Stack()
print(sum(p.numel() for p in stack.parameters()))   # 64 = 4 + 3 * (4*4 + 4)：plain_layers 的 60 个没算进来
print(list(stack.state_dict().keys())[:3])    # ['scale', 'mask', 'layers.0.weight']

emb = nn.Embedding(num_embeddings=100, embedding_dim=16)   # 词表 100，每个 token 16 维
tokens = torch.tensor([[1, 5, 9], [2, 0, 0]])              # (B, T) = (2, 3)，整数 token id
h = nn.LayerNorm(16)(emb(tokens))                          # (2, 3, 16)：LayerNorm 只归一化最后一维
ffn = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 16))   # 按顺序串起来
print(ffn(h).shape)                           # torch.Size([2, 3, 16])：nn.Linear 只作用在最后一维
```

**F 和 nn 怎么选**：有参数的层（Linear、Embedding、LayerNorm）用 `nn` 模块，在 `__init__` 里建好。没有参数的运算两种写法等价，`F.relu(x)` 和 `nn.ReLU()(x)` 一样，`F.cross_entropy` 和 `nn.CrossEntropyLoss()` 一样。Dropout 用 `nn.Dropout`，它自己会看 `model.train()` / `model.eval()`；`F.dropout(x, p)` 的 `training` 参数默认是 True，忘了传 `training=self.training` 的话 eval 时也在丢。

### 数据、训练骨架、保存与加载

接着用上面的 `MLP`。完整的训练 / 验证循环（记录 train 和 val loss、早停、用 `copy.deepcopy` 保存最优权重）见专题 10.1 节，这里只列骨架和保存加载的写法：

```python
import io

from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)                         # 固定种子：初始化、randn、shuffle 都可复现
X = torch.randn(64, 4)                        # (N, D)
y = (X[:, 0] + X[:, 1] > 0).long()            # (N,)，二分类标签，long
dataset = TensorDataset(X, y)                 # dataset[i] = (X[i], y[i])
loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(len(dataset), len(loader))              # 64 4

model = MLP(4, 16, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()                             # 1. 训练模式：Dropout 生效
    for xb, yb in loader:                     # xb: (16, 4)，yb: (16,)
        optimizer.zero_grad()                 # 2. 清掉上一步的梯度
        loss = criterion(model(xb), yb)       # 3. 前向 + loss，logits (16, 2)
        loss.backward()                       # 4. 反向
        optimizer.step()                      # 5. 更新参数

model.eval()                                  # 6. 评估模式：Dropout 关掉
with torch.no_grad():                         # 7. 不建计算图
    acc = (model(X).argmax(dim=1) == y).float().mean().item()   # bool 先转 float 才能 mean
print(f"train acc: {acc:.4f}")                # train acc: 1.0000

buffer = io.BytesIO()                         # 用内存代替文件；真实场景写 "model.pt"
torch.save(model.state_dict(), buffer)        # 只存参数和 buffer：一个「名字 -> 张量」的字典
buffer.seek(0)
restored = MLP(4, 16, 2)                      # 先建同样结构的模型，再灌参数
print(restored.load_state_dict(torch.load(buffer, weights_only=True)))   # <All keys matched successfully>
restored.eval()
with torch.no_grad():
    print(torch.equal(restored(X), model(X)))  # True
```

- 保存推荐只存 `state_dict`。`torch.load` 的 `weights_only` 参数 1.13 就有，默认值随版本变过：1.13 和 2.4 默认 False（2.4 不传这个参数会给 FutureWarning），2.6 起默认 True。2.8 上实测，`torch.save(model)` 存下的整个模型直接 `torch.load` 会报 `UnpicklingError`。加载 `state_dict` 时显式写 `weights_only=True`，三个版本行为一致，也没有警告。

### NumPy 互转

```python
arr = np.zeros(3)                             # float64
t = torch.from_numpy(arr)                     # 共享内存，dtype 保持 float64
arr[0] = 7.0
print(t)                                      # tensor([7., 0., 0.], dtype=torch.float64)
t_copy = torch.tensor(arr)                    # 复制一份，之后互不影响
out = torch.ones(2, requires_grad=True) * 2   # 带梯度的张量
print(out.detach().cpu().numpy())             # [2. 2.]
```

`out.numpy()` 会报 RuntimeError：带梯度的张量要先 `.detach()`，在 GPU 上的要先 `.cpu()`，统一写 `.detach().cpu().numpy()` 最省心。反过来 `.numpy()` 和 `from_numpy` 一样共享内存，改一边另一边跟着变。

### 常见坑

**1. loss 的 target：dtype 和形状要对上。**

```python
logits = torch.tensor([[2.0, 0.5, -1.0],
                       [0.1, 1.5, 0.3]])      # (N, C)，模型直接输出的分数，没过 softmax
target = torch.tensor([0, 1])                 # (N,)，long 类别下标
ce = F.cross_entropy(logits, target)
soft = F.cross_entropy(logits, F.one_hot(target, 3).float())   # (N, C) 的 float target：当成每类的概率
print(round(ce.item(), 4), torch.allclose(ce, soft))           # 0.3391 True
try:
    F.cross_entropy(logits, target.float())   # (N,) 的 float 标签
except RuntimeError as e:
    print(e)                                  # expected scalar type Long but found Float
bce = F.binary_cross_entropy_with_logits(logits[:, 0], torch.tensor([1.0, 0.0]))   # (N,) 对 (N,)，float
print(round(bce.item(), 4))                   # 0.4357
```

`CrossEntropyLoss` 的 target 有两种合法写法：`(N,)` 的 long 类别下标（最常用），或者和 logits 同形状的 float 概率（soft label，专题 6.1 节的 label smoothing 和蒸馏会用到）。`(N,)` 的 float 标签哪种都不符合，所以报错。`BCEWithLogitsLoss` 正好相反：target 要 float、形状和 logits 一样，传 long 会报 RuntimeError，`(N, 1)` 对 `(N,)` 会报 ValueError。两个 loss 都吃原始 logits，模型最后不要再接 softmax / sigmoid（原因见专题 6.1 节）。

**2. 原地操作破坏 autograd。** 很多算子的反向要用到前向的结果，原地改掉它，backward 时就会报错：

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.exp()                                   # exp 的反向要用输出 y 本身：d exp(x) / dx = exp(x)
y.add_(1)                                     # 原地改了 y（带下划线的方法都是原地操作）
try:
    y.sum().backward()
except RuntimeError as e:
    print(type(e).__name__)                   # RuntimeError：... modified by an inplace operation
y = x.exp() + 1                               # 换成非原地写法
y.sum().backward()
print(x.grad)                                 # tensor([2.7183, 7.3891])
```

**3. (N, 1) 和 (N,) 悄悄广播。** `nn.Linear(D, 1)` 输出 `(N, 1)`，回归标签通常是 `(N,)`，`pred - target` 会广播成 `(N, N)`，loss 照样算得出来（NumPy 版的例子见 0.1 节）。`F.mse_loss` 遇到这种情况只给 UserWarning，照样按广播算：3 个样本全部预测对，loss 却是 1.3333。先 `pred.squeeze(1)` 对齐形状，或者在 loss 前加一句 `assert pred.shape == target.shape`（专题 6.1 节的 `mse_loss` 就这么写）。

**4. 评估前忘了 `model.eval()`。** Dropout 还开着，同一个输入前向两次结果不同，`torch.no_grad()` 也不会关掉它。两者的分工见专题 10.1 节。

### 自测

```python
def test_torch_basics() -> None:
    torch.manual_seed(42)
    logits = torch.randn(8, 5)                # (N, C)
    target = torch.randint(0, 5, (8,))        # (N,)
    assert target.dtype == torch.long                                     # 1. randint 默认就是 int64
    picked = logits.gather(1, target.unsqueeze(1)).squeeze(1)   # (N,)
    assert torch.equal(picked, logits[torch.arange(8), target])           # 2. gather = 花式索引
    manual_ce = -F.log_softmax(logits, dim=1)[torch.arange(8), target].mean()
    assert torch.allclose(manual_ce, F.cross_entropy(logits, target))     # 3. 手写 CE 对拍内置
    pred, y = torch.randn(8, 1).squeeze(1), torch.randn(8)                # 形状先对齐成 (N,)
    assert torch.allclose(((pred - y) ** 2).mean(), F.mse_loss(pred, y))  # 4. 手写 MSE 对拍内置
    w = torch.randn(3, requires_grad=True)
    (w ** 2).sum().backward()
    assert torch.allclose(w.grad, 2 * w.detach())                        # 5. d sum(w^2) / dw = 2w
    torch.optim.SGD([w], lr=0.1).zero_grad()
    assert w.grad is None or not w.grad.any()                            # 6. 1.13 填 0，2.x 设成 None
    assert sum(p.numel() for p in Stack().parameters()) == 64             # 7. 普通 list 的层不算
    print("all tests passed")


if __name__ == "__main__":
    test_torch_basics()
```

### 关键追问

- **`view` 和 `reshape` 有什么区别？** `view` 只改形状不复制，要求内存连续，否则报错；`reshape` 能 view 就 view，不能就复制，拿不准时用它。`transpose`、`permute`、`expand` 之后的张量不连续，`.contiguous()` 按当前形状复制出一份连续的。
- **`detach()`、`torch.no_grad()`、`requires_grad_(False)` 有什么区别？** `detach()` 把一个张量从计算图上摘下来，和原张量共享数据；`no_grad` 让一段代码里的运算都不建图；`requires_grad_(False)` 冻住参数本身，微调时冻结 backbone 就这么写：`for p in model.backbone.parameters(): p.requires_grad_(False)`，再只把 `[p for p in model.parameters() if p.requires_grad]` 交给优化器。`model.eval()` 和梯度无关，它和 `no_grad` 的区别见专题 10.1 节。
- **叶子张量是什么？为什么中间结果的 `.grad` 是 None？** 自己建的 `requires_grad=True` 的张量和模型参数是叶子，`backward()` 只把梯度存进叶子的 `.grad`。`h = w * 2` 这种中间结果的梯度用完就丢，想看要在 backward 之前调 `h.retain_grad()`。
- **`nn.Parameter` 和 `register_buffer` 怎么选？** 要梯度、要被优化器更新的用 `nn.Parameter`；不训练但要随模型保存、随 `.to(device)` 移动的（causal mask、running mean）用 buffer。普通属性张量两样都不会发生。
- **`torch.no_grad()` 和 `torch.inference_mode()` 有什么区别？** 都不建计算图。`inference_mode` 更彻底，连张量的版本计数这类 autograd 记账也省掉，所以更快一点；代价是在它里面建的张量之后不能参与需要反向的计算，否则报 RuntimeError。1.13、2.4、2.8 都有，纯推理两个都行。
- **`torch.save(model)` 和 `torch.save(model.state_dict())` 选哪个？** 选后者。前者用 pickle 存整个对象，加载时依赖原来的类定义和模块路径，代码一重构就加载不了；而且 2.6 起 `torch.load` 默认 `weights_only=True`，直接拒绝加载它。

---

## 0.4 手撸矩阵计算

ML Core 测评的算法题（矩阵归一化、前向传播等）不考 sklearn 和 PyTorch，官方样题还要求不 import 任何库（见第 0 节）。矩阵只能用 list of lists 表示：`A[i]` 是第 i 行，`A[i][j]` 是第 i 行第 j 列，形状 $(n, m)$ 就是 `(len(A), len(A[0]))`。本节把常考的矩阵运算用纯 Python 写一遍，写法和 CodeSignal 单函数题一致：辅助函数放在模块顶层，入口是 `solution(...)`，输入输出都是 list。

本节的约定：**题解代码是纯 Python，一个 import 都没有**，`math`、`typing` 也不用，所以类型标注只写 `list`、`dict`、`float` 这些内置类型。空矩阵 `[]` 直接返回 `[]`；形状对不上、行长不一致就 `raise ValueError`，报错信息写清两边的形状。每段题解后面跟一段自测，用 NumPy 算参考答案来对拍。**NumPy 只出现在本地自测里**，提交时只交纯 Python 的题解。

### 形状检查与转置

list of lists 不会帮你检查「每行一样长」，后面的函数都先用 `get_shape` 查一遍。转置 $(A^\top)_{ji}=A_{ij}$ 把形状 $(n,m)$ 变成 $(m,n)$，原来的第 i 行变成第 i 列；按列算统计量时（比如后面的归一化），先转置再按行处理最顺手。

```python
def get_shape(A: list) -> tuple:
    """返回 (行数, 列数)。空矩阵记为 (0, 0)；每一行必须一样长。"""
    n_rows = len(A)
    n_cols = len(A[0]) if n_rows > 0 else 0
    for i, row in enumerate(A):
        if len(row) != n_cols:
            raise ValueError(f"ragged matrix: row 0 has {n_cols} columns, row {i} has {len(row)}")
    return n_rows, n_cols


def transpose(A: list) -> list:
    """A: (n, m) -> (m, n)"""
    n, m = get_shape(A)
    return [[A[i][j] for i in range(n)] for j in range(m)]   # 外层走列 j，内层走行 i
```

- 转置的一行写法是 `[list(col) for col in zip(*A)]`。`zip(*A)` 等价于 `zip(A[0], A[1], ...)`，每次从所有行里各取一个元素，正好是一列。它产出 tuple，要转成 list：`[[1, 3]] == [(1, 3)]` 的结果是 `False`，测试会判错。zip 遇到长短不一的行会悄悄截断，所以也要先 `get_shape`。
- 非方阵是最常见的坑：内外层都写成 `range(n)`，等于默认行数等于列数，$(2,3)$ 的矩阵会丢掉第 3 列，$(3,2)$ 的会越界。

### 矩阵乘法

$C_{ij}=\sum_k A_{ik}B_{kj}$：C 的第 i 行第 j 列，等于 A 的第 i 行和 B 的第 j 列做点积。形状规则 $(n,m)\times(m,p)\to(n,p)$，中间的 m 必须相等。核心版是照着公式写的 i-j-k 三重循环，最内层 `C[i][j] += A[i][k] * B[k][j]`。面试版加上形状检查，并把循环换成 i-k-j 顺序：

```python
def matmul(A: list, B: list) -> list:
    """A: (n, m), B: (m, p) -> A @ B: (n, p)"""
    n, m = get_shape(A)
    m_b, p = get_shape(B)
    if n == 0:                                   # 空输入：没有行要算
        return []
    if m != m_b:
        raise ValueError(f"shape mismatch: A is ({n}, {m}), B is ({m_b}, {p}); need A cols == B rows")
    C = [[0.0] * p for _ in range(n)]            # (n, p)，每一行都是新的 list
    for i in range(n):
        row_c = C[i]                             # (p,)：C 的第 i 行
        for k in range(m):
            a_ik = A[i][k]                       # 标量，整个内层循环都复用它
            row_b = B[k]                         # (p,)：B 的第 k 行
            for j in range(p):
                row_c[j] += a_ik * row_b[j]      # 内层沿着同一行从左往右走
    return C


def solution(A: list, B: list) -> list:
    return matmul(A, B)
```

两种顺序的乘加次数一样，都是 $nmp$ 次（约 $2nmp$ 次浮点运算）：时间 $O(nmp)$，方阵就是 $O(n^3)$，额外空间是结果的 $O(np)$。为什么换成 i-k-j：

- i-j-k 的最内层是 k 在变，`B[k][j]` 每一步换一行，等于竖着按列读 B。i-k-j 的最内层是 j 在变，`B[k][j]` 和 `C[i][j]` 都在同一行里从左往右读。C 和 NumPy 的数组按行优先（row-major）连续存储，顺着读能一直命中 CPU 缓存，所以 C 代码里 i-k-j 更快。
- 纯 Python 的 list 存的是指向 float 对象的指针，缓存的差别被解释器开销盖住了，提速来自另一点：内层循环里 `A[i][k]`、`B[k]`、`C[i]` 都不变，可以提到循环外，每次乘加少几次索引。本机在 150×150 的随机矩阵上实测（5 次取最快），提了行的 i-k-j 比 i-j-k 快约 1.5 倍；只换顺序、不提行，两者几乎一样快。

**坑：结果矩阵不要写成 `[[0.0] * p] * n`。** 外层的 `* n` 复制的是同一个行对象的引用：`C = [[0] * 2] * 2` 之后执行 `C[0][0] = 1`，C 变成 `[[1, 0], [1, 0]]`，两行一起被改了。

有了转置，还能把内层循环交给 C 实现的 `zip` 和 `sum`：C 的每个元素是「A 的一行」和「$B^\top$ 的一行」的点积。本机实测这个写法比 i-k-j 版再快 1.2 到 1.6 倍。

```python
def matmul_zip(A: list, B: list) -> list:
    """A: (n, m), B: (m, p) -> (n, p)"""
    if len(A) > 0 and get_shape(A)[1] != len(B):  # zip 遇到长度不一致会悄悄截断，形状要自己查
        raise ValueError(f"shape mismatch: A has {len(A[0])} columns, B has {len(B)} rows")
    B_T = transpose(B)                                                             # (p, m)
    return [[sum(a * b for a, b in zip(row, col)) for col in B_T] for row in A]   # (n, p)
```

自测：和 NumPy 的 `A @ B` 对拍随机矩阵（含非方阵、$1\times N$、$N\times1$），再测空输入、形状不匹配和行长不一致。`assert_close`、`raises_value_error` 后面的自测都会复用。

```python
import numpy as np


def assert_close(mine: list, ref: np.ndarray) -> None:
    """对拍：形状和数值都要和 NumPy 的参考答案一致"""
    assert np.shape(mine) == ref.shape and np.allclose(mine, ref), (np.shape(mine), ref.shape)


def raises_value_error(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def test_matmul() -> None:
    rng = np.random.default_rng(0)
    for n, m, p in [(1, 1, 1), (2, 3, 4), (5, 1, 3), (1, 4, 1), (4, 4, 4)]:
        A, B = rng.standard_normal((n, m)), rng.standard_normal((m, p))
        assert_close(solution(A.tolist(), B.tolist()), A @ B)
        assert_close(matmul_zip(A.tolist(), B.tolist()), A @ B)
        assert transpose(A.tolist()) == A.T.tolist()                          # 必须是 list，不能是 tuple
    assert solution([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19.0, 22.0], [43.0, 50.0]]
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert solution([], [[1.0, 2.0]]) == [] and transpose([]) == []           # 空输入
    for fn in (solution, matmul_zip):
        assert raises_value_error(fn, [[1.0, 2.0]], [[1.0, 2.0]])             # (1, 2) x (1, 2)
        assert raises_value_error(fn, [[1.0, 2.0], [3.0]], [[1.0], [2.0]])    # 行长不一致
    print("matmul: all tests passed")


if __name__ == "__main__":
    test_matmul()
```

### 按列归一化：min-max 与 z-score

ML Core 考纲里的「矩阵归一化」一般指按列归一化：每一列是一个特征，各自缩放到可比的范围。下标 j 表示统计量都按第 j 列算：

$$
x'_{ij}=\frac{x_{ij}-\min_j}{\max_j-\min_j}\quad\text{(min-max)},\qquad
x'_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}\quad\text{(z-score)}
$$

σ 用总体标准差（除以 N），和 `np.std` 的默认值、sklearn 的 `StandardScaler` 一致；pandas 的 `.std()` 默认除以 N-1，题目写明用哪种就按题目来。

常数列的分母是 0，这里约定输出全 0。0.1 节的 NumPy 版用的是同一套约定（总体标准差，常数列输出 0），sklearn 的两个 Scaler 也把这种分母当成 1。判断常数列要比较原始数据，`max == min` 是精确的。`std == 0` 靠不住：`[0.1, 0.1, 0.1]` 的均值算出来是 0.10000000000000002，σ 是 1.3877787807814457e-17，一除全变成 -1.0。`np.std` 在这一列上给出同一个 σ，所以 NumPy 版也要用原始数据判断常数列。

```python
def min_max_normalize(X: list) -> list:
    """X: (N, D)，每列缩放到 [0, 1]；常数列输出 0"""
    if len(X) == 0:
        return []
    cols = transpose(X)                                            # (D, N)：按列算统计量更顺手
    lo, hi = [min(c) for c in cols], [max(c) for c in cols]        # (D,), (D,)
    return [[(v - l) / (h - l) if h > l else 0.0 for v, l, h in zip(row, lo, hi)]
            for row in X]                                          # (N, D)


def z_score_normalize(X: list) -> list:
    """X: (N, D)，每列减均值、除以总体标准差；常数列输出 0"""
    if len(X) == 0:
        return []
    N, cols = len(X), transpose(X)                                 # cols: (D, N)
    mean = [sum(c) / N for c in cols]                              # (D,)
    std = [(sum((v - mu) ** 2 for v in c) / N) ** 0.5             # (D,)：两遍法，先均值再偏差平方
           for c, mu in zip(cols, mean)]
    const = [max(c) == min(c) for c in cols]                       # (D,)：用原始数据判断常数列
    return [[0.0 if is_c else (v - mu) / s for v, mu, s, is_c in zip(row, mean, std, const)]
            for row in X]                                          # (N, D)


def solution(X: list, method: str = "z-score") -> list:
    """X: (N, D)，每列一个特征；method 是 "z-score" 或 "min-max" -> (N, D)"""
    if method == "min-max":
        return min_max_normalize(X)
    if method == "z-score":
        return z_score_normalize(X)
    raise ValueError(f"unknown method: {method!r}")
```

时间 $O(ND)$，转置多占 $O(ND)$ 空间。自测先跑 0.1 节的样例，再拿 NumPy 的公式对拍带常数列的随机矩阵：

```python
def test_normalize() -> None:
    out = solution([[1, 10, 5], [2, 20, 5], [3, 30, 5]])           # 0.1 节的样例
    assert [[round(v, 4) for v in row] for row in out] == [[-1.2247, -1.2247, 0.0], [0.0, 0.0, 0.0], [1.2247, 1.2247, 0.0]]
    X = np.random.default_rng(3).standard_normal((6, 4))
    X[:, 2] = 0.1                                                  # 第 3 列是常数列
    const = X.max(axis=0) == X.min(axis=0)                         # (D,)
    span = np.where(const, 1.0, X.max(axis=0) - X.min(axis=0))     # 常数列的分母换成 1，免得除以 0
    std = np.where(const, 1.0, X.std(axis=0))                      # np.std 默认 ddof=0，除以 N
    assert_close(solution(X.tolist(), "min-max"), np.where(const, 0.0, (X - X.min(axis=0)) / span))
    assert_close(solution(X.tolist(), "z-score"), np.where(const, 0.0, (X - X.mean(axis=0)) / std))
    assert solution([[0.1], [0.1], [0.1]]) == [[0.0], [0.0], [0.0]]   # 用 std == 0 判断会得到 -1.0
    assert solution([[4.0, 7.0]], "min-max") == [[0.0, 0.0]]       # 只有一行：每列都是常数
    assert solution([]) == [] and raises_value_error(solution, [[1.0]], "l2")
    print("normalize: all tests passed")


if __name__ == "__main__":
    test_normalize()
```

### 前向传播：一个隐藏层的全连接网络

ML Core 考纲里的「前向传播」：给定权重，算出网络输出。结构是 Linear → 激活 → Linear → Softmax，即 $H=\phi(XW_1+b_1)$，$P=\operatorname{softmax}(HW_2+b_2)$。

X 是 $(N,D_{in})$，$W_1$ 是 $(D_{in},D_h)$，$W_2$ 是 $(D_h,C)$，输出 P 是 $(N,C)$，每行是一个样本在 C 个类别上的概率。W 存成 (in, out)，和专题第 11 节的 NumPy 层是同一个约定；PyTorch 的 `nn.Linear` 把权重存成 (out, in)，题目给的权重要先看清是哪种。

除了上面的 `matmul`，还差两块积木：

- **加偏置（手写广播）。** NumPy 的 `XW + b` 会把 $(D,)$ 的 b 广播到每一行；纯 Python 没有广播，要自己把同一个 b 加到每一行上（下面的 `add_bias`）。按列广播（形状 $(N,1)$，比如每一行减去这一行自己的均值）也一样写，把 `bj` 换成「每行一个数」。矩阵乘向量同理，$y=Ax$ 就是 `[sum(a * b for a, b in zip(row, x)) for row in A]`，同样要先检查 `len(x)`。
- **指数。** 不能 `import math`，就用 `E ** z` 代替 `math.exp(z)`：`E = 2.718281828459045` 和 `math.e` 完全相等，本机在 $[-100,100]$ 上和 `math.exp` 的相对误差最大约 5e-15。纯 Python 的浮点溢出会直接抛异常，`E ** 1000` 抛出 `OverflowError: (34, 'Result too large')`，不像 NumPy 返回 `inf` 加一个警告。所以 softmax 要先减最大值，sigmoid 要按正负分两支（原理见专题第 4 节和第 6 节）。

```python
E = 2.718281828459045       # 自然常数 e，等于 math.e


def add_bias(Y: list, b: list) -> list:
    """Y: (N, D), b: (D,) -> (N, D)。手写广播：同一个 b 加到每一行"""
    N, D = get_shape(Y)
    if N == 0:
        return []
    if D != len(b):
        raise ValueError(f"shape mismatch: Y is ({N}, {D}), b has length {len(b)}")
    return [[y + bj for y, bj in zip(row, b)] for row in Y]      # (N, D)


def sigmoid(z: float) -> float:
    if z >= 0:                                     # 只对负数取指数，E ** 大正数会抛 OverflowError
        return 1.0 / (1.0 + E ** (-z))
    ez = E ** z
    return ez / (1.0 + ez)


def softmax_row(z: list) -> list:
    """z: (C,) -> (C,)，和为 1"""
    z_max = max(z)                                 # 先减最大值，最大的指数项是 E ** 0 = 1
    exps = [E ** (v - z_max) for v in z]           # (C,)
    total = sum(exps)
    return [e / total for e in exps]


ACTIVATIONS = {"relu": lambda z: max(z, 0.0), "sigmoid": sigmoid}


def solution(X: list, W1: list, b1: list, W2: list, b2: list, activation: str = "relu") -> list:
    """
    Args:
        X: (N, D_in)；W1: (D_in, D_h)；b1: (D_h,)；W2: (D_h, C)；b2: (C,)
        activation: "relu" 或 "sigmoid"
    Returns:
        (N, C)，每行是一个样本在 C 个类别上的概率，和为 1
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation: {activation!r}")
    if len(X) == 0:
        return []
    act = ACTIVATIONS[activation]
    Z1 = add_bias(matmul(X, W1), b1)               # (N, D_h)
    H = [[act(v) for v in row] for row in Z1]      # (N, D_h)：逐元素激活
    logits = add_bias(matmul(H, W2), b2)           # (N, C)
    return [softmax_row(row) for row in logits]    # (N, C)
```

时间 $O(N(D_{in}D_h+D_hC))$，几乎都花在两次矩阵乘法上；形状检查交给了 `matmul` 和 `add_bias`。专题第 11 节有同一个网络的 NumPy 向量化写法，还带反向传播。

```python
def test_forward() -> None:
    rng = np.random.default_rng(4)
    X, W1, b1, W2, b2 = (rng.standard_normal(s) for s in [(5, 3), (3, 4), (4,), (4, 2), (2,)])
    args = [a.tolist() for a in (X, W1, b1, W2, b2)]
    np_acts = {"relu": lambda z: np.maximum(z, 0.0), "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z))}
    for name, act in np_acts.items():
        logits = act(X @ W1 + b1) @ W2 + b2                           # (5, 2)
        ref = np.exp(logits - logits.max(axis=1, keepdims=True))
        assert_close(solution(*args, activation=name), ref / ref.sum(axis=1, keepdims=True))
    assert_close(add_bias((X @ W1).tolist(), b1.tolist()), X @ W1 + b1)   # NumPy 广播：(5, 4) + (4,)
    assert softmax_row([1000.0, 0.0]) == [1.0, 0.0]                   # 大 logits 不溢出
    assert sigmoid(-1000.0) == 0.0 and sigmoid(1000.0) == 1.0
    assert solution([], *args[1:]) == []
    assert raises_value_error(solution, [[1.0, 2.0]], *args[1:])      # X 是 (1, 2)，W1 是 (3, 4)
    assert raises_value_error(add_bias, [[1.0, 2.0]], [1.0, 2.0, 3.0])
    assert raises_value_error(solution, *args, activation="tanh")
    print("forward: all tests passed")


if __name__ == "__main__":
    test_forward()
```

### 稀疏矩阵乘法

这是 LeetCode 311（Sparse Matrix Multiplication），可以看成矩阵乘法的进阶版。矩阵里大部分是 0 时（one-hot 特征、词袋向量、图的邻接矩阵、用户和物品的评分矩阵），三重循环大部分时间在乘 0。改成只存非零元素：`{行号: {列号: 值}}`。

关键观察来自 i-k-j 顺序：固定 i 和 k 时，只要 `A[i][k] == 0`，整个内层 j 循环都白做。所以只遍历 A 第 i 行的非零元素，再只遍历 B 第 k 行的非零元素：

```python
def to_sparse(A: list) -> dict:
    """稠密 (n, m) -> {i: {j: A[i][j]}}，只存非零元素，全零的行整行不存"""
    sparse = {}
    for i, row in enumerate(A):
        nonzeros = {j: v for j, v in enumerate(row) if v != 0}
        if nonzeros:
            sparse[i] = nonzeros
    return sparse


def sparse_matmul(A_sp: dict, B_sp: dict) -> dict:
    """{i: {k: a}} 乘 {k: {j: b}} -> {i: {j: c}}"""
    C_sp = {}
    for i, row_a in A_sp.items():
        acc = {}                                           # C 的第 i 行，只记非零位置
        for k, a in row_a.items():                         # A 第 i 行的非零元素
            for j, b in B_sp.get(k, {}).items():           # B 第 k 行的非零元素
                acc[j] = acc.get(j, 0.0) + a * b
        if acc:
            C_sp[i] = acc
    return C_sp


def solution(A: list, B: list) -> list:
    """A: (n, m), B: (m, p)，都是稠密的 list of lists，大部分元素是 0 -> A @ B: (n, p)"""
    n, m = get_shape(A)
    m_b, p = get_shape(B)
    if n == 0:
        return []
    if m != m_b:
        raise ValueError(f"shape mismatch: A is ({n}, {m}), B is ({m_b}, {p})")
    C_sp = sparse_matmul(to_sparse(A), to_sparse(B))
    return [[C_sp.get(i, {}).get(j, 0.0) for j in range(p)] for i in range(n)]   # (n, p)
```

复杂度按非零元素算。A 的每个非零元素 $A_{ik}$ 要和 B 第 k 行的每个非零元素各乘一次，乘法总次数是 $\sum_{A_{ik}\ne0}\text{nnz}(B_{k,:})$，不超过 $\text{nnz}(A)$ 乘以 B 单行最多的非零个数。稠密输入转字典要扫一遍，$O(nm+mp)$；输出稠密矩阵要 $O(np)$。题目直接给稀疏格式时，这两部分都省掉了。矩阵本身不稀疏时，字典的开销会让它比普通三重循环更慢。

```python
def test_sparse() -> None:
    rng = np.random.default_rng(5)
    for n, m, p, density in [(4, 5, 3, 0.3), (6, 6, 6, 0.1), (3, 4, 2, 1.0), (2, 3, 2, 0.0), (1, 5, 1, 0.5)]:
        A = rng.standard_normal((n, m)) * (rng.random((n, m)) < density)   # 只留约 density 比例的非零
        B = rng.standard_normal((m, p)) * (rng.random((m, p)) < density)
        assert_close(solution(A.tolist(), B.tolist()), A @ B)
    A, B = [[1, 0, 0], [-1, 0, 3]], [[7, 0, 0], [0, 0, 0], [0, 0, 1]]
    assert to_sparse(A) == {0: {0: 1}, 1: {0: -1, 2: 3}}
    assert solution(A, B) == [[7.0, 0.0, 0.0], [-7.0, 0.0, 3.0]]
    assert solution([], [[1.0]]) == [] and raises_value_error(solution, [[1.0, 0.0]], [[1.0]])
    print("sparse matmul: all tests passed")


if __name__ == "__main__":
    test_sparse()
```

### 关键追问

- **纯 Python 和 NumPy 的矩阵乘法差在哪？** 复杂度都是 $O(n^3)$，差在常数。NumPy 的浮点矩阵乘调用 BLAS：C 实现、分块（tiling）让数据留在缓存里、SIMD 向量指令、多线程；纯 Python 每次乘加都要过解释器，还要新建 float 对象。
- **矩阵乘法能比 O(n³) 更快吗？** Strassen 算法把 2×2 分块乘法从 8 次乘法降到 7 次，复杂度 $O(n^{\log_2 7})\approx O(n^{2.81})$。它常数大、数值稳定性差，常用的 BLAS 实现还是 $O(n^3)$ 算法加分块优化。
- **归一化的统计量用哪部分数据算？** 只用训练集算 min、max 或 μ、σ，再用同一组数去变换验证集和测试集，否则就是数据泄漏（见 0.2 节）。测试集变换后超出 $[0,1]$ 是正常的。
- **min-max 和 z-score 怎么选？** min-max 对异常值敏感，一个极端值就能把其他点挤进很窄的区间；z-score 好一些，但均值和标准差也会被异常值拉偏，更稳的是中位数加四分位距（sklearn 的 `RobustScaler`）。KNN、K-Means、SVM 和用梯度下降训练的模型需要归一化；决策树按阈值切分，不需要。
- **方差为什么不用 E[x²] − E[x]² 一遍算完？** 两个很接近的数相减会丢掉有效数字。`[0.1, 0.1, 0.1]` 用这个公式算出的方差是 -1.734723475976807e-18，`[1e8 + 0.1] * 3` 算出 2.0，真实值都是 0。负数的 `** 0.5` 在 Python 里不报错，直接返回复数，后面全错。用两遍法（先求均值，再求偏差平方的均值），数据流场景用 Welford 在线算法。
- **工程里稀疏矩阵怎么存？** 常用 CSR（Compressed Sparse Row）：非零值数组、对应的列号数组、每行起点数组，`scipy.sparse.csr_matrix` 和 `torch.sparse_csr_tensor` 都支持。本节的「行号到字典」和 CSR 一样按行组织，能直接拿到某一行的全部非零元素；CSR 用三个连续数组代替字典，更省内存，遍历也更快。

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