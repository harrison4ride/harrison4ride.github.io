# 12. ML Coding 专题

[返回目录](README.md)

本页按专题整理常见手写模型（Transformer 与解码、经典 ML、神经网络层、NLP），以及 MLE 电面里的高频算法题（Top-K、蓄水池抽样）。CodeSignal 的题型、代码规范，NumPy、Pandas、PyTorch 的基本语法和纯 Python 矩阵运算在 [11. ML Coding 基础](11-ml-coding-basics.md)，正文里写作「基础篇 0.x 节」。面试时不只要写出能跑的代码，还要主动说明：

- 输入输出 shape。
- 时间/空间复杂度。
- 数值稳定性。
- 边界条件。
- 和框架实现的差异。

---

## 1. Scaled Dot-Product Attention

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

## 1.1 位置编码（白板讲解）

「白板介绍位置编码」是概念题，不用写长代码，3 到 5 分钟讲清楚三件事：为什么需要、有哪几类、各自加在哪里。概念细节见 [06. LLM 基础](06-llm-foundations.md) 的位置编码和 RoPE 两节，RoPE 的实现见 2.1 节，PyTorch 版正弦编码模块见 2.5 节。本节的 NumPy 代码很短，用来验证白板上说的每个性质。

### 白板顺序（先画什么，再写什么）

1. **问题（30 秒）**：写「猫追狗」和「狗追猫」，画 3 个 token 两两相连的 attention。说一句：attention 只算内容的两两点积，打乱顺序，输出只会跟着打乱，两句话里「猫」的表示完全一样。
2. **绝对位置（1 到 1.5 分钟）**：画「词向量 + 位置向量 = 输入」。先说查表（learned，BERT、GPT-2），再写正弦公式，旁边画几根转速不同的时钟指针。
3. **相对位置（1 分钟）**：画一个 $T\times T$ 的分数矩阵，在上面叠一个只依赖 $i-j$ 的偏置矩阵（同一条对角线上的值相同）。点出 Shaw 和 T5。
4. **ALiBi 和 RoPE（1 分钟）**：ALiBi 写出那个按距离线性递减的下三角矩阵；RoPE 画一个箭头旋转 $m\theta$，写一行 $\langle R_m q, R_n k\rangle = q^\top R_{n-m} k$。
5. **对比表（30 秒）**：「加在哪里、学不学、能不能外推、谁在用」四列，收尾。

### 原理与直觉：attention 不认顺序

不加 mask 时，用置换矩阵 $P$ 打乱输入的行，Q、K、V 的行跟着打乱，分数矩阵的行和列同时打乱，按行做的 softmax 不受影响，再用 $P^\top P=I$ 消掉中间一对：

$$
\operatorname{Attn}(PX)=\operatorname{softmax}\!\left(\frac{PQK^\top P^\top}{\sqrt{d_k}}\right)PV
=P\,\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)P^\top P\,V
=P\,\operatorname{Attn}(X)
$$

这叫**置换等变**（permutation equivariant）。FFN、LayerNorm 都对每个位置独立计算，同样不认顺序，所以不加位置信息的 encoder 看到的只是一个「词袋」。下面先验证这个性质，再验证加上 causal mask 之后它就不成立了（追问里会用到）。

```python
import numpy as np


def self_attention(x: np.ndarray, w_q: np.ndarray, w_k: np.ndarray, w_v: np.ndarray,
                   causal: bool = False) -> np.ndarray:
    """
    单头 self-attention，输入里不含任何位置信息。
    Args:
        x: (T, C)
        w_q, w_k, w_v: (C, C)
        causal: True 时加 causal mask
    Returns:
        (T, C)
    """
    q, k, v = x @ w_q, x @ w_k, x @ w_v                     # 各 (T, C)
    scores = q @ k.T / np.sqrt(x.shape[-1])                 # (T, T)
    if causal:
        visible = np.tril(np.ones(scores.shape, dtype=bool))
        scores = np.where(visible, scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)    # 稳定化
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)  # (T, T)，每行和为 1
    return weights @ v                                      # (T, C)


def test_permutation_equivariance() -> None:
    rng = np.random.default_rng(0)
    seq_len, d_model = 5, 8
    x = rng.normal(size=(seq_len, d_model))                 # (T, C)
    w_q, w_k, w_v = [rng.normal(size=(d_model, d_model)) for _ in range(3)]
    perm = rng.permutation(seq_len)

    # 1. 不加 mask：先打乱再算 == 先算再打乱
    out = self_attention(x, w_q, w_k, w_v)
    assert np.allclose(self_attention(x[perm], w_q, w_k, w_v), out[perm])

    # 2. 加 causal mask 后不再成立：mask 本身带了「谁在前面」的信息
    out_causal = self_attention(x, w_q, w_k, w_v, causal=True)
    assert not np.allclose(self_attention(x[perm], w_q, w_k, w_v, causal=True), out_causal[perm])
    print("all tests passed")


if __name__ == "__main__":
    test_permutation_equivariance()
```

### 绝对位置 1：可学习的位置表

最直接的做法：开一张 $(L_{\max}, d)$ 的表，第 pos 行就是位置 pos 的向量，和词向量表一起训练，输入是 `tok_emb[ids] + pos_emb[:T]`。BERT（$L_{\max}=512$）和 GPT-2（$L_{\max}=1024$）都这么做。局限是表只有 $L_{\max}$ 行，超过 $L_{\max}$ 的位置根本查不到向量，想加长只能扩表再训练。靠后的位置在训练中出现得少，也更难学好：BERT 预训练 90% 的步数用长度 128，最后 10% 才用 512，专门用来学后面那些位置向量。

### 绝对位置 2：正弦编码

原始 Transformer（Vaswani et al., 2017）用固定公式，不需要训练：

$$
PE_{(pos,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad
PE_{(pos,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

$d$ 是 d_model，$i$ 从 0 取到 $d/2-1$，每个 $i$ 对应一对 (sin, cos)。记 $\omega_i=10000^{-2i/d}$，两种直觉任选一种讲：

- **时钟指针**：每对 $(\sin(pos\,\omega_i), \cos(pos\,\omega_i))$ 是单位圆上的一根指针，位置每加 1 就转 $\omega_i$ 弧度。$i=0$ 转得最快，每步 1 弧度，约 6.28 个位置转一圈，像秒针；$i$ 越大转得越慢，$d=512$ 时最慢的一根约 60611 个位置才转一圈，像时针。低维快、高维慢，一个位置就是所有指针读数的组合。
- **平滑的二进制计数器**：二进制数里最低位每步翻转，越高的位翻得越慢。正弦编码把每一位的 0/1 跳变换成了连续的波。

**关键性质：平移 = 旋转。** 由和角公式，每一对都满足

$$
\begin{bmatrix}\sin((pos+k)\omega_i)\\ \cos((pos+k)\omega_i)\end{bmatrix}
=
\begin{bmatrix}\cos k\omega_i & \sin k\omega_i\\ -\sin k\omega_i & \cos k\omega_i\end{bmatrix}
\begin{bmatrix}\sin(pos\,\omega_i)\\ \cos(pos\,\omega_i)\end{bmatrix}
$$

右边的 2×2 矩阵只和 $k$ 有关，和 pos 无关。把 $d/2$ 个这样的块拼成块对角矩阵 $M_k$，就有 $PE_{pos+k}=M_k\,PE_{pos}$：模型用一个固定的线性变换就能表达「往后数 $k$ 个」，这是原论文选正弦的理由之一。推论：$M_k$ 是正交矩阵，所以 $PE_{pos}\cdot PE_{pos+k}=\sum_i\cos(k\omega_i)$，两个位置编码的相似度只和距离有关。

```python
import numpy as np


def sinusoidal_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Args:
        max_len: 编码多少个位置
        d_model: 编码维度，必须是偶数（sin/cos 成对）
    Returns:
        (max_len, d_model)，第 pos 行是位置 pos 的编码
    """
    assert d_model % 2 == 0, "d_model 必须是偶数"
    pos = np.arange(max_len)[:, None]                       # (L, 1)
    two_i = np.arange(0, d_model, 2)[None, :]               # (1, C/2)，公式里的 2i
    angle = pos / np.power(10000.0, two_i / d_model)        # (L, C/2)，广播
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle)                             # 偶数维放 sin
    pe[:, 1::2] = np.cos(angle)                             # 奇数维放 cos
    return pe


def shift_matrix(k: int, d_model: int) -> np.ndarray:
    """返回 (C, C) 块对角矩阵 M_k，满足 PE[pos + k] = M_k @ PE[pos]，与 pos 无关。"""
    omega = np.power(10000.0, -np.arange(0, d_model, 2) / d_model)   # (C/2,)
    m = np.zeros((d_model, d_model))
    for j, w in enumerate(omega):
        c, s = np.cos(k * w), np.sin(k * w)
        m[2 * j:2 * j + 2, 2 * j:2 * j + 2] = [[c, s], [-s, c]]    # 第 j 对指针转 k*w
    return m


def test_sinusoidal_encoding() -> None:
    max_len, d_model = 64, 16
    pe = sinusoidal_encoding(max_len, d_model)              # (64, 16)

    # 1. 形状、取值范围、位置 0 是 sin(0)=0 和 cos(0)=1 交替
    assert pe.shape == (max_len, d_model)
    assert np.all(np.abs(pe) <= 1.0)
    assert np.allclose(pe[0], np.tile([0.0, 1.0], d_model // 2))

    # 2. 和逐元素照抄公式的双重循环对拍
    ref = np.zeros((max_len, d_model))
    for p in range(max_len):
        for j in range(0, d_model, 2):
            ref[p, j] = np.sin(p / 10000 ** (j / d_model))
            ref[p, j + 1] = np.cos(p / 10000 ** (j / d_model))
    assert np.allclose(pe, ref)

    # 3. 平移 = 旋转：同一个 M_k 对所有 pos 都成立
    for k in (1, 5, 20):
        m = shift_matrix(k, d_model)                        # (C, C)
        assert np.allclose(pe[k:], pe[:-k] @ m.T)           # (L-k, C)
        assert np.allclose(m @ m.T, np.eye(d_model))        # 正交：只转不缩放
    print("all tests passed")


if __name__ == "__main__":
    test_sinusoidal_encoding()
```

### Example

用上面的 `self_attention` 和 `sinusoidal_encoding`，把白板上的几个说法跑一遍。

```python
rng = np.random.default_rng(0)
d_model = 8
emb = rng.normal(size=(3, d_model))                         # (V, C)，0=猫 1=追 2=狗
w_q, w_k, w_v = [rng.normal(size=(d_model, d_model)) for _ in range(3)]
cat_chases_dog = emb[[0, 1, 2]]                             # 「猫追狗」 (3, C)
dog_chases_cat = emb[[2, 1, 0]]                             # 「狗追猫」 (3, C)

# 1. 不加位置：「猫」在两句里的输出一模一样
out1 = self_attention(cat_chases_dog, w_q, w_k, w_v)
out2 = self_attention(dog_chases_cat, w_q, w_k, w_v)
print(np.allclose(out1[0], out2[2]))                        # True

# 2. 加上正弦编码：句首的「猫」和句尾的「猫」分开了
pe = sinusoidal_encoding(3, d_model)                        # (3, C)
out1 = self_attention(cat_chases_dog + pe, w_q, w_k, w_v)
out2 = self_attention(dog_chases_cat + pe, w_q, w_k, w_v)
print(np.allclose(out1[0], out2[2]))                        # False

# 3. 低维快、高维慢：第 0 列每步变化很大，第 6 列几乎不动
print(np.round(sinusoidal_encoding(4, 8), 3))
# [[ 0.     1.     0.     1.     0.     1.     0.     1.   ]
#  [ 0.841  0.54   0.1    0.995  0.01   1.     0.001  1.   ]
#  [ 0.909 -0.416  0.199  0.98   0.02   1.     0.002  1.   ]
#  [ 0.141 -0.99   0.296  0.955  0.03   1.     0.003  1.   ]]

# 4. 相似度只看距离 k = 0, 1, 4, 16：起点 10 和起点 30 结果相同，k=0 时是 d/2，越远越小
pe = sinusoidal_encoding(64, 16)
print([round(float(pe[10] @ pe[10 + k]), 4) for k in (0, 1, 4, 16)])  # [8.0, 7.4852, 5.5597, 4.214]
print([round(float(pe[30] @ pe[30 + k]), 4) for k in (0, 1, 4, 16)])  # [8.0, 7.4852, 5.5597, 4.214]
```

### 相对位置：把距离直接写进 attention 分数

「猫」和「狗」隔了几个词，往往比它们各自排在第几更重要。相对位置编码不改输入，在每一层 attention 打分时加一项只依赖 $i-j$ 的量，$s_{ij}=q_i^\top k_j/\sqrt{d_k}+b(i-j)$。最早的做法是 Shaw et al.（2018, *Self-Attention with Relative Position Representations*）：为每个相对距离（超过最大距离就截断）学一个向量，打分时加到 key 上（也加到 value 上）。T5（Raffel et al., 2020, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*）简化成每个 head 一个可学习的**标量** $b(i-j)$，各层共享；距离要分桶，近处一个距离一个桶，远处按对数合并，一共 32 个桶，超过 128 的距离共用一个桶。

ALiBi（Press et al., 2022, *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*）连偏置都不学：$b(i-j)=-m_h\,(i-j)$，离得越远罚得越多，输入端不加任何位置向量。每个 head 一个固定斜率 $m_h$，8 个 head 时取 $1/2, 1/4, \ldots, 1/256$。白板上画 4 个 token 的偏置矩阵（causal，右上角已被 mask）：

$$
B=-m_h\begin{bmatrix}0&&&\\1&0&&\\2&1&0&\\3&2&1&0\end{bmatrix}
$$

线性惩罚对任意距离都有定义，训练时没见过的更远距离只是罚得更重，所以外推好，BLOOM 用的就是 ALiBi。

RoPE（Su et al., 2021, *RoFormer: Enhanced Transformer with Rotary Position Embedding*）不加偏置，把 q 和 k 按各自位置旋转，点积里绝对角度抵消，只剩相对距离 $m-n$。可以把它看成正弦编码「平移 = 旋转」那条性质直接用在 Q、K 上。公式、实现和扩长见 2.1 节。

### 对比表

| 方法 | 加在哪里 | 可学习 / 固定 | 长度外推 | 代表模型 |
| --- | --- | --- | --- | --- |
| 正弦（Vaswani et al., 2017） | 输入 embedding | 固定 | 公式能算任意位置，但没训练过的长度效果不保证 | 原始 Transformer |
| 可学习绝对位置 | 输入 embedding | 可学习 | 不能超过表长 $L_{\max}$ | BERT、GPT-2 |
| 相对位置（Shaw 2018、T5） | attention 分数 | 可学习 | T5 远距离共用一个桶，任意长度都能算 | T5 |
| ALiBi | attention 分数 | 固定 | 好，论文的设计目标 | BLOOM |
| RoPE | Q、K（旋转后再点积） | 固定 | 原版一般，靠位置插值等方法扩长 | LLaMA |

### 关键追问

- **为什么相加而不是拼接？** 拼接会让维度变大，后面所有矩阵跟着变大。而且拼接后进第一个线性层时 $W[x; p]=W_x x+W_p p$，本来就是「各自投影再相加」，直接相加损失的表达力不多。常见的解释是高维空间里模型可以把语义和位置放进近似正交的子空间，相加后仍能用线性投影分开。
- **为什么是 10000？** 原论文的经验选择，没有推导。它让波长从 $2\pi$ 等比增长到约 $10000\cdot 2\pi$，最慢的指针在常见序列长度内转不满一圈，不同位置的读数组合不会重复。它是超参数：RoPE 沿用了 10000，Llama 3 为了更长的上下文把 RoPE 的 base 调到 500000。
- **为什么 sin 和 cos 都要？** 一是单靠 sin 定不了相位，$\sin a=\sin(\pi-a)$，两个不同位置可能读数相同，(sin, cos) 合起来是单位圆上的一个点，相位唯一。二是平移性质要用到：$\sin(a+b)=\sin a\cos b+\cos a\sin b$，想线性地得到 $\sin(a+b)$，必须同时有 $\sin a$ 和 $\cos a$。
- **位置信息在哪一层注入？** 绝对位置（可学习、正弦）只在最底层加一次，靠残差一路传上去；T5 偏置、ALiBi、RoPE 在每一层的 attention 打分里都重新作用一次，而且只影响打分（RoPE 只转 Q、K，不碰 V）。
- **decoder-only 不加位置编码行不行（NoPE）？** 可以学到一些位置：causal mask 让第 $i$ 个 token 恰好看到 $i$ 个 token，上面的自测也验证了加 mask 后置换等变不再成立，Haviv et al.（2022, *Transformer Language Models without Positional Encodings Still Learn Positional Information*）发现这样的语言模型和加了位置编码的版本表现接近；双向 encoder 没有 mask，去掉位置编码就真成了词袋。
- **长度外推怎么办？** 可学习位置直接不行；正弦能算但模型没见过；ALiBi 为外推设计；RoPE 最常用的是位置插值（Chen et al., 2023, *Extending Context Window of Large Language Models via Positional Interpolation*）：目标长度 $L'$ 上的位置 $m$ 缩成 $mL/L'$，让旋转角落回训练时见过的范围，再少量微调。NTK-aware、YaRN 等改进见 2.1 节和 [06. LLM 基础](06-llm-foundations.md)。
- **可学习和正弦哪个好？** 原论文两种都试过，结果几乎一样；选正弦是希望模型能外推到比训练更长的序列。

---

## 2. Multi-Head Attention

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

## 2.1 RoPE（旋转位置编码）

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

## 2.2 LayerNorm

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

## 2.3 Beam Search

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

## 2.4 Top-K / Top-P 采样

### 原理与直觉

Greedy 每步取最大值，输出死板且容易复读；直接按全词表采样又会偶尔抽到概率极低的垃圾 token（词表 15 万个，长尾加起来的概率不小）。两种截断办法：

- **Top-K**：只留概率最高的 $K$ 个，重新归一化后采样。简单，但 $K$ 是固定的——分布尖锐时放进太多垃圾，分布平坦时又砍掉合理选项。
- **Top-P（Nucleus）**：把 token 按概率从高到低排序，取**累计概率刚好超过 $p$** 的最小集合。候选集大小随分布自动伸缩，是它相对 Top-K 的核心优势。

温度在截断之前作用于 logits：$p_i=\dfrac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$，$T<1$ 让分布更尖，$T>1$ 更平。

### 实现

```python
import numpy as np


def softmax(x):
    """这里输入是单条一维 logits，所以可以用不带 axis 的简版（见第 6 节的说明）"""
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

## 2.5 手撸 Transformer（PyTorch）

### 原理与直觉

面试说「手写一个 Transformer」，最常见的是 GPT 那种 decoder-only 语言模型。它只有两种零件，反复堆叠：

- **Multi-Head Attention**：token 之间交换信息，每个位置去「看」别的位置。
- **FFN（前馈网络）**：每个 token 自己独立加工，不和别的位置交流。

每个零件外面包一层 LayerNorm 和残差连接，合起来叫一个 **Block**。记 B = batch，T = 序列长度，C = `d_model`，H = 头数，`d_k = C / H`，V = 词表大小，整个模型的数据流是：

`input_ids (B, T)` → token embedding + 位置编码 → `(B, T, C)` → N 个 Block → LayerNorm → `lm_head` → `logits (B, T, V)`

原理前面都讲过：attention 见第 1 节，多头见第 2 节，位置编码见 1.1 节，LayerNorm 见 2.2 节，`view`、`masked_fill`、`nn.ModuleList` 等语法见基础篇 0.3 节。本节用 CodeSignal Learn 课程的 `nn.Module` 写法把这些零件拼成一个能训练的模型，分 5 步，最后补一个 Encoder-Decoder 的延伸。时间紧时优先写第 1、2、4、5 步，位置编码几行带过。

### 第 1 步：Scaled Dot-Product Attention

公式和「为什么除以 $\sqrt{d_k}$」见第 1 节。PyTorch 版和 NumPy 版一一对应：`@` 换成 `torch.matmul`，`np.where(mask, scores, -1e9)` 换成 `scores.masked_fill(mask == 0, -1e9)`。

```python
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 dropout: Optional[nn.Module] = None
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: (B, H, T_q, d_k)；k: (B, H, T_k, d_k)；v: (B, H, T_k, d_v)
        mask: 可广播到 (B, H, T_q, T_k)，1 = 可见，0 = 屏蔽
        dropout: 作用在注意力权重上的 nn.Dropout，可选
    Returns:
        output: (B, H, T_q, d_v)；weights: (B, H, T_q, T_k)
    """
    d_k = q.size(-1)
    # 1. 打分并缩放：(B, H, T_q, d_k) @ (B, H, d_k, T_k) -> (B, H, T_q, T_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # 2. 屏蔽：mask 为 0 的位置填 -1e9，softmax 之后权重就是 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 3. 沿最后一维（T_k）归一化，每个 query 的权重和为 1
    weights = F.softmax(scores, dim=-1)                 # (B, H, T_q, T_k)
    if dropout is not None:
        weights = dropout(weights)
    # 4. 用权重加权求和 V
    output = torch.matmul(weights, v)                   # (B, H, T_q, d_v)
    return output, weights


if __name__ == "__main__":
    torch.manual_seed(42)
    q, k, v = torch.randn(3, 2, 4, 5, 8).unbind(0)      # 各自 (B=2, H=4, T=5, d_k=8)
    causal = torch.tril(torch.ones(5, 5))               # (T, T) 下三角，1 = 可见
    out, weights = scaled_dot_product_attention(q, k, v, causal)
    print(weights[0, 0, 1])   # tensor([0.2470, 0.7530, 0.0000, 0.0000, 0.0000])
    if hasattr(F, "scaled_dot_product_attention"):      # torch 2.0+ 才有；它的 bool mask 是 True = 可见
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=causal.bool())
        assert torch.allclose(out, ref, atol=1e-6)
```

位置 1 只把权重分给位置 0 和自己，后三个位置的权重正好是 0。torch 2.0 起有内置的 `F.scaled_dot_product_attention`（GPU 上会自动选 FlashAttention 等高效内核），上面在 torch 2.x 下和它对拍，torch 1.x 没有这个函数就跳过。mask 只要能**广播**到 `(B, H, T_q, T_k)` 就行：因果 mask 传 `(T, T)`，padding mask 传 `(B, 1, 1, T_k)`，两个一起用时直接相乘，`causal * pad` 广播成 `(B, 1, T, T)`。

### 第 2 步：MultiHeadAttention

切头、拼头的思路见第 2 节。这里先投影、切成 `(B, H, T, d_k)`，再用第 1 步的 `scaled_dot_product_attention` 一次算完 H 个头。PyTorch 写法只有一个坑：`transpose` 之后张量在内存里不连续，拼头时要先 `.contiguous()` 再 `view`，或者直接用 `reshape`（原因见基础篇 0.3 节）。写完和 `nn.MultiheadAttention` 对拍，有两个细节：

1. **权重布局**：PyTorch 把 $W_q,W_k,W_v$ 按行拼成一个 `(3C, C)` 的 `in_proj_weight`，顺序是 q、k、v。`nn.Linear` 的 `weight` 也是 `(out, in)` 布局，直接 `torch.cat(..., dim=0)` 拷过去。
2. **mask 约定**：我们 1 = 可见。`nn.MultiheadAttention` 的 bool `attn_mask` 是 **True = 屏蔽**，要传 `mask == 0`；padding 走另一个参数 `key_padding_mask`，形状 `(B, T_k)`，True = padding。torch 2.0 起的 `F.scaled_dot_product_attention` 的 bool mask 又是 True = 可见，传 float 的 0/1 mask 则会被当成加到分数上的偏置，等于没屏蔽。约定混用是最常见的 bug。

```python
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头注意力：投影 → 切头 → 各头并行算 attention → 拼回 → w_o 融合。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads                 # 每个头分到的维度
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)              # 作用在注意力权重上

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, H, T, d_k)"""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_k) -> (B, T, C)"""
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, C)；key、value: (B, T_k, C)
            mask: 可广播到 (B, H, T_q, T_k)，1 = 可见，0 = 屏蔽
        Returns:
            (B, T_q, C)
        """
        # 1. 线性投影，再切成多头
        q = self.split_heads(self.w_q(query))           # (B, H, T_q, d_k)
        k = self.split_heads(self.w_k(key))             # (B, H, T_k, d_k)
        v = self.split_heads(self.w_v(value))           # (B, H, T_k, d_k)
        # 2. H 个头一起算（H 维和 B 维一样当作 batch 维）
        out, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout)   # (B, H, T_q, d_k)
        # 3. 拼回 (B, T_q, C)，再过 w_o
        return self.w_o(self.combine_heads(out))


def test_mha_matches_pytorch() -> None:
    torch.manual_seed(42)
    batch_size, seq_len, d_model, num_heads = 2, 5, 16, 4
    x = torch.randn(batch_size, seq_len, d_model)                # (B, T, C)
    mha = MultiHeadAttention(d_model, num_heads)                 # dropout 默认 0，结果确定
    ref = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    # 1. 拷贝权重：W_q、W_k、W_v 按行拼成 (3C, C)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.w_q.weight, mha.w_k.weight, mha.w_v.weight], dim=0))
        ref.in_proj_bias.copy_(torch.cat([mha.w_q.bias, mha.w_k.bias, mha.w_v.bias]))
    ref.out_proj.load_state_dict(mha.w_o.state_dict())          # w_o 对应 out_proj，直接整层拷
    # 2. 无 mask、因果 mask、padding mask 各比一次；传给 PyTorch 的 mask 都是 True = 屏蔽
    causal = torch.tril(torch.ones(seq_len, seq_len))            # (T, T)，1 = 可见
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])     # (B, T)，第 2 句后两个是 padding
    cases = [(None, {}), (causal, {"attn_mask": causal == 0}),
             (valid[:, None, None, :], {"key_padding_mask": valid == 0})]
    for mask, kwargs in cases:
        out = mha(x, x, x, mask)                                 # (B, T, C)
        ref_out, _ = ref(x, x, x, **kwargs)
        assert out.shape == x.shape and torch.allclose(out, ref_out, atol=1e-6)
    print("MHA matches nn.MultiheadAttention")


if __name__ == "__main__":
    test_mha_matches_pytorch()   # MHA matches nn.MultiheadAttention
```

### 第 3 步：位置编码

attention 不认顺序，所以要把位置信息加进输入；为什么、有哪几类、正弦公式和它的性质见 1.1 节。下面是原论文正弦编码的 PyTorch 模块，数值和 1.1 节的 NumPy 版一致。要点是 `register_buffer`：编码表不参与训练（后面数 GPT 的参数量时不算它），但会跟着 `.to(device)` 移动、存进 `state_dict`；写成普通属性的话，模型搬到 GPU 后它还留在 CPU 上。换成 GPT-2 的可学习位置就是 `nn.Embedding(max_len, C)`；RoPE 改的是 attention 内部的 Q、K，见 2.1 节。

```python
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码：加到 token embedding 上，再做 dropout。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0, "d_model 必须是偶数（sin/cos 成对）"
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1).float()          # (max_len, 1)
        # 1/10000^(2i/C) 写成 exp(-2i * ln(10000) / C)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (C/2,)
        pe = torch.zeros(max_len, d_model)                              # (max_len, C)
        pe[:, 0::2] = torch.sin(position * div_term)                    # 偶数维放 sin，广播成 (max_len, C/2)
        pe[:, 1::2] = torch.cos(position * div_term)                    # 奇数维放 cos
        self.register_buffer("pe", pe.unsqueeze(0))                     # (1, max_len, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)"""
        x = x + self.pe[:, :x.size(1)]                                  # (1, T, C) 广播到 batch
        return self.dropout(x)
```

### 第 4 步：FFN 和 TransformerBlock（Pre-LN）

FFN 是两层全连接，中间维度通常放大 4 倍（原论文 512 → 2048）。它对每个位置**单独**做同样的变换，所以叫 Position-wise。一个 Block = Attention 子层 + FFN 子层，每个子层都是「LayerNorm → 子层 → dropout → 加回残差」。原论文用 Post-LN，$x\leftarrow\operatorname{LN}(x+\operatorname{Sublayer}(x))$；GPT-2 起改成 Pre-LN，$x\leftarrow x+\operatorname{Sublayer}(\operatorname{LN}(x))$，区别只在 LayerNorm 的位置。Pre-LN 的残差主干从头到尾不经过 LayerNorm，梯度能直接传到底层，训练更稳。代价是主干上的数值一直没被归一化，所以模型最后要补一个 `ln_f`。Block 里的 attention 用第 2 步的 `MultiHeadAttention`。

```python
from typing import Optional

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """Linear(C -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> C)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()                     # 原论文用 ReLU，GPT 系列用 GELU
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, d_ff) -> (B, T, C)"""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """decoder-only Block（Pre-LN）：x + Attn(LN(x))，再 x + FFN(LN(x))。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)              # 残差 dropout：子层输出加回主干之前

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, C)，mask: (T, T) 因果 mask（1 = 可见）。返回 (B, T, C)。"""
        # 1. 自注意力子层：Q、K、V 都来自同一个 LN(x)
        h = self.ln1(x)
        x = x + self.dropout(self.attn(h, h, h, mask))  # (B, T, C)
        # 2. FFN 子层
        x = x + self.dropout(self.ffn(self.ln2(x)))     # (B, T, C)
        return x
```

用第 2 步的方法拷贝权重后，这个 Block 和 `nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)` 输出一致（`src_mask` 同样要转成 True = 屏蔽）。`norm_first=True` 就是 Pre-LN；它叫 Encoder 只是因为没有 cross-attention，加上因果 mask 就是 GPT 的 Block。

### 第 5 步：拼成 GPT 语言模型

用上面的 `PositionalEncoding` 和 `TransformerBlock`。几个要点：

- **N 个 Block 放进 `nn.ModuleList`**：放在普通 Python list 里的子模块不会被注册，`model.parameters()` 找不到它们，优化器也就不会更新，而且不报错。
- **weight tying**：`lm_head.weight` 和 `tok_emb.weight` 形状都是 `(V, C)`，直接让它们指向同一个 Parameter。
- **loss**：语言模型的标签是输入右移一位，`targets[t] = input[t + 1]`。logits reshape 成 `(B*T, V)`、标签成 `(B*T,)` 再算交叉熵。padding 位置把标签设成 -100，`F.cross_entropy` 默认 `ignore_index=-100` 会跳过它们（6.1 节有手写版）。

```python
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT(nn.Module):
    """decoder-only 语言模型：Embedding + 位置 → N 个 Block → LayerNorm → lm_head。"""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 max_len: int, dropout: float = 0.0):
        super(GPT, self).__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, 4 * d_model, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)               # Pre-LN 必须补最后这个 LayerNorm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight       # weight tying：同一个 (V, C) 参数
        nn.init.normal_(self.tok_emb.weight, std=0.02)  # GPT-2 的小方差初始化，初始 logits 接近 0

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T)；targets: (B, T) 下一个 token 的 id，可选
        Returns:
            logits: (B, T, V)；loss: 标量，没给 targets 时为 None
        """
        seq_len = input_ids.size(1)
        assert seq_len <= self.max_len, "序列超过 max_len"
        # 1. 因果 mask：下三角为 1，位置 t 只能看到 <= t 的位置
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))   # (T, T)
        # 2. token embedding + 位置编码
        x = self.pos_enc(self.tok_emb(input_ids))       # (B, T, C)
        # 3. 依次过 N 个 Block
        for block in self.blocks:
            x = block(x, mask)                          # (B, T, C)
        # 4. 最后的 LayerNorm + 投影到词表
        logits = self.lm_head(self.ln_f(x))             # (B, T, V)
        loss = None
        if targets is not None:   # 展平成 (B*T, V) 和 (B*T,)；切片来的 targets 不连续，用 reshape
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


@torch.no_grad()
def generate(model: GPT, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """贪心解码：每步把整个前缀喂进去，只取最后一个位置。(B, T) -> (B, T + max_new_tokens)"""
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids[:, -model.max_len:])            # (B, T, V)，超长时只留最后 max_len 个
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)    # (B, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)          # (B, T + 1)
    return input_ids
```

`generate` 每生成一个 token 都把整个前缀重算一遍，KV cache 省的就是这部分重复计算。批量生成、采样、KV cache、beam search 的 PyTorch 写法见 2.6 节；那里的函数假设 `model(input_ids)` 直接返回 logits，套这里的 GPT 时取 `model(input_ids)[0]`。

### Example：自测

用上面的 `GPT` 和 `generate`。`test_gpt` 检查形状、loss、因果性、weight tying 和参数量；`overfit_one_batch` 是 10.1 节讲过的过拟合检查：在同一个小 batch 上训练几十步，loss 应该从 $\ln V$ 附近降到接近 0，降不下去就说明模型或训练循环有 bug。数据故意用 4 条起点不同的序列：只用一条 0 1 2 … 的话，位置 t 上的 token 恒等于 t mod 10，模型只靠位置编码就能把 loss 压到接近 0，学到的只是「位置 → token」的对照表（本机试过，从 5 6 7 开头会接着生成 3 4 5 …）。

```python
import torch
import torch.nn.functional as F


def test_gpt() -> None:
    torch.manual_seed(42)
    vocab_size, d_model, num_heads, num_layers, max_len = 10, 32, 4, 2, 32
    model = GPT(vocab_size, d_model, num_heads, num_layers, max_len, dropout=0.1).eval()   # eval 关掉 dropout
    # 1. 形状和 loss：标签 y 是 x 右移一位；loss 和手算的「正确 token 的平均 -log p」一致
    data = torch.randint(0, vocab_size, (2, 9))        # (B, T + 1)
    x, y = data[:, :-1], data[:, 1:]                   # 各 (B, T)
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, vocab_size) and loss.dim() == 0
    manual = -F.log_softmax(logits, dim=-1).gather(-1, y.unsqueeze(-1)).mean()
    assert torch.allclose(loss, manual, atol=1e-6)
    # 2. 因果性：改掉位置 4 的 token，位置 0~3 的 logits 不变，位置 4 变了
    x2 = x.clone()
    x2[:, 4] = (x2[:, 4] + 1) % vocab_size
    logits2, _ = model(x2)
    assert torch.allclose(logits[:, :4], logits2[:, :4], atol=1e-6)
    assert not torch.allclose(logits[:, 4], logits2[:, 4])
    # 3. weight tying：同一个 Parameter 对象，parameters() 只数一次
    assert model.lm_head.weight is model.tok_emb.weight
    # 4. 参数量 = VC + L(12C^2 + 13C) + 2C
    num_params = sum(p.numel() for p in model.parameters())
    expected = vocab_size * d_model + num_layers * (12 * d_model ** 2 + 13 * d_model) + 2 * d_model
    assert num_params == expected
    print(f"params: {num_params}, GPT tests passed")


def overfit_one_batch(steps: int = 50) -> GPT:
    """在同一个 batch 上反复训练，返回训练好的模型。"""
    torch.manual_seed(42)
    offsets = torch.tensor([[0], [3], [5], [7]])         # (4, 1)：4 条序列从不同数字开始
    data = (torch.arange(17) + offsets) % 10             # (4, 17)：规律是「下一个数 = 当前数 + 1」
    x, y = data[:, :-1], data[:, 1:]                     # (4, 16)，y 是 x 右移一位
    model = GPT(vocab_size=10, d_model=32, num_heads=4, num_layers=2, max_len=32, dropout=0.0)   # 关掉正则
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            print(f"step {step + 1}, train loss: {loss.item():.4f}")
    assert loss.item() < 0.01
    return model


if __name__ == "__main__":
    test_gpt()                                           # params: 25792, GPT tests passed
    model = overfit_one_batch()
    prompt = torch.tensor([[2, 3, 4], [8, 9, 0]])        # (2, 3)：训练数据里没有这样开头的序列
    print(generate(model, prompt, max_new_tokens=8).tolist())
    # step 1, train loss: 2.3088
    # step 26, train loss: 0.3366
    # step 50, train loss: 0.0068
    # [[2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2], [8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8]]
```

参数量这样数：每个 Block 里 attention 是 4 个 `(C, C)` 线性层加偏置，共 $4C^2+4C$；FFN 是 `C → 4C → C`，共 $8C^2+5C$；两个 LayerNorm 各 $2C$。每层合计 $12C^2+13C$，再加 embedding 的 $VC$（tying 后只算一次）和 `ln_f` 的 $2C$。拿 GPT-2 small（V = 50257，C = 768，12 层，再加 1024 × 768 的可学习位置向量）套这个公式，正好是 124,439,808，也就是常说的 124M。

过拟合检查的数字在本机 torch 1.13、2.4、2.8 上相同。第一步的 loss 2.3088 接近 $\ln 10\approx2.3026$，说明模型一开始在 10 个 token 上近似均匀地猜。这要靠 embedding 的 std 0.02 初始化：tying 之后 embedding 同时是输出层，换成 `nn.Embedding` 默认的 N(0, 1) 初始化，初始 logits 的标准差在 8 以上，初始 loss 超过 20。50 步后 loss 降到 0.0068，两个训练里没见过的开头也都接上了「加 1」的规律。

### 延伸：Encoder-Decoder

翻译这类 seq2seq 任务用原论文的 Encoder-Decoder 结构。Encoder 层就是第 4 步的 Block 去掉因果 mask（只留源端 padding mask）。Decoder 层多一个 **cross-attention** 子层，各子层的分工：

| 子层 | Q 来自 | K、V 来自 | mask |
| --- | --- | --- | --- |
| masked self-attention | decoder | decoder | 因果 mask（+ 目标端 padding） |
| cross-attention | decoder | encoder 输出 `memory` | 源端 padding mask |
| FFN | 无 | 无 | 无 |

cross-attention 不加因果 mask：源句在翻译开始前就完整给出，decoder 的每个位置都可以看整句源句。下面的代码用第 2 步的 `MultiHeadAttention` 和第 4 步的 `PositionwiseFeedForward`。

```python
from typing import Optional

import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    """Encoder-Decoder 里的 Decoder 层（Pre-LN）：self-attn → cross-attn → FFN。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: decoder 输入 (B, T_tgt, C)；memory: encoder 输出 (B, T_src, C)
            tgt_mask: (T_tgt, T_tgt) 因果 mask；src_mask: (B, 1, 1, T_src) 源端 padding mask，1 = 真实 token
        Returns:
            (B, T_tgt, C)
        """
        # 1. masked self-attention：Q = K = V = decoder
        h = self.ln1(x)
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask))
        # 2. cross-attention：Q 来自 decoder，K、V 来自 memory；注意力矩阵 (B, H, T_tgt, T_src)
        h = self.ln2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, src_mask))
        # 3. FFN
        x = x + self.dropout(self.ffn(self.ln3(x)))
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    layer = DecoderLayer(d_model=16, num_heads=4, d_ff=64)
    x, memory = torch.randn(2, 4, 16), torch.randn(2, 6, 16)          # (B, T_tgt, C)、(B, T_src, C)
    src_mask = torch.tensor([[1] * 6, [1, 1, 1, 1, 0, 0]])[:, None, None, :]   # (B, 1, 1, T_src)
    out = layer(x, memory, torch.tril(torch.ones(4, 4)), src_mask)    # (B, T_tgt, C)
    memory[1, 4:] = 100.0                                             # 第 2 句的 padding 位置随便改，输出不变
    assert torch.allclose(out, layer(x, memory, torch.tril(torch.ones(4, 4)), src_mask), atol=1e-6)
```

拷贝权重后，它和 `nn.TransformerDecoderLayer(d_model, nhead, d_ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)` 输出一致：我们的 `cross_attn` 对应它的 `multihead_attn`，`tgt_mask` 要转成 True = 屏蔽，`src_mask` 对应它的 `memory_key_padding_mask`（形状 `(B, T_src)`，True = padding）。

### 关键追问

- **为什么要因果 mask？** 训练时一次前向就算出 T 个位置的预测（teacher forcing），位置 t 的标签就是第 t+1 个输入。不遮住未来，模型直接「抄答案」，训练 loss 很低，生成时没有未来可抄，就完全不会了。mask 让训练和逐个生成时看到的信息一致，上面的因果性测试检查的就是这一点。
- **Encoder-only、Decoder-only、Encoder-Decoder 怎么选？** 区别在 mask 和有没有 cross-attention。BERT 是 encoder-only，双向注意力，用完形填空式的 MLM 预训练，适合分类、抽取这类理解任务；GPT 是 decoder-only，因果 mask，预测下一个 token，适合生成；原论文和 T5 是 Encoder-Decoder，适合翻译、摘要这种输入输出分开的任务。现在的大模型基本都是 decoder-only，把各种任务都写成「给前缀、续写」。
- **Pre-LN 和 Post-LN 有什么区别？** Post-LN（原论文）在初始化时靠近输出层的梯度很大，学习率一大就训崩，所以原论文要配学习率 warmup。Pre-LN 的残差主干是直通的，深层也好训，Xiong et al.（2020）发现它可以去掉 warmup。GPT-2 之后的主流 LLM 基本都是 Pre-Norm（LLaMA 把 LayerNorm 换成了 RMSNorm），代价是最后要补 `ln_f`。
- **dropout 放在哪几处？** 原论文写明两处：每个子层输出加回残差之前，以及 embedding 加位置编码之后。常见实现还有两处：注意力权重上（GPT-1 论文写明用了 attention dropout），FFN 两个 Linear 之间（PyTorch 的 `nn.TransformerEncoderLayer` 有）。上面的代码四处都放了。推理时 `model.eval()` 会全部关掉；很多大模型预训练直接把 dropout 设成 0。
- **复杂度和显存？** 每层的投影和 FFN 是 $O(TC^2)$，注意力打分 $QK^\top$ 和加权求和 $V$ 是 $O(T^2C)$，序列一长 $T^2$ 项就占主导。显存上注意力矩阵 `(B, H, T, T)` 是 $O(T^2)$，训练时还要存着给反向用。FlashAttention 分块计算、不存整个注意力矩阵，额外显存降到 $O(T)$，计算量还是 $O(T^2C)$。
- **为什么 weight tying？** 输入 embedding 把 token 映射成向量，`lm_head` 把向量映射回 token 的分数，两边用同一张表，语义一致、一起学。另外省参数：GPT-2 small 共享省掉 V × C = 50257 × 768 = 38,597,376 个参数，约占总量 124,439,808 的 31%。原论文也共享了 embedding 和 softmax 前的线性层。
- **原论文为什么把 embedding 乘 $\sqrt{d_{\text{model}}}$？** 论文只写了做法，没给理由。常见解释：共享权重时 embedding 要用小方差初始化（比如方差 $1/d_{\text{model}}$），输出层的 logits 才不会太大（见上面 std 0.02 的对比）；输入端再乘 $\sqrt{d_{\text{model}}}$，词向量每个元素的量级回到 1 左右，和取值在 -1 到 1 之间的正弦编码相当，位置信息不至于盖过词义。本节按 GPT-2 的做法没有乘，上面的过拟合检查说明小模型照样学到了词义。
- **mask 填 `-1e9` 还是 `-inf`？** 关键在某一行全被屏蔽的情况，例如左 padding 加因果 mask 时，开头的 PAD 位置能看到的全是 PAD。`-inf` 会让这一行 softmax 算出 NaN；到了下一层，真实 token 对这些位置的权重虽然是 0，但 0 乘 NaN 还是 NaN，整句都变成 NaN。`-1e9` 只会得到一行均匀权重，这些 PAD 位置的输出没有意义，但 loss 会忽略它们。fp16 下 `-1e9` 超出表示范围，`masked_fill` 会直接报错，要换成 `torch.finfo(scores.dtype).min`。

---

## 2.6 解码：Greedy、采样、KV Cache 与 Beam Search（PyTorch）

2.3 节和 2.4 节用 NumPy 讲了 beam search 和 top-k / top-p 采样的原理，每次只处理一条序列。面试里更常见的要求是：给你一个 PyTorch 语言模型，写出批量生成的循环。本节所有函数只依赖一个接口：`logits = model(input_ids)`，`input_ids` 是形状 `(B, T)` 的 LongTensor，`logits` 的形状是 `(B, T, V)`。2.5 节的 GPT 返回的是 `(logits, loss)`，套用下面的函数时把 `model(x)` 换成 `model(x)[0]`。

### 原理与直觉

自回归生成是一个循环，每一步做三件事：

1. 把当前序列喂进模型，拿到 `logits`，形状 `(B, T, V)`。
2. 只取最后一个位置 `logits[:, -1, :]`，形状 `(B, V)`。位置 $t$ 的输出预测的是第 $t+1$ 个 token，所以最后一个位置就是「下一个 token」的分数。
3. 按某种规则挑出下一个 token，拼到序列末尾，回到第 1 步。

各种解码方法只在第 3 步不同：

| 方法        | 第 3 步怎么挑                              | 输出 | 常见用途               |
| ----------- | ------------------------------------------ | ---- | ---------------------- |
| Greedy      | 取 argmax                                  | 确定 | 代码、数学、抽取、评测 |
| Beam Search | 保留累计 log-prob 最高的 $k$ 条路径        | 确定 | 翻译、摘要、语音识别（宽度常取 4 到 5） |
| 采样        | 温度缩放、top-k / top-p 截断后按概率随机抽 | 随机 | 对话、创意写作         |

一次生成一个 batch 时，每行生成 EOS 的时间不同。用一个形状 `(B,)` 的布尔向量 `finished` 记录哪些行已经结束，结束的行之后只补 PAD，全部结束就提前退出。另外，第 2 步只用最后一个位置，第 1 步却把整个前缀重算了一遍，这部分浪费由后面的 KV Cache 消除。

### 准备：一个能跑的小模型

为了让本节自己就能运行，先定义一个很小的语言模型 `TinyLM`：词嵌入 + 位置嵌入 + 一层因果多头自注意力（写法同 2.5 节的 MultiHeadAttention）+ 输出头。参数随机初始化、没有训练，输出没有语义，但足够检查解码代码写得对不对。词表只有 8 个 token，约定 0 是 PAD、1 是 BOS、2 是 EOS。

```python
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2    # 特殊 token，普通 token 是 3..7


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(CausalSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, H, d_k) -> (B, H, T, d_k)"""
        return x.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C)"""
        batch_size, seq_len, d_model = x.shape
        q = self.split_heads(self.w_q(x))                                     # (B, H, T, d_k)
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)   # (B, H, T, T)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))     # 1 = 可见
        scores = scores.masked_fill(mask == 0, -1e9)
        out = torch.matmul(F.softmax(scores, dim=-1), v)                      # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(out)                                                  # (B, T, C)


class TinyLM(nn.Module):
    """接口：input_ids (B, T) -> logits (B, T, V)"""

    def __init__(self, vocab_size: int = 8, d_model: int = 16, num_heads: int = 2, max_len: int = 32):
        super(TinyLM, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(input_ids.size(1), device=input_ids.device)  # (T,)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)                 # (B, T, C)，位置嵌入广播到每行
        x = x + self.attn(x)                                            # 残差
        return self.lm_head(self.ln_f(x))                               # (B, T, V)


torch.manual_seed(6)
model = TinyLM()
prompts = torch.tensor([[BOS_ID, 3], [BOS_ID, 4], [BOS_ID, 5]])   # (B=3, T=2)
print(model(prompts).shape)   # torch.Size([3, 2, 8])
```

### 先写核心版：批量 Greedy

```python
@torch.no_grad()        # 等价于把整个函数体放进 with torch.no_grad():
def greedy_generate(model: nn.Module, input_ids: torch.Tensor, max_new_tokens: int,
                    eos_id: int = EOS_ID, pad_id: int = PAD_ID) -> torch.Tensor:
    """
    Args:
        input_ids: (B, T)，每行一个 prompt，长度相同
    Returns:
        (B, T + n)，n <= max_new_tokens；每行 EOS 之后全是 pad_id
    """
    model.eval()                                                   # 关掉 dropout
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)  # (B,)
    for _ in range(max_new_tokens):
        logits = model(input_ids)                                  # 1. (B, T, V)，整个前缀重算
        next_token = logits[:, -1, :].argmax(dim=-1)               # 2. 只看最后一个位置，(B,)
        next_token = next_token.masked_fill(finished, pad_id)      # 3. 已结束的行只补 PAD
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)   # (B, T + 1)
        finished = finished | (next_token == eos_id)               # 4. 更新结束标记
        if finished.all():                                         # 5. 全部结束就提前退出
            break
    return input_ids


out = greedy_generate(model, prompts, max_new_tokens=8)
for row in out.tolist():
    print(row)
# [1, 3, 6, 5, 4, 5, 5, 7, 3, 2]
# [1, 4, 5, 5, 4, 5, 5, 5, 5, 2]
# [1, 5, 5, 5, 2, 0, 0, 0, 0, 0]
```

第 3 行在第 3 步就生成了 EOS（2），之后只补 PAD（0）；第 1、2 行到第 8 步才生成 EOS。第 2 行一直在重复 5，greedy 在真实模型上也常这样复读，后面用重复惩罚处理。两个常见 bug：

- `masked_fill(finished, pad_id)` 要放在更新 `finished` 之前，这一步刚生成的 EOS 才能保留下来。
- `finished` 要用 `|` 累积。直接写 `finished = next_token == eos_id`，下一步补上的 PAD 会把这一行改回「未结束」，生成又继续了。

### 面试版：温度、Top-K、Top-P 与重复惩罚

处理顺序是：重复惩罚 → 温度 → top-k → top-p → softmax → `torch.multinomial`，HuggingFace 的 `generate` 也是这个顺序。原理见 2.4 节。下面的重复惩罚、top-k、top-p 三个函数已经和 HuggingFace transformers 的 `RepetitionPenaltyLogitsProcessor`、`TopKLogitsWarper`、`TopPLogitsWarper` 在随机输入上对拍，输出逐元素相同。批量版本有几个 PyTorch 技巧：

- **过滤用 `masked_fill(..., float("-inf"))`**：被砍掉的 token 的 logit 设成 $-\infty$，softmax 后概率正好是 0，剩下的自动重新归一化，不用像 2.4 节那样手动除以和。Top-K 屏蔽比每行第 $k$ 大的值还小的 logit，和第 $k$ 大并列的都会留下。
- **Top-P 的 sorted-cumsum 技巧**：每行降序排序后算累计概率。如果一个 token 前面所有 token 的累计概率已经 $\ge p$，它就是多余的。第一个 token 前面的累计概率是 0，只要 $p>0$ 就永远保留，所以至少剩 1 个。最后用 `scatter` 把排序后的掩码放回原来的顺序。
- **可复现**：给 `torch.multinomial` 传 `generator=torch.Generator().manual_seed(0)`，结果只由这个种子决定，和全局随机状态无关。
**重复惩罚**来自 CTRL 论文：已经出现过的 token，logit 除以 $\theta$（论文建议 $\theta\approx1.2$）。HuggingFace 的实现补了一条：logit 为负时改成乘以 $\theta$。负数除以 $\theta>1$ 会变大，比如 $-2/1.5\approx-1.33$，惩罚就反了；乘以 $\theta$ 才能让它变小。OpenAI API 的 frequency / presence penalty 用减法：$z_i\leftarrow z_i-\alpha_{\text{freq}}\,c_i-\alpha_{\text{pres}}\,\mathbf{1}[c_i>0]$，$c_i$ 是 token $i$ 已出现的次数，前者出现越多扣得越多，后者出现过就扣一个固定值。

```python
def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    """logits: (B, V)；input_ids: (B, T)，已有的 token（prompt + 已生成）。返回 (B, V)"""
    score = logits.gather(1, input_ids)                              # (B, T)，出现过的 token 的 logit
    score = torch.where(score > 0, score / penalty, score * penalty)
    return logits.scatter(1, input_ids, score)    # 写回原位置；重复出现的 token 写的是同一个值，只罚一次


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """每行只保留最大的 k 个 logit，其余置为 -inf。logits: (B, V)"""
    k = min(k, logits.size(-1))
    kth_value = torch.topk(logits, k, dim=-1).values[:, -1:]         # (B, 1)，每行第 k 大的值
    return logits.masked_fill(logits < kth_value, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """保留累计概率刚好达到 p 的最小集合，至少保留 1 个。logits: (B, V)"""
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)   # (B, V)
    probs = F.softmax(sorted_logits, dim=-1)
    cum_before = probs.cumsum(dim=-1) - probs        # 排在它前面的 token 的累计概率
    sorted_remove = cum_before >= p                  # 前面已经凑够 p，它就多余；第 1 个的 cum_before = 0
    remove = sorted_remove.scatter(1, sorted_idx, sorted_remove)     # 排序后的顺序 -> 原顺序
    return logits.masked_fill(remove, float("-inf"))


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                      top_p: float = 1.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """logits: (B, V) -> next_token: (B,)"""
    if temperature == 0:                             # T -> 0 的极限就是 greedy，单独处理避免除零
        return logits.argmax(dim=-1)
    logits = logits / temperature                    # 1. 温度
    if top_k > 0:
        logits = top_k_filter(logits, top_k)         # 2. top-k 粗筛
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)         # 3. top-p 细筛
    probs = F.softmax(logits, dim=-1)                # 4. -inf 的概率正好是 0，剩下的自动重新归一化
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)   # (B,)


@torch.no_grad()
def generate(model: nn.Module, input_ids: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
             repetition_penalty: float = 1.0, eos_id: int = EOS_ID, pad_id: int = PAD_ID,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """和 greedy_generate 同一个循环，只把 argmax 换成 sample_next_token"""
    model.eval()
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :]                                    # (B, V)
        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        next_token = sample_next_token(logits, temperature, top_k, top_p, generator)   # (B,)
        next_token = next_token.masked_fill(finished, pad_id)
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
        finished = finished | (next_token == eos_id)
        if finished.all():
            break
    return input_ids
```

用上面的 `model` 和 `prompts` 试一下（`temperature=0`、`top_k=1` 等于 greedy，以及同一种子可复现，放在最后的自测里检查）：

```python
# 1. top-p 的小例子：概率 [0.15, 0.5, 0.05, 0.3]，p = 0.7 时只留 0.5 和 0.3
toy = torch.log(torch.tensor([[0.15, 0.5, 0.05, 0.3]]))
print(torch.isfinite(top_p_filter(toy, 0.7)).tolist())   # [[False, True, False, True]]

# 2. 重复惩罚：greedy 的第 2 行是 [1, 4, 5, 5, 4, 5, 5, 5, 5, 2]
print(generate(model, prompts, 8, temperature=0, repetition_penalty=1.5)[1].tolist())  # [1, 4, 5, 5, 2, 0, 0, 0, 0, 0]
```

加了惩罚之后，第 2 行生成两个 5 就输出了 EOS，不再复读。第二个 5 生成时已经被惩罚过，仍然被选中，说明惩罚只是压低分数，分数足够高的 token 照样会被选。

### KV Cache

不带缓存时，第 $t$ 步要把长度为 $t$ 的前缀整个重算一遍，生成 $T$ 个 token 一共要过约 $T^2/2$ 个 token 的投影和 FFN。其实旧位置的 K、V 算出来之后就不会再变：因果 mask 让每个位置只看自己和左边，后来的 token 影响不到它。所以把每层的 K、V 存下来，每步只为新 token 算 $q,k,v$，把新的 $k,v$ 拼到缓存末尾，再用这一个 $q$ 和缓存里所有的 K、V 做注意力，总共只过 $T$ 个 token。

代码只需改三处：`torch.cat` 拼接缓存；causal mask 加偏移（第 $i$ 个新 token 的绝对位置是 $T_{\text{past}}+i$）；位置嵌入从 $T_{\text{past}}$ 接着编号。下面的类继承前面的模型，参数名完全一样，所以能直接 `load_state_dict`，保证和不带缓存的版本用同一套权重。2.5 节的多层 GPT 改法相同，每层返回自己的 `(k, v)`，`past_kv` 变成长度为层数的列表。

```python
KVPair = Tuple[torch.Tensor, torch.Tensor]    # (k, v)，各 (B, H, T, d_k)


class KVCacheAttention(CausalSelfAttention):
    """参数和 CausalSelfAttention 完全相同，forward 多一个 past_kv"""

    def forward(self, x: torch.Tensor, past_kv: Optional[KVPair] = None) -> Tuple[torch.Tensor, KVPair]:
        """
        Args:
            x: (B, T_new, C)，只含新 token
            past_kv: None 或 (k, v)，各 (B, H, T_past, d_k)
        Returns:
            out: (B, T_new, C)
            present_kv: (k, v)，各 (B, H, T_past + T_new, d_k)
        """
        batch_size, t_new, d_model = x.shape
        q = self.split_heads(self.w_q(x))                  # (B, H, T_new, d_k)，只为新 token 算
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
        if past_kv is not None:                            # 1. 新的 K/V 接到缓存后面
            k = torch.cat([past_kv[0], k], dim=2)          # (B, H, T_total, d_k)
            v = torch.cat([past_kv[1], v], dim=2)
        t_total = k.size(2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)   # (B, H, T_new, T_total)
        # 2. 第 i 个新 token 的绝对位置是 T_past + i，能看到 <= 它的所有位置
        mask = torch.ones(t_new, t_total, device=x.device).tril(diagonal=t_total - t_new)
        scores = scores.masked_fill(mask == 0, -1e9)
        out = torch.matmul(F.softmax(scores, dim=-1), v)   # (B, H, T_new, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, t_new, d_model)
        return self.w_o(out), (k, v)


class KVCacheLM(TinyLM):
    """和 TinyLM 同样的参数；forward(input_ids, past_kv) -> (logits, present_kv)"""

    def __init__(self, vocab_size: int = 8, d_model: int = 16, num_heads: int = 2, max_len: int = 32):
        super(KVCacheLM, self).__init__(vocab_size, d_model, num_heads, max_len)
        self.attn = KVCacheAttention(d_model, num_heads)   # 换成带缓存的注意力，参数名不变

    def forward(self, input_ids: torch.Tensor,
                past_kv: Optional[KVPair] = None) -> Tuple[torch.Tensor, KVPair]:
        t_past = 0 if past_kv is None else past_kv[0].size(2)
        pos = torch.arange(t_past, t_past + input_ids.size(1), device=input_ids.device)  # 3. 位置接着编号
        x = self.tok_emb(input_ids) + self.pos_emb(pos)    # (B, T_new, C)
        attn_out, present_kv = self.attn(x, past_kv)
        x = x + attn_out
        return self.lm_head(self.ln_f(x)), present_kv      # (B, T_new, V)


@torch.no_grad()
def greedy_generate_cached(model: nn.Module, input_ids: torch.Tensor, max_new_tokens: int,
                           eos_id: int = EOS_ID, pad_id: int = PAD_ID) -> torch.Tensor:
    """输入输出和 greedy_generate 相同，model 是 KVCacheLM"""
    model.eval()
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    step_input, past_kv = input_ids, None      # 第 1 步喂整个 prompt（prefill），之后只喂新 token（decode）
    for _ in range(max_new_tokens):
        logits, past_kv = model(step_input, past_kv)               # (B, T_step, V)
        next_token = logits[:, -1, :].argmax(dim=-1)               # (B,)
        next_token = next_token.masked_fill(finished, pad_id)
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
        finished = finished | (next_token == eos_id)
        if finished.all():
            break
        step_input = next_token[:, None]                           # (B, 1)
    return input_ids
```

测试分两层：先比 logits（一次算完 vs. 先 prefill 4 个 token 再逐个 decode），再比生成的 token。

```python
cached_model = KVCacheLM()
cached_model.load_state_dict(model.state_dict())     # 同一套权重

with torch.no_grad():
    seq = greedy_generate(model, prompts, 8)                     # (3, 10)
    full_logits = model(seq)                                     # 不用缓存，一次算完，(3, 10, V)
    logits, past_kv = cached_model(seq[:, :4])                   # prefill 前 4 个 token
    pieces = [logits]
    for t in range(4, seq.size(1)):                              # 再一个一个 decode
        logits, past_kv = cached_model(seq[:, t:t + 1], past_kv)
        pieces.append(logits)
    cached_logits = torch.cat(pieces, dim=1)                     # (3, 10, V)

print(past_kv[0].shape)                                          # torch.Size([3, 2, 10, 8])，即 (B, H, T, d_k)
print(sum(t.numel() for t in past_kv))                           # 960，缓存里一共存了多少个数，见下面的显存公式
print(torch.allclose(full_logits, cached_logits, atol=1e-6))     # True
assert torch.equal(greedy_generate_cached(cached_model, prompts, 8), greedy_generate(model, prompts, 8))
```

**显存账**：每条序列的 KV Cache 大小是

$$
\text{KV Cache}=2\times n_{\text{layers}}\times n_{\text{heads}}\times d_{\text{head}}\times T\times\text{bytes}
$$

2 是 K 和 V 各一份；GQA 模型的 $n_{\text{heads}}$ 要换成 KV 头数；bytes 是每个数的字节数（fp16 是 2）；batch 里有多少条序列就再乘多少。上面的 TinyLM 是 1 层、2 个头、$d_{\text{head}}=8$，10 个 token、3 条序列正好是 960 个数，和 `numel` 数出来的一致。代入 Llama-3-8B（32 层、8 个 KV 头、$d_{\text{head}}=128$、fp16），每个 token 是 $2\times32\times8\times128\times2=131072$ 字节，即 128 KiB；8192 个 token 的上下文就是 $2^{30}$ 字节（1 GiB），而这只是一条序列。

### Beam Search（单条 prompt）

思路和长度惩罚见 2.3 节。这里换成真实模型，每条 beam 用一个字典 `{"tokens": [...], "score": 0.0}` 表示，`score` 是生成部分的累计 log-prob，排名用 $\text{score}/L^{\alpha}$（$L$ 是生成部分的长度，$\alpha=0$ 时就是原始分数）。和 2.3 节相比有两点不同：

1. 每条 beam 单独前向一次，用 `F.log_softmax` 拿到 log 概率，再用 `torch.topk` 取前 `beam_width` 个扩展。
2. 已经以 EOS 结尾的 beam 不单独移出，原样留在候选池里，用自己的分数继续和别的 beam 竞争；留下的 beam 全部以 EOS 结尾时停止。

```python
@torch.no_grad()
def beam_search(model: nn.Module, prompt: List[int], beam_width: int = 3,
                max_new_tokens: int = 10, eos_id: int = EOS_ID, alpha: float = 0.0) -> Dict:
    """
    Args:
        prompt: token 列表，不以 EOS 结尾；alpha: 长度惩罚，排名用 score / L ** alpha
    Returns:
        {"tokens": [...], "score": float}，tokens 含 prompt，L 是生成部分的长度
    """
    model.eval()
    prompt_len = len(prompt)

    def is_done(beam: Dict) -> bool:
        return len(beam["tokens"]) > prompt_len and beam["tokens"][-1] == eos_id

    def rank(beam: Dict) -> float:
        length = max(len(beam["tokens"]) - prompt_len, 1)
        return beam["score"] / length ** alpha

    beams = [{"tokens": list(prompt), "score": 0.0}]
    for _ in range(max_new_tokens):
        candidates = []
        for beam in beams:
            if is_done(beam):
                candidates.append(beam)                  # 1. 已结束：原样带到下一轮，继续参与排名
                continue
            logits = model(torch.tensor([beam["tokens"]]))[0, -1, :]          # (1, T, V) -> (V,)
            log_probs = F.log_softmax(logits, dim=-1)                          # (V,)
            top_lp, top_ids = torch.topk(log_probs, min(beam_width, log_probs.numel()))  # 2. 每条只展开前 k 个
            for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                candidates.append({"tokens": beam["tokens"] + [tok], "score": beam["score"] + lp})
        beams = sorted(candidates, key=rank, reverse=True)[:beam_width]        # 3. 全局剪枝
        if all(is_done(b) for b in beams):               # 4. 全部结束就停
            break
    return beams[0]                                      # 已按 rank 降序排好
```

怎么证明写对了？第一，`beam_width=1` 时每步只留 1 条，结果必须等于 greedy。第二，暴力枚举所有续写，每条用一次完整前向打分（和 beam search 逐步累加的算法完全独立）。词表 8、最多生成 3 个 token 时，前两步的候选最多 8 条和 64 条，`beam_width=64` 一条都不会剪掉，最后一步再取最大，结果必须和暴力枚举相同。

```python
import itertools


@torch.no_grad()
def brute_force_best(model: nn.Module, prompt: List[int], vocab_size: int,
                     max_new_tokens: int, eos_id: int = EOS_ID, alpha: float = 0.0) -> Dict:
    """枚举所有续写，每条用一次完整前向打分，返回 score / L ** alpha 最高的一条"""
    best, best_rank = {"tokens": [], "score": 0.0}, float("-inf")
    for cont in itertools.product(range(vocab_size), repeat=max_new_tokens):
        cont = list(cont)
        if eos_id in cont:
            cont = cont[:cont.index(eos_id) + 1]                 # EOS 之后的部分不算
        tokens = list(prompt) + cont
        log_probs = F.log_softmax(model(torch.tensor([tokens]))[0], dim=-1)   # (T, V)
        pos = torch.arange(len(prompt) - 1, len(tokens) - 1)    # 位置 t 的输出预测第 t + 1 个 token
        score = log_probs[pos, torch.tensor(cont)].sum().item()
        if score / len(cont) ** alpha > best_rank:
            best, best_rank = {"tokens": tokens, "score": score}, score / len(cont) ** alpha
    return best


prompt = [BOS_ID, 3]
for width in [1, 3, 64]:
    result = beam_search(model, prompt, beam_width=width, max_new_tokens=3)
    print(width, result["tokens"], round(result["score"], 4))
best = brute_force_best(model, prompt, vocab_size=8, max_new_tokens=3)
print("brute force", best["tokens"], round(best["score"], 4))
print(beam_search(model, prompt, beam_width=64, max_new_tokens=3, alpha=1.0)["tokens"])
# 1 [1, 3, 6, 5, 4] -4.62
# 3 [1, 3, 2] -1.9475
# 64 [1, 3, 2] -1.9475
# brute force [1, 3, 2] -1.9475
# [1, 3, 4, 5, 4]
```

宽度 1 的结果和上面 greedy 第 1 行的前 3 个新 token 一样，但错过了最优解；宽度 3 和 64 都找到了暴力枚举给出的 `[1, 3, 2]`：直接生成 EOS，一步就结束。累计 log-prob 每多一个 token 就多加一个负数，所以 $\alpha=0$ 时偏爱短句；加上 $\alpha=1$ 的长度惩罚后，换成了 3 个 token 的 `[1, 3, 4, 5, 4]`。

### 自测

换一个随机种子的模型和随机 prompt，把上面的性质都用 `assert` 检查一遍：

```python
def test_decoding():
    torch.manual_seed(42)
    lm, lm_cached = TinyLM(), KVCacheLM()
    lm_cached.load_state_dict(lm.state_dict())
    batch = torch.cat([torch.full((4, 1), BOS_ID), torch.randint(3, 8, (4, 2))], dim=1)  # (4, 3)

    # 1. greedy：EOS 之后全是 PAD；KV Cache 版本的输出完全一致
    out = greedy_generate(lm, batch, 6)
    for row in out[:, 3:].tolist():
        assert EOS_ID not in row or set(row[row.index(EOS_ID) + 1:]) <= {PAD_ID}
    assert torch.equal(greedy_generate_cached(lm_cached, batch, 6), out)

    # 2. T = 0、top_k = 1 都等于 greedy；同一个种子可复现；p 极小时 top-p 只留 argmax
    assert torch.equal(generate(lm, batch, 6, temperature=0), out)
    assert torch.equal(generate(lm, batch, 6, top_k=1), out)
    runs = [generate(lm, batch, 6, top_p=0.8, generator=torch.Generator().manual_seed(1)) for _ in range(2)]
    assert torch.equal(runs[0], runs[1])
    logits = torch.randn(5, 8)
    assert torch.equal(torch.isfinite(top_p_filter(logits, 1e-9)), F.one_hot(logits.argmax(-1), 8).bool())

    # 3. beam_width = 1 等于 greedy；足够宽的 beam 等于暴力枚举
    for prompt in batch[:2].tolist():
        greedy_row = greedy_generate(lm, torch.tensor([prompt]), 3)[0].tolist()
        assert beam_search(lm, prompt, beam_width=1, max_new_tokens=3)["tokens"] == greedy_row
        for alpha in [0.0, 1.0]:
            beam = beam_search(lm, prompt, beam_width=64, max_new_tokens=3, alpha=alpha)
            best = brute_force_best(lm, prompt, vocab_size=8, max_new_tokens=3, alpha=alpha)
            assert beam["tokens"] == best["tokens"] and abs(beam["score"] - best["score"]) < 1e-4
    print("all tests passed")


if __name__ == "__main__":
    test_decoding()   # all tests passed
```

### 关键追问

- **为什么 beam search 在开放式生成里输出很平淡？** 它找的是整句概率最高的序列，而概率最高的往往是最常见、最笼统的说法，还容易复读。Holtzman et al. (2020) 发现人写的文本并不落在概率最高的区域，所以开放式生成改用 top-p 采样。翻译这类任务的输出被源句约束，概率最高的那句通常就是对的，beam search 仍然好用。
- **一个 batch 里 prompt 长短不一怎么办？** decoder-only 模型用左 padding，例如 `[PAD, PAD, BOS, a]` 和 `[BOS, b, c, d]`，这样每行最后一个位置都是真实 token，`logits[:, -1, :]` 对每行都有意义，新 token 也整齐地拼在右边。如果右 padding，短句最后一列是 PAD，取到的是 PAD 位置的预测。同时要传 `attention_mask`（真实 token 为 1，PAD 为 0），让所有位置都看不到 PAD；位置编号也要从第一个真实 token 开始，HuggingFace 的很多模型用 `attention_mask.cumsum(-1) - 1` 算 position ids。本节的 TinyLM 为了简短没有实现 attention_mask，所以例子里的 prompt 等长。
- **Prefill 和 decode 有什么区别？为什么说 KV Cache 让 decode 变成访存密集？** Prefill 一次处理整段 prompt，几百上千个 token 一起做矩阵乘法，算力被打满，属于计算密集。Decode 每步每条序列只有 1 个新 token，矩阵乘法退化成矩阵乘向量，算量很小，却要把全部权重和整份 KV Cache 从显存读一遍。KV Cache 把重复计算换成了读显存，所以 decode 的速度由显存带宽决定，增大 batch 可以分摊读权重的开销。
- **为什么只缓存 K、V，不缓存 Q？** 新 token 的输出只需要它自己的 $q$ 和所有位置的 $k,v$。旧位置的 $q$ 只用来算旧位置的输出，那些输出已经算完了。另外，KV Cache 成立的前提是因果 mask：BERT 这类双向模型里旧位置也能看到新 token，每加一个 token 旧位置的表示都会变，缓存就失效了。
- **KV Cache 太大怎么办？** 从显存公式的每一项下手。减少 KV 头数：MQA 让所有 query 头共用 1 组 K、V，GQA 按组共用，Llama-3-8B 用 32 个 query 头配 8 个 KV 头，缓存是标准多头的 1/4。降低精度：把缓存量化成 int8 或 fp8，比 fp16 省一半。限制 $T$：Mistral 7B 的滑动窗口注意力只缓存最近 4096 个 token。另外，vLLM 的 PagedAttention 把缓存切成固定大小的块按需分配，减少按最大长度预留造成的显存浪费。
- **Beam search 加 KV Cache 要注意什么？** 每条 beam 有自己的缓存。剪枝后留下的 beam 可能来自同一个父 beam，要按父 beam 的下标用 `index_select` 重排缓存。工业实现还会把 `B` 个样本的 beam 摊平成 `(B * k, T)` 一起前向，再对每个样本的 `k * V` 个候选统一排序剪枝。
- **重复惩罚有什么坑？** 除以 / 乘以 $\theta$ 的规则依赖 logit 的正负号，logits 整体平移一下（softmax 结果不变），惩罚效果就变了。惩罚对 prompt 里的 token 也生效，代码、专有名词这类本来就需要重复的内容会被误伤。

---

## 3. KMeans

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

### 面试版实现（CodeSignal 风格）

接口仿 sklearn：`fit` 返回自己，`predict` 返回簇下标，结果存在 `centroids`、`labels_`、`inertia_` 里。比核心版多处理了四件事：初始化方式可选（默认 k-means++），空簇保留旧中心，收敛判断用 `tol` 代替浮点相等，以及跑 `n_init` 次、留 inertia 最小的一次。最后一条不是可有可无：k-means 只保证收敛到局部最优，单次 k-means++ 也可能把两团点合成一团（见下面的 Example）。

```python
from typing import Optional, Tuple, Union

import numpy as np


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4,
                 init: Union[str, np.ndarray] = "k-means++", n_init: int = 10,
                 random_state: Optional[int] = None):
        """
        Args:
            n_clusters: 簇数 k
            max_iter: 最多迭代几轮
            tol: 中心整体移动的距离不超过 tol 就停
            init: "k-means++"、"random"，或者直接给定初始中心 (k, d)
            n_init: 换不同的初始中心跑几次，留 inertia 最小的一次
            random_state: 随机种子，固定后结果可复现
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None    # (k, d)
        self.labels_: Optional[np.ndarray] = None      # (n,)
        self.inertia_: Optional[float] = None          # 每个点到所属中心的平方距离之和
        self.n_iter_ = 0

    @staticmethod
    def _sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """X: (n, d)，C: (k, d)。返回平方距离 (n, k)，广播写法见上面的拆解"""
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """X: (n, d)。返回初始中心 (k, d)"""
        n_samples = X.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters={self.n_clusters} > n_samples={n_samples}")
        if not isinstance(self.init, str):                   # 1. 直接给定初始中心
            return np.array(self.init, dtype=float)
        if self.init == "random":                            # 2. 随机挑 k 个不同的点
            return X[rng.choice(n_samples, size=self.n_clusters, replace=False)].copy()
        centroids = [X[rng.integers(n_samples)]]             # 3. k-means++：第一个中心随机挑
        for _ in range(1, self.n_clusters):
            d2 = self._sq_dist(X, np.array(centroids)).min(axis=1)    # (n,) 到最近已选中心
            probs = d2 / d2.sum() if d2.sum() > 0 else None           # 全部重合时退回均匀抽
            centroids.append(X[rng.choice(n_samples, p=probs)])
        return np.array(centroids)

    def _run_once(self, X: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
        """跑一次完整的 k-means。X: (n, d)。返回最终中心 (k, d) 和迭代轮数"""
        centroids = self._init_centroids(X, rng)                      # (k, d)
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            # 1. 分配：每个点归到最近的中心
            labels = self._sq_dist(X, centroids).argmin(axis=1)       # (n,)
            # 2. 更新：每个簇取均值；空簇保留旧中心
            new_centroids = centroids.copy()
            for j in range(self.n_clusters):
                members = X[labels == j]                              # (n_j, d)
                if len(members) > 0:
                    new_centroids[j] = members.mean(axis=0)
            # 3. 中心几乎不动就停
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift <= self.tol:
                break
        return centroids, n_iter

    def fit(self, X: np.ndarray) -> "KMeans":
        """X: (n, d)"""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n_runs = self.n_init if isinstance(self.init, str) else 1   # 给定初始中心时每次都一样
        best = None
        for _ in range(n_runs):
            centroids, n_iter = self._run_once(X, rng)
            d2 = self._sq_dist(X, centroids)                          # (n, k)
            inertia = float(d2.min(axis=1).sum())
            if best is None or inertia < best[0]:                     # 4. 留 inertia 最小的一次
                best = (inertia, centroids, d2, n_iter)
        # 5. labels_ 用最终中心算，和 centroids 对得上
        self.inertia_, self.centroids, d2, self.n_iter_ = best
        self.labels_ = d2.argmin(axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (m, d)。返回每个点最近中心的下标 (m,)"""
        if self.centroids is None:
            raise RuntimeError("call fit before predict")
        return self._sq_dist(np.asarray(X, dtype=float), self.centroids).argmin(axis=1)
```

### CodeSignal ML Core 版（纯 Python）

ML Core 测评把 k-Means 列在算法实现题里，而且不许用 NumPy（见基础篇第 0 节）。写法和基础篇 0.4 节一样：数据是 list of lists，辅助函数放顶层，入口是 `solution`。初始中心怎么选以题面为准，这里取前 k 个点；平局时取下标小的中心，和 `np.argmin` 一致。

```python
def squared_distance(p: list, q: list) -> float:
    return sum((a - b) ** 2 for a, b in zip(p, q))


def assign_clusters(data: list, centroids: list) -> list:
    """每个点最近中心的下标；平局取下标小的"""
    labels = []
    for point in data:
        dists = [squared_distance(point, c) for c in centroids]
        labels.append(dists.index(min(dists)))
    return labels


def update_centroids(data: list, labels: list, centroids: list) -> list:
    """每个簇取均值；空簇保留旧中心"""
    k, d = len(centroids), len(data[0])
    sums = [[0.0] * d for _ in range(k)]
    counts = [0] * k
    for point, label in zip(data, labels):
        counts[label] += 1
        for j in range(d):
            sums[label][j] += point[j]
    return [[s / counts[c] for s in sums[c]] if counts[c] else list(centroids[c]) for c in range(k)]


def solution(data: list, k: int, max_iter: int = 100) -> list:
    """data: n 个点，每个点是 d 个 float 的 list。返回每个点的簇下标（长度 n 的 list）"""
    if k > len(data):
        raise ValueError(f"k={k} > number of points={len(data)}")
    centroids = [list(p) for p in data[:k]]                    # 1. 初始化：前 k 个点
    labels = assign_clusters(data, centroids)
    for _ in range(max_iter):
        centroids = update_centroids(data, labels, centroids)  # 2. 更新中心
        new_labels = assign_clusters(data, centroids)          # 3. 重新分配
        if new_labels == labels:                               # 4. 分配不再变化就收敛了
            break
        labels = new_labels
    return labels
```

纯 Python 版用「分配不再变化」判断收敛，这是精确的：分配不变，均值就不变，再迭代也是原地不动。每轮复杂度和 NumPy 版一样是 $O(nkd)$，只是常数大。

### Example

```python
np.random.seed(42)
X = np.random.rand(100, 2)                       # (100, 2)
model = KMeans(n_clusters=3, random_state=42).fit(X)
print(model.centroids.shape, model.labels_.shape, round(model.inertia_, 4))   # (3, 2) (100,) 5.8103

# 三团分得很开的点：只跑一次 k-means++（n_init=1）时，seed=0 恰好陷进局部最优
rng = np.random.default_rng(0)
centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
blobs = np.vstack([c + rng.normal(scale=0.5, size=(50, 2)) for c in centers])   # (150, 2)
one = KMeans(n_clusters=3, n_init=1, random_state=0).fit(blobs)
ten = KMeans(n_clusters=3, n_init=10, random_state=0).fit(blobs)
print(round(one.inertia_, 2), sorted(np.bincount(one.labels_).tolist()))   # 719.32 [23, 27, 100]：两团被合成了一团
print(round(ten.inertia_, 2), sorted(np.bincount(ten.labels_).tolist()))   # 75.58 [50, 50, 50]：跑 10 次，留最好的

print(solution([[0.0, 0.0], [10.0, 10.0], [0.5, 0.0], [10.0, 9.5]], k=2))   # [0, 1, 0, 1]
```

### 自测

模仿 CodeSignal 的 hidden tests：分得很开的三团点必须被正确分开；`labels_`、`inertia_` 要和最终中心一致；同一个种子结果可复现；纯 Python 版和 NumPy 版用同样的初始中心时，结果逐个相同。

```python
import numpy as np


def test_kmeans() -> None:
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(50, 2)) for c in centers])   # (150, 2)
    truth = np.repeat(np.arange(3), 50)                                          # (150,)

    model = KMeans(n_clusters=3, random_state=0).fit(X)
    # 1. 每个真实簇整体落在同一个预测簇里，三个预测簇各不相同（编号可以不同）
    assert all(len(set(model.labels_[truth == c])) == 1 for c in range(3))
    assert len(set(model.labels_)) == 3
    # 2. labels_、inertia_、predict 都和最终中心一致
    d2 = ((X[:, None, :] - model.centroids[None, :, :]) ** 2).sum(axis=2)        # (150, 3)
    assert np.array_equal(model.labels_, d2.argmin(axis=1))
    assert np.isclose(model.inertia_, d2.min(axis=1).sum())
    assert np.array_equal(model.predict(X), model.labels_)
    # 3. 固定 random_state 可复现
    assert np.allclose(KMeans(n_clusters=3, random_state=0).fit(X).centroids, model.centroids)
    # 4. 纯 Python 版 == NumPy 版（同样以前 k 个点为初始中心，tol=0 表示中心完全不动才停）
    Xs = X[rng.permutation(len(X))]                                              # 打乱，前 3 个点来自不同的团
    ref = KMeans(n_clusters=3, init=Xs[:3], tol=0.0).fit(Xs)
    assert solution(Xs.tolist(), 3) == ref.labels_.tolist()
    # 5. 边界：k 比样本多、没 fit 就 predict，都要报错
    for bad in (lambda: KMeans(n_clusters=200).fit(X), lambda: KMeans(n_clusters=3).predict(X),
                lambda: solution([[0.0], [1.0]], 3)):
        try:
            bad()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError("should raise")
    print("all tests passed")


if __name__ == "__main__":
    test_kmeans()
```

### 关键追问

- **为什么不用 `sqrt`？** 最近 centroid 的 argmin 不受平方根影响，平方距离更省。
- **收敛到全局最优吗？** 不保证，只保证目标函数单调不增并收敛到局部最优或稳定点。
- **空簇怎么办？** 保留旧 centroid、随机重置、或重置到当前误差最大的点。
- **复杂度？** 每轮 $O(nkd)$，其中 $n$ 是样本数，$k$ 是簇数，$d$ 是维度。
- **实际优化？** k-means++ 初始化、多次随机重启、标准化特征。
- **CodeSignal 上不许用 NumPy 怎么写？** 用上面的纯 Python 版：辅助函数放顶层，入口是 `solution`。初始化方式、平局规则、最大轮数都以题面为准；题面规定了初始中心（比如前 k 个点）却没照做，是最常见的丢分原因，因为 hidden tests 比对的是确定的输出。

---

## 4. Logistic Regression

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

## 5. Multiple Linear Regression

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

## 5.1 多项式回归

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

## 6. Softmax

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

## 6.1 PyTorch 实现 Loss 函数

第 6 节用 NumPy 写了 softmax 和交叉熵，第 11 节手推了它的梯度 $(p-\text{onehot})/N$。面试里更常见的问法是「用 PyTorch 实现这个 loss」：给一个公式（focal loss、对比损失、蒸馏 loss），让你写出来，再说明怎么证明它是对的。本节把高频的几个 loss 各写一遍，每个都和 PyTorch 内置实现（没有内置的就和暴力循环）对拍 loss 值和梯度。用到的 `gather`、`masked_fill`、autograd 语法见基础篇 0.3 节。

### 原理与直觉

**只写 forward，backward 交给 autograd。** 只要 loss 全程由 torch 运算拼出来（`log_softmax`、`gather`、`softplus`、`where` 等），`loss.backward()` 会自动算好梯度。所以 PyTorch 版比第 11 节短得多：不用缓存中间量，也不用推导梯度。要小心的是数值稳定，以及别把计算图弄断：中途用 `.item()`、`.numpy()` 转成数字再参与计算，那一部分就收不到梯度。

**两种写法。**

- **函数式**：`def focal_loss(logits, target, ..., reduction="mean")`，和 `F.cross_entropy` 一个风格。`reduction` 三选一：`"none"` 返回逐样本 loss，形状 `(N,)`；`"sum"` 求和；`"mean"` 求平均。先算出逐样本 loss，最后一步再 reduce，三种 reduction 共用一份代码。
- **`nn.Module` 子类**：超参数（α、γ、温度）放进 `__init__`，计算放进 `forward`，用法和 `nn.CrossEntropyLoss()` 一样是 `criterion(logits, target)`。写法和 CodeSignal Learn 课程里的模块一致（`super(ClassName, self).__init__()`）。Loss 一般没有可学习参数，Module 只是在函数外面包一层。

**数值稳定的三条规则。**

1. **输入一律用 logits。** 多分类用 `F.log_softmax`，或者 `logits - torch.logsumexp(logits, dim=-1, keepdim=True)`；二分类用 `F.softplus` 或 `F.logsigmoid`。第 4 节 NumPy 版的稳定式搬进 autograd 有个坑，见下面 BCE 一节。
2. **不要先 softmax 再 log。** 概率下溢成 0 时 `log` 得到 `-inf`，反向全是 `nan`（下面有例子）。`logsumexp` 内部先减了最大值，和第 6 节的技巧一样。
3. **要忽略的位置先换成合法下标再算，最后清零。** 拿 `ignore_index=-100` 直接做花式索引，类别数 C < 100 时报越界；C ≥ 100 时（比如语言模型的词表）不报错，悄悄读到倒数第 100 列（`gather` 则会直接报错）。

### 先写核心版

面试官说「用 PyTorch 写交叉熵」，先写这 10 行，再问要不要支持 `ignore_index` 和 label smoothing。

```python
import torch
import torch.nn.functional as F


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """logits: (N, C) 未经 softmax 的分数；target: (N,) 整数类别。返回标量。"""
    # 1. log-softmax = z - logsumexp(z)，(N, C)；logsumexp 内部先减最大值，不会溢出
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # 2. 每行取真实类别那一列，(N,)；和第 6 节 NumPy 版的花式索引一样
    nll = -log_probs[torch.arange(logits.size(0)), target]
    # 3. 对 batch 求平均，得到标量
    return nll.mean()


torch.manual_seed(0)
logits = torch.randn(4, 5)              # (N=4, C=5)
target = torch.tensor([1, 0, 4, 2])     # (N,)
print(f"{cross_entropy(logits, target).item():.4f}")     # 2.4194
print(f"{F.cross_entropy(logits, target).item():.4f}")   # 2.4194

# 反例：先 softmax 再 log。torch.autograd.grad(loss, x) 直接返回梯度，不写进 x.grad
x = torch.tensor([[0.0, -200.0]], requires_grad=True)   # (1, 2)，真实类别是 1
bad = -torch.log(torch.softmax(x, dim=1))[0, 1]         # e^{-200} 下溢成 0，log(0) = -inf
print(bad.item(), torch.autograd.grad(bad, x)[0])       # inf tensor([[nan, nan]])
good = F.cross_entropy(x, torch.tensor([1]))            # 内部走 log_softmax
print(good.item(), torch.autograd.grad(good, x)[0])     # 200.0 tensor([[ 1., -1.]])：p - onehot，第 11 节
```

### 怎么对拍：loss 和梯度一起比

自己写的 loss 要证明两件事：forward 的值对，梯度也对。把同一份输入复制两份，分别过自己的实现和内置实现，各自 `backward()`，再比较 loss 和 `.grad`：

```python
from typing import Callable

import torch


def compare_with_reference(my_fn: Callable, ref_fn: Callable, x: torch.Tensor,
                           atol: float = 1e-6) -> bool:
    """my_fn / ref_fn: 输入 x、返回标量 loss。loss 值和 x.grad 都一致才返回 True"""
    x1 = x.detach().clone().requires_grad_(True)    # 两份独立的叶子张量，梯度互不干扰
    x2 = x.detach().clone().requires_grad_(True)
    loss1, loss2 = my_fn(x1), ref_fn(x2)
    loss1.backward()
    loss2.backward()
    return torch.allclose(loss1, loss2, atol=atol) and torch.allclose(x1.grad, x2.grad, atol=atol)


print(compare_with_reference(lambda z: cross_entropy(z, target),
                             lambda z: F.cross_entropy(z, target), logits))   # True

# gradcheck：autograd 的梯度 vs 中心差分，输入必须是 float64
x64 = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)   # (N, C)
print(torch.autograd.gradcheck(lambda z: cross_entropy(z, target), (x64,)))   # True
```

没有内置实现可比时，用 `torch.autograd.gradcheck`：它把 autograd 的梯度和中心差分 $\big(f(x+h)-f(x-h)\big)/2h$（第 11 节的数值梯度）比较。输入要用 float64，float32 的差分误差太大，基本都会报 `GradcheckError`，PyTorch 也会弹警告提醒。gradcheck 主要抓三类问题：计算图被 `.item()`、`.detach()` 弄断（数值梯度有、解析梯度没有）；用 `torch.autograd.Function` 手写的 backward；`max`、`abs` 这类函数在拐点上取的梯度（下面 BCE 一节的 x = 0 就是一例）。forward 公式本身写错，要靠和内置实现对拍才能发现。

### 面试版实现

#### 1. Cross-Entropy：ignore_index 与 label smoothing

**Label smoothing**：把 one-hot 目标换成软目标 $q$，真实类别拿大头，其余类别分一点。PyTorch 把 ε 均匀分给全部 $C$ 个类（包括真实类别），代入交叉熵 $-\sum_k q_k\log p_k$ 拆成两项，第一项是普通交叉熵，第二项是对均匀分布的交叉熵，也就是代码里的 `-log_probs.mean(dim=1)`：

$$
q_k=(1-\varepsilon)\,\mathbf{1}[k=y]+\frac{\varepsilon}{C},\qquad
\mathcal L_i=(1-\varepsilon)\,(-\log p_{y_i})+\varepsilon\cdot\frac{1}{C}\sum_{k=1}^{C}(-\log p_k)
$$

**ignore_index**：padding 位置的 target 设成 `ignore_index`，这些位置不计入 loss。`reduction="mean"` 时分母是**有效位置的个数** $\sum_i m_i$，其中 $m_i=\mathbf{1}[y_i\ne\text{ignore}]$。先写一个共用的 `reduce_loss`，后面每个 loss 都先算逐元素 loss，最后交给它。写完和 `F.cross_entropy` 对拍三种 reduction。

```python
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """loss: 逐元素 loss。'none' 原样返回，'sum' 求和，'mean' 求平均"""
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction}")


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100,
                       label_smoothing: float = 0.0, reduction: str = "mean") -> torch.Tensor:
    """
    Args:
        logits: (N, C) 未经 softmax 的分数
        target: (N,) 整数类别；等于 ignore_index 的位置不计入 loss
        label_smoothing: ε，均匀分给 C 个类
        reduction: 'mean' | 'sum' | 'none'
    Returns:
        'none' 时 (N,)，被忽略的位置为 0；否则标量（全部被忽略时 mean 是 nan，和内置一致）
    """
    log_probs = F.log_softmax(logits, dim=1)                         # 1. (N, C)
    valid = target != ignore_index                                    # 2. (N,) bool，True 表示要算
    safe_target = target.masked_fill(~valid, 0)                       # 3. 忽略位置换成合法下标 0
    nll = -log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)   # 4. (N,) 真实类别的 -log p，gather 见基础篇 0.3 节
    smooth = -log_probs.mean(dim=1)                                   # 5. (N,) 对均匀分布的交叉熵
    loss = (1 - label_smoothing) * nll + label_smoothing * smooth     # 6. (N,)
    loss = loss.masked_fill(~valid, 0.0)                              # 7. 忽略位置清零
    if reduction == "mean":
        return loss.sum() / valid.sum()                               # 分母 = 有效位置数
    return reduce_loss(loss, reduction)


torch.manual_seed(0)
logits = torch.randn(6, 5)                            # (N=6, C=5)
target = torch.tensor([1, -100, 3, -100, 0, 2])       # (N,)，两个位置被忽略
for reduction in ["mean", "sum", "none"]:           # 'none' 求和成标量后再比梯度
    ok = compare_with_reference(
        lambda z: cross_entropy_loss(z, target, label_smoothing=0.1, reduction=reduction).sum(),
        lambda z: F.cross_entropy(z, target, label_smoothing=0.1, reduction=reduction).sum(), logits)
    print(reduction, ok)                              # mean True / sum True / none True

per_sample = cross_entropy_loss(logits, target, label_smoothing=0.1, reduction="none")   # (N,)
print(f"{per_sample.sum().item() / 4:.4f} {per_sample.mean().item():.4f}")   # 1.8065 1.2043
all_pad = torch.full((6,), -100)                      # 整个 batch 都是 padding
print(cross_entropy_loss(logits, all_pad), F.cross_entropy(logits, all_pad))   # tensor(nan) tensor(nan)
```

`reduction="mean"` 的结果是 1.8065，即总和除以 4 个有效位置。拿 `reduction="none"` 的结果自己再 `.mean()`，被忽略的 0 也进了分母（除以 6），得到偏小的 1.2043。全部位置都被忽略时分母是 0，两边都返回 nan（梯度两边都是 0）；训练里可能出现整批都是 padding 的情况，就跳过这一步，或者用 `sum` 再除以 `max(有效数, 1)`。

**语言模型的 (B, T, V)。** 位置 $t$ 的 logits 预测第 $t+1$ 个 token，所以先错开一位，再展平成 `(B*(T-1), V)`（训练循环里的写法见 10.1 节）：

```python
torch.manual_seed(0)
B, T, V, pad_id = 2, 6, 10, 0
lm_logits = torch.randn(B, T, V)                    # (B, T, V) 模型输出
tokens = torch.randint(1, V, (B, T))                # (B, T) 真实 token，避开 pad_id
tokens[0, 4:] = pad_id                              # 第 0 条句子最后两个位置是 padding
flat_logits = lm_logits[:, :-1].reshape(-1, V)      # (B*(T-1), V)：位置 t 预测 t+1；切片后不连续，view 会报错
flat_labels = tokens[:, 1:].reshape(-1)             # (B*(T-1),) = (10,)，其中 2 个是 pad_id
print(compare_with_reference(
    lambda z: cross_entropy_loss(z, flat_labels, ignore_index=pad_id, label_smoothing=0.1),
    lambda z: F.cross_entropy(z, flat_labels, ignore_index=pad_id, label_smoothing=0.1),
    flat_logits))                                   # True
```

10 个预测位置里有 2 个的 label 是 padding，所以 `mean` 除以 8。内置版也接受 `(B, V, T-1)` 的 logits 配 `(B, T-1)` 的 target，即 `lm_logits[:, :-1].transpose(1, 2)`，类别必须在第 1 维。`(B, T, V)` 不能直接传：它会把第 1 维的 $T$ 当成类别数，形状对不上时报错；凑巧 $T=V$ 时不报错，结果却是错的。

#### 2. BCE with logits 与 pos_weight

二分类（或多标签）用 sigmoid。`pos_weight` 记作 $w_p$，给正样本那一项加权，正负样本比例悬殊时常用（例如负:正 = 9:1 就设 9）：

$$
\mathcal L=-\big[w_p\,y\log\sigma(x)+(1-y)\log(1-\sigma(x))\big]
=w_p\,y\operatorname{softplus}(-x)+(1-y)\operatorname{softplus}(x)
$$

这里用到 $-\log\sigma(x)=\operatorname{softplus}(-x)$ 和 $-\log(1-\sigma(x))=\operatorname{softplus}(x)$，其中 $\operatorname{softplus}(x)=\log(1+e^{x})$。`F.softplus` 内部用稳定写法，大的正数直接返回 $x$，不会溢出；`F.softplus(-x)` 也可以写成 `-F.logsigmoid(x)`。$w_p=1$ 时这个式子和第 4 节的 $\max(x,0)-xy+\log(1+e^{-|x|})$ 相等。

**为什么不直接搬第 4 节的 `max` / `abs` 写法？** 它的 forward 没问题，梯度在 x = 0 这一点会错。`max(x, 0)` 和 $|x|$ 在 0 处不可导，autograd 取的是 `clamp` 传 1、`abs` 传 0，本该抵消的两部分没有抵消，梯度变成 $1-y$，正确值是 $\sigma(0)-y=0.5-y$。

随机输入几乎不会正好落在 0，随机对拍发现不了，而输出层零初始化后第一步的 logits 就是全 0；在 x = 0 处跑 gradcheck 会报 `GradcheckError`。NumPy 版没有这个问题，因为第 4 节的梯度是按 $\sigma(x)-y$ 手写的。`F.softplus` 的 backward 直接按 sigmoid 算，没有这个坑；不让用它时，`torch.logaddexp(torch.zeros_like(x), x)` 也是稳定的 softplus，0 处梯度正确。下面对拍用多标签输入，其中两行放 ±100 的极端 logits，最后演示 x = 0 处的梯度：

```python
def bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                    pos_weight: Optional[torch.Tensor] = None,
                    reduction: str = "mean") -> torch.Tensor:
    """logits: (N,) 或 (N, C)；target: 同形状的 0/1（或 [0, 1] 软标签）；pos_weight: 标量张量或 (C,)"""
    w_p = 1.0 if pos_weight is None else pos_weight
    pos_term = w_p * target * F.softplus(-logits)       # 1. 正样本项：-w_p · y · log σ(x)
    neg_term = (1 - target) * F.softplus(logits)        # 2. 负样本项：-(1 - y) · log(1 - σ(x))
    return reduce_loss(pos_term + neg_term, reduction)


torch.manual_seed(0)
x = torch.randn(8, 3) * 5                           # (N=8, C=3) 多标签 logits
x[0], x[1] = 100.0, -100.0                          # 极端值：sigmoid 在 float32 里饱和
y = torch.randint(0, 2, (8, 3)).float()             # (N, C) 0/1 标签
pw = torch.tensor([1.0, 3.0, 9.0])                  # (C,) 每个类别一个正样本权重
print(compare_with_reference(lambda z: bce_with_logits(z, y, pw),
                             lambda z: F.binary_cross_entropy_with_logits(z, y, pos_weight=pw), x))   # True

x0 = torch.zeros(3, requires_grad=True)             # (N,) 全 0 的 logits
y0 = torch.tensor([1.0, 0.0, 1.0])
ported = (x0.clamp(min=0) - x0 * y0 + torch.log1p(torch.exp(-x0.abs()))).sum()   # 第 4 节的稳定式
print(torch.autograd.grad(ported, x0)[0])           # tensor([0., 1., 0.])：1 - y，错
print(torch.autograd.grad(bce_with_logits(x0, y0, reduction="sum"), x0)[0])   # tensor([-0.5000,  0.5000, -0.5000])
```

#### 3. Focal Loss（nn.Module 写法）

风控、CTR 预估这类任务里，绝大多数样本是负样本，模型很快就能把它们分对（$p_t$ 接近 1），但数量巨大，loss 加起来仍然主导梯度。Focal loss（Lin et al., 2017）给每个样本乘 $(1-p_t)^\gamma$，分得越对，权重越小：

$$
\mathrm{FL}=-\alpha_t\,(1-p_t)^{\gamma}\log p_t,\qquad
p_t=\begin{cases}p,&y=1\\1-p,&y=0\end{cases},\qquad
\alpha_t=\begin{cases}\alpha,&y=1\\1-\alpha,&y=0\end{cases}
$$

$-\log p_t$ 正好是逐元素的 BCE，所以直接复用上面的 `bce_with_logits`，再用 $p_t=e^{-\text{BCE}}$ 拿到 $p_t$，不必单独算 sigmoid。PyTorch 没有内置 focal loss，最基本的检查是：γ = 0 且不加 α 时，结果必须等于 BCE。

```python
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: Optional[float] = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha          # 正样本权重，None 表示不加
        self.gamma = gamma          # 聚焦参数，0 时退化成 BCE
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (N,) 未经 sigmoid 的分数；target: (N,) 0/1 标签。'none' 时返回 (N,)，否则标量"""
        ce = bce_with_logits(logits, target, reduction="none")    # 1. (N,) = -log p_t，稳定
        p_t = torch.exp(-ce)                                      # 2. (N,) 模型给真实类别的概率
        loss = (1 - p_t) ** self.gamma * ce                       # 3. 分得越对，权重越小
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)   # 4. (N,)
            loss = alpha_t * loss
        return reduce_loss(loss, self.reduction)


torch.manual_seed(0)
x = torch.randn(16) * 3                             # (N,)
y = torch.randint(0, 2, (16,)).float()              # (N,)
print(compare_with_reference(lambda z: BinaryFocalLoss(alpha=None, gamma=0.0)(z, y),
                             lambda z: F.binary_cross_entropy_with_logits(z, y), x))   # True

# 直觉：正样本的 p_t 取 0.9 / 0.5 / 0.1，focal loss 和 BCE 的比值就是 (1 - p_t)^2
x_demo = torch.logit(torch.tensor([0.9, 0.5, 0.1]))  # 反解 logits，使 sigmoid(x) = p_t
ratio = BinaryFocalLoss(alpha=None, reduction="none")(x_demo, torch.ones(3)) / \
    bce_with_logits(x_demo, torch.ones(3), reduction="none")
print(ratio)                                        # tensor([0.0100, 0.2500, 0.8100])
```

#### 4. InfoNCE / CLIP 对比损失

一个 batch 里有 $N$ 对（图，文）。第 $i$ 张图的正样本是第 $i$ 段文字，其余 $N-1$ 段都当负样本（in-batch negatives）。embedding 做 L2 归一化后两两点积，得到 $N\times N$ 的相似度矩阵，再除以温度 τ：

$$
s_{ij}=\frac{\hat u_i^\top\hat v_j}{\tau},\qquad
\mathcal L_{\text{img}}=\frac1N\sum_{i}-\log\frac{e^{s_{ii}}}{\sum_{j}e^{s_{ij}}},\qquad
\mathcal L=\frac12\big(\mathcal L_{\text{img}}+\mathcal L_{\text{txt}}\big)
$$

$\mathcal L_{\text{txt}}$ 按列做同样的事。每一行是一个 $N$ 分类问题，正确答案在对角线上，所以 label 就是 `arange(N)`，直接调交叉熵；按列算就是对转置后的矩阵再调一次。下面和照抄公式的单方向循环版对拍（float64，顺便比梯度），两个方向就是交换两个参数各算一次。

```python
def clip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """img_emb, txt_emb: (N, D)，第 i 行互为正样本对。返回两个方向的平均 loss（标量）"""
    img = F.normalize(img_emb, dim=-1)                       # 1. (N, D) L2 归一化，点积 = cosine
    txt = F.normalize(txt_emb, dim=-1)                       #    (N, D)
    sim = img @ txt.t() / temperature                        # 2. (N, N)，对角线是正样本对
    labels = torch.arange(sim.size(0), device=sim.device)    # 3. (N,) 第 i 行的答案是第 i 列
    loss_img = F.cross_entropy(sim, labels)                  # 4. 每张图在 N 段文字里找自己那段
    loss_txt = F.cross_entropy(sim.t(), labels)              # 5. 每段文字在 N 张图里找自己那张
    return (loss_img + loss_txt) / 2


def info_nce_loop(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """暴力版单方向 InfoNCE，只用来对拍。a, b: (N, D)，a[i] 的正样本是 b[i]"""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    n = a.size(0)
    total = 0.0
    for i in range(n):                                   # 样本 i：分母遍历所有 j
        row = torch.stack([a[i] @ b[j] / temperature for j in range(n)])   # (N,)
        total = total - torch.log(torch.exp(row[i]) / torch.exp(row).sum())
    return total / n


torch.manual_seed(0)
img_emb = torch.randn(5, 8, dtype=torch.float64)    # (N=5, D=8)
txt_emb = torch.randn(5, 8, dtype=torch.float64)
loop_both = lambda z: (info_nce_loop(z, txt_emb, 0.07) + info_nce_loop(txt_emb, z, 0.07)) / 2
print(compare_with_reference(lambda z: clip_loss(z, txt_emb, 0.07), loop_both, img_emb))   # True
# 循环版直接 exp 也不会溢出：cosine 在 [-1, 1]，除以 0.07 后最大约 14.3
```

#### 5. 知识蒸馏的 KL Loss

Hinton et al.（2015）让学生模仿老师的「软标签」：两边的 logits 都除以温度 $T$ 再 softmax，$T>1$ 让分布变平，把「这张 3 有点像 8」这类类别间的信息露出来。loss 是老师分布到学生分布的 KL 散度，再乘 $T^2$：

$$
\mathcal L_{\text{KD}}=T^2\cdot\frac1N\sum_{i}\sum_{k}p^{t}_{ik}\big(\log p^{t}_{ik}-\log p^{s}_{ik}\big),\qquad
p^{t}=\operatorname{softmax}(z^{t}/T),\quad p^{s}=\operatorname{softmax}(z^{s}/T)
$$

训练时再和硬标签的交叉熵加权：$\alpha\,\mathcal L_{\text{KD}}+(1-\alpha)\,\mathrm{CE}(z^{s},y)$。老师的 logits 要 `detach()`（或在 `torch.no_grad()` 下算），梯度只回传给学生。内置写法是 `F.kl_div(input, target)`：`input` 是**学生的 log 概率**，`target` 是**老师的概率**，算的是 $\mathrm{KL}(\text{target}\,\Vert\,\exp(\text{input}))$。参数顺序和直觉相反，最容易写反。`reduction` 要用 `"batchmean"`（除以 $N$）；`"mean"` 会再除以类别数 $C$，和 KL 的定义对不上，PyTorch 会给出警告。

```python
def distillation_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         temperature: float = 4.0) -> torch.Tensor:
    """student_logits, teacher_logits: (N, C)。返回 T^2 * KL(p_teacher || p_student)，按样本平均"""
    teacher_logits = teacher_logits.detach()                        # 老师不更新
    log_p_s = F.log_softmax(student_logits / temperature, dim=-1)   # 1. (N, C)
    log_p_t = F.log_softmax(teacher_logits / temperature, dim=-1)   # 2. (N, C) 老师也取 log，避免 0 * log 0
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)          # 3. (N,) 每个样本的 KL
    return kl.mean() * temperature ** 2                             # 4. 按样本平均，再乘 T^2


torch.manual_seed(0)
student = torch.randn(8, 10)                        # (N=8, C=10)
teacher = torch.randn(8, 10) * 3                    # (N, C) 老师更「自信」
T = 4.0
builtin_kd = lambda z: F.kl_div(F.log_softmax(z / T, dim=-1), F.softmax(teacher / T, dim=-1),
                                reduction="batchmean") * T * T
print(compare_with_reference(lambda z: distillation_kl_loss(z, teacher, T), builtin_kd, student))   # True

# 为什么乘 T^2：看学生 logits 的梯度范数
for temp in [1.0, 2.0, 4.0, 8.0]:
    s = student.clone().requires_grad_(True)
    distillation_kl_loss(s, teacher, temp).backward()
    g = s.grad.norm().item()
    print(f"T={temp:.0f}  without T^2: {g / temp ** 2:.4f}  with T^2: {g:.4f}")
# T=1  without T^2: 0.2745  with T^2: 0.2745
# T=2  without T^2: 0.0816  with T^2: 0.3263
# T=4  without T^2: 0.0200  with T^2: 0.3206
# T=8  without T^2: 0.0049  with T^2: 0.3142
```

不乘 $T^2$ 时，$T$ 从 2 起每翻一倍，梯度缩成约 1/4；乘上之后梯度范数基本不随 $T$ 变，和硬标签 CE 的相对权重 α 就不用跟着 $T$ 重新调。

#### 6. MSE 与 Huber

MSE 就是 `reduce_loss((pred - target) ** 2, reduction)`，主要考形状检查。Huber 在误差小时和 MSE 一样平滑，误差大时变成线性，离群点的梯度被限制在 ±δ 以内，不会主导训练：

$$
\ell_\delta(d)=\begin{cases}\tfrac12 d^2,&|d|<\delta\\ \delta\,\big(|d|-\tfrac12\delta\big),&|d|\ge\delta\end{cases},\qquad d=\hat y-y
$$

```python
def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0,
               reduction: str = "mean") -> torch.Tensor:
    """pred, target: 同形状，例如 (N,)；|d| < delta 用平方，否则用线性"""
    assert pred.shape == target.shape, f"shape mismatch: {pred.shape} vs {target.shape}"
    abs_diff = (pred - target).abs()                                     # (N,)
    loss = torch.where(abs_diff < delta, 0.5 * abs_diff ** 2,            # 小误差：二次
                       delta * (abs_diff - 0.5 * delta))                 # 大误差：线性
    return reduce_loss(loss, reduction)


torch.manual_seed(0)
pred, y_true = torch.randn(10) * 3, torch.randn(10)  # (N,)，一部分误差超过 delta
print(compare_with_reference(lambda z: huber_loss(z, y_true, delta=2.0),
                             lambda z: F.huber_loss(z, y_true, delta=2.0), pred))   # True
```

断言形状是为了防 `(N, 1)` 减 `(N,)` 广播成 `(N, N)` 的经典 bug（基础篇 0.3 节的常见坑 3），MSE 也要加。`F.smooth_l1_loss(pred, y, beta=δ)` 等于 Huber 除以 δ。

### 自测

模仿 CodeSignal 的 hidden tests：随机输入、随机忽略位置，加上 logits 为 0、全部忽略两个边界，和内置实现对拍（InfoNCE、KD、Huber 上面已对拍过）。用上面定义的函数：

```python
def run_tests() -> None:
    torch.manual_seed(42)
    for _ in range(5):
        # 1. CE：随机 label smoothing，随机忽略约 30% 的位置，两种 reduction
        logits, target = torch.randn(12, 7), torch.randint(0, 7, (12,))     # (N, C), (N,)
        target[torch.rand(12) < 0.3] = -100
        target[0] = 3                                                       # 保证至少一个有效位置
        eps = float(torch.rand(1)) * 0.3
        for red in ["mean", "sum"]:
            assert compare_with_reference(
                lambda z: cross_entropy_loss(z, target, label_smoothing=eps, reduction=red),
                lambda z: F.cross_entropy(z, target, label_smoothing=eps, reduction=red), logits)
        # 2. BCE：(N, C) 多标签 + 随机 pos_weight；第 0 行 logits 全为 0，检查 x = 0 处的梯度
        x, y, pw = torch.randn(6, 4) * 4, torch.randint(0, 2, (6, 4)).float(), torch.rand(4) * 5
        x[0] = 0.0
        assert compare_with_reference(lambda z: bce_with_logits(z, y, pw),
                                      lambda z: F.binary_cross_entropy_with_logits(z, y, pos_weight=pw), x)
        # 3. Focal：γ = 0 且不加 α 时等于 BCE
        assert compare_with_reference(lambda z: BinaryFocalLoss(None, 0.0)(z, y),
                                      lambda z: F.binary_cross_entropy_with_logits(z, y), x)
    # 4. 边界：全部被忽略时 mean 是 nan，和内置一致；reduction='none' 返回 (N,)
    all_pad = torch.full((3,), -100)
    assert torch.isnan(cross_entropy_loss(torch.randn(3, 4), all_pad))
    assert torch.isnan(F.cross_entropy(torch.randn(3, 4), all_pad))
    assert cross_entropy_loss(torch.randn(3, 4), torch.tensor([0, -100, 2]), reduction="none").shape == (3,)
    print("all tests passed")


if __name__ == "__main__":
    run_tests()
```

### 关键追问

- **为什么 loss 吃 logits，不吃概率？** 数值稳定：logsumexp 和 softplus 的写法不会出现 $\log0$，先 softmax 再 log 会得到 `inf`，梯度是 `nan`。模型最后一层接了 `nn.Softmax` 再喂给 `F.cross_entropy`，等于做了两次 softmax（基础篇 0.3 节、本页 10.1 节）：不报错，但 loss 降不到 0（10 个类时最低约 1.46），梯度也被压扁。
- **有 ignore_index 时 mean 除以什么？** 除以有效位置数 $\sum_i m_i$。语言模型里不同 batch 的有效 token 数不同；梯度累积（10.1 节）时如果每个 micro-batch 各自取 mean 再平均，有效 token 数不一样时结果和一次算大 batch 不相等。严格的写法是各 micro-batch 用 `reduction="sum"`，最后除以总有效 token 数。
- **类别不均衡怎么处理？** 多分类给 `F.cross_entropy` 传 `weight=w`（形状 `(C,)`），`mean` 的分母随之变成有效位置的权重和 $\sum_i w_{y_i}$；二分类用 `pos_weight`；简单负样本特别多时用 focal loss。
- **Label smoothing 的直觉？** one-hot 目标要求真实类别概率等于 1，logits 只能无限拉大，模型越来越过度自信。软目标的最优解是有限的 logits，校准更好，泛化通常也更好（Szegedy et al., 2016 用 ε = 0.1）。副作用是 loss 降不到 0，最低就是软目标分布的熵。定义有两种：PyTorch 把 ε 分给全部 $C$ 类，也有写法把 ε 分给其余 $C-1$ 类，面试时先问清楚。
- **Focal loss 的直觉？** $(1-p_t)^\gamma$ 让分得好的样本几乎不贡献 loss：γ = 2 时 $p_t=0.9$ 的样本权重只有 0.01，$p_t=0.5$ 是 0.25，训练集中在难样本上。α 再调正负样本的整体权重。原论文的默认组合是 γ = 2、α = 0.25：正样本虽然少，α 却小于 0.5，因为大量简单负样本已经被 γ 压掉了。
- **InfoNCE 里的温度起什么作用？** 归一化后的 cosine 只在 [-1, 1]，不除温度的话 softmax 接近均匀分布，模型表达不出「很确定」。τ 越小分布越尖，梯度越集中在最难的负样本上；太小则训练不稳定。CLIP 把温度做成可学习参数（初始 0.07），并限制 1/τ 不超过 100。in-batch negatives 的个数等于 batch size 减 1，所以对比学习通常要大 batch。
- **蒸馏为什么乘 T²？** 不乘时，KL 对学生 logits 的梯度是 $(p^{s}-p^{t})/(NT)$：链式法则从 $z/T$ 带出一个 $1/T$；$T$ 较大时两个分布都接近均匀，$p^{s}-p^{t}$ 本身也约正比于 $1/T$，合起来梯度按 $1/T^2$ 缩小（上面的实验里 $T$ 从 2 到 8，不乘时梯度从 0.0816 降到 0.0049）。乘 $T^2$ 把量级拉回来，改 $T$ 时不用重调和 CE 的权重。
- **reduction 用 mean 还是 sum？跟学习率有什么关系？** 用 sum 时梯度随 batch size 线性变大，改 batch size 就等于改了学习率；用 mean 时梯度量级和 batch size 无关，更好调，所以是默认值。即使用 mean，增大 batch 后每个 epoch 的步数变少，SGD 常按线性缩放规则同比放大学习率（Goyal et al., 2017）。Adam 对 loss 乘常数基本不敏感，因为更新量是 $\hat m/\sqrt{\hat v}$，常数在分子分母里抵消。

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

## 8.3 手写 AUC

8.2 节的 precision、recall 都要先定一个阈值。ROC-AUC 把所有阈值一起评估，只看模型把样本排得对不对，是 CTR 预估、风控、推荐排序里最常用的离线指标。面试常让你不用 sklearn 手写它，追问集中在平局（tie）处理和复杂度上。

### 原理与直觉

把样本按分数从高到低排好，阈值从最高一路往下降。每越过一个样本，它就从「判负」变成「判正」：

- 它是正样本：TP 加 1，纵轴 $\text{TPR}=TP/P$ 往上走 $1/P$。TPR 就是 recall。
- 它是负样本：FP 加 1，横轴 $\text{FPR}=FP/N$ 往右走 $1/N$。FPR 是负样本里被误报的比例。

这里 $P$、$N$ 是正、负样本个数。这条从 $(0,0)$ 走到 $(1,1)$ 的折线就是 ROC 曲线，AUC 是它下面的面积。

**概率含义（面试最常考）**：随机抽一个正样本和一个负样本，AUC 就是正样本分数更高的概率，分数相等算一半：

$$
\text{AUC}=\Pr(s^+>s^-)+\tfrac12\Pr(s^+=s^-)=\frac{1}{PN}\sum_{i\in\text{pos}}\sum_{j\in\text{neg}}\Big(\mathbf{1}[s_i>s_j]+\tfrac12\,\mathbf{1}[s_i=s_j]\Big)
$$

为什么面积等于这个概率：折线每往右走一步（越过一个负样本），新增面积是「宽 $1/N$ 乘当前高度」，当前高度是排在这个负样本前面的正样本数除以 $P$。所有负样本加起来，面积就是「排对的（正, 负）对数」除以 $PN$。分数相同的一组样本属于同一个阈值，曲线在这里走一条斜线，斜线下的梯形正好给组内每个（正, 负）对记 1/2。

- **不依赖阈值**：AUC 把所有阈值扫了一遍，不用像 8.2 节那样先选阈值。
- **只看排序**：对分数做任何严格递增变换（乘正数、加常数、过 sigmoid、取 log），排序不变，AUC 就不变。所以拿 logits 算和拿概率算结果一样。

### 先写核心版

按概率含义直接数对：把正负样本分开，两两比较。时间 $O(PN)$，数据一大就会超过 CodeSignal 的时间限制，但面试时先写它最不容易错，后面也拿它当对拍的标准答案。

```python
def solution(labels: list, scores: list) -> float:
    """labels: 长度 n，1/True 是正类；scores: 长度 n，概率或 logits 都行。返回 [0, 1] 的 AUC。"""
    # 1. 按标签把分数分成两堆；if y 同时兼容 1/0、True/False、1.0/0.0
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    if not pos or not neg:
        raise ValueError("only one class present, AUC is undefined")
    # 2. 每个（正, 负）对：正样本分高记 1，相等记 0.5
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    # 3. 除以总对数 P * N
    return wins / (len(pos) * len(neg))


print(solution([1, 1, 0, 1, 0, 0], [0.9, 0.8, 0.8, 0.4, 0.3, 0.1]))   # 0.8333333333333334
```

手算核对：3 个正样本乘 3 个负样本共 9 对。0.9 赢 3 对；0.8 和负样本 0.8 打平记 0.5，再赢 0.3、0.1，共 2.5；0.4 赢 2 对。合计 7.5 / 9。

### 面试版实现

#### 1. 排名公式（Mann-Whitney U）：O(n log n)

把所有样本按分数**升序**排名，最小的名次是 1，记正样本 $i$ 的名次为 $r_i$：

$$
\text{AUC}=\frac{\sum_{i\in\text{pos}} r_i-\frac{P(P+1)}{2}}{P\cdot N}
$$

推导：没有平局时，$r_i-1$ 是排在正样本 $i$ 下面的样本数，其中负样本个数就是它赢的对数。下面的正样本个数加上它自己，对所有正样本求和是 $1+2+\cdots+P=P(P+1)/2$，减掉这部分就只剩赢的对数（即统计量 $U$）。有平局时，同分的一组人都拿**平均名次**，相当于组内每个（正, 负）对各记 1/2。

```python
def auc_rank(labels: list, scores: list) -> float:
    """排名公式，纯 Python，不 import 任何库。时间 O(n log n)。"""
    n = len(scores)
    n_pos = sum(1 for y in labels if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("only one class present, AUC is undefined")
    order = sorted(range(n), key=lambda i: scores[i])     # 1. 按分数升序排下标
    # 2. 扫描同分组 order[i..j]，它们占名次 i+1 .. j+1，统一给平均名次
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    # 3. 正样本名次和减去 P(P+1)/2，得到赢的对数 U，再除以 P * N
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

NumPy 版用 `np.unique` 一次拿到「每个样本属于第几个分数组」和「每组几个人」，平均名次一行算完：

```python
import numpy as np


def auc_rank_np(labels, scores) -> float:
    y = np.asarray(labels).astype(bool)             # (n,)，True 是正类
    s = np.asarray(scores, dtype=float)             # (n,)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("only one class present, AUC is undefined")
    # 1. uniq 升序；inverse[i] 是 s[i] 所在的组号，(n,)；counts 是每组人数，(K,)
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    # 2. 第 k 组占名次 end-count+1 .. end，平均名次 = end - (count-1)/2
    ends = np.cumsum(counts)                        # (K,)
    ranks = (ends - (counts - 1) / 2.0)[inverse]    # (n,)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
```

#### 2. ROC 曲线 + 梯形面积

按分数降序排，**同分的样本合成一个阈值**，每组结束处记一个 (FPR, TPR) 点，最后用梯形公式求面积。用上面 import 的 numpy：

```python
def roc_curve(labels, scores) -> tuple:
    """返回 (fpr, tpr, thresholds)，形状都是 (K+1,)，K 是不同分数的个数，第一个点是 (0, 0)"""
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)                          # 1. 分数从高到低，(n,)
    y, s = y[order], s[order]
    # 2. 每个同分组的最后一个位置：阈值取到这里时，它和它前面的样本都判正，(K,)
    last = np.concatenate([np.where(np.diff(s) != 0)[0], [s.size - 1]])
    tps = np.cumsum(y)[last]                        # (K,) 累计 TP
    fps = (last + 1) - tps                          # (K,) 累计 FP = 判正的个数 - TP
    if tps[-1] == 0 or fps[-1] == 0:
        raise ValueError("only one class present, AUC is undefined")
    fpr = np.concatenate([[0.0], fps / fps[-1]])    # 3. 前面补上起点 (0, 0)
    tpr = np.concatenate([[0.0], tps / tps[-1]])
    return fpr, tpr, np.concatenate([[np.inf], s[last]])


def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    # 每段梯形面积 = 宽 (Δfpr) * 上下底平均；同分组走斜线，梯形给它一半的分
    # 不用 np.trapz：NumPy 2.0 把它改名为 np.trapezoid，旧名会报 DeprecationWarning
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
```

#### 3. 边界条件

- **只有一类**：$PN=0$，AUC 无定义。上面每个实现都抛 `ValueError`，sklearn 1.2 的 `roc_auc_score` 也抛 `ValueError`。别偷偷返回 0.5，那会掩盖数据切分的问题。
- **平局**：必须给平均名次，或者把同分样本合成一个阈值。逐个样本处理时，结果会随同分样本的排列顺序变化（见下面 Example）。
- **标签类型**：`if y` 和 `.astype(bool)` 同时兼容 int、bool、float 标签。标签是 -1/+1 时 -1 也被当成正类，要先转成 `y == 1`。

### Example

用上面的 `solution`、`auc_rank`、`auc_rank_np`、`roc_curve`、`auc_trapezoid`：

```python
import numpy as np

labels = [1, 1, 0, 1, 0, 0]
scores = [0.9, 0.8, 0.8, 0.4, 0.3, 0.1]
fpr, tpr, _ = roc_curve(labels, scores)
print(np.round(fpr, 4))     # [0.     0.     0.3333 0.3333 0.6667 1.    ]
print(np.round(tpr, 4))     # [0.     0.3333 0.6667 1.     1.     1.    ]
print([f(labels, scores) for f in (solution, auc_rank, auc_rank_np)], auc_trapezoid(fpr, tpr))
# [0.8333333333333334, 0.8333333333333334, 0.8333333333333334] 0.8333333333333334
```

第二个点到第三个点是一条斜线：阈值降到 0.8 时一个正样本和一个负样本同时变成「判正」，这就是那对平局。

平局处理错会怎样：下面的反例用 `argsort` 两次得到普通名次，同分样本按出现顺序拿 1、2、3、4，不取平均。另外，严格单调变换不改变 AUC，但浮点数会「制造」平局：logits 是 38 和 40 时，sigmoid 在 float64 下都舍入成 1.0，所以离线评估时直接用 logits 算 AUC 更稳。

```python
same = [0.5, 0.5, 0.5, 0.5]                                 # 分数全一样，正确答案是 0.5
naive = np.argsort(np.argsort(same, kind="stable")) + 1    # (4,)，错误示范：名次 1, 2, 3, 4
for y in ([1, 0, 1, 0], [0, 1, 0, 1]):
    pos_rank_sum = naive[np.array(y) == 1].sum()
    print((pos_rank_sum - 3) / 4, auc_rank(y, same))        # P = N = 2：减 P(P+1)/2 = 3，除以 PN = 4
# 0.25 0.5
# 0.75 0.5

logits = np.array([38.0, 40.0])                     # 负样本 38，正样本 40
probs = 1 / (1 + np.exp(-logits))                   # (2,)
print(probs, auc_rank_np([0, 1], logits), auc_rank_np([0, 1], probs))   # [1. 1.] 1.0 0.5
```

### 变体：GAUC（推荐 / 广告常考）

全局 AUC 会拿用户 A 的正样本和用户 B 的负样本比。可是线上排序只在**同一个用户**的候选里进行，跨用户的比较对业务没有意义：模型只要学会「活跃用户点得多」，全局 AUC 就能很高，每个用户内部的排序却可能很差。GAUC 先对每个用户单独算 AUC，再加权平均：

$$
\text{GAUC}=\frac{\sum_u w_u\,\text{AUC}_u}{\sum_u w_u}
$$

权重 $w_u$ 常取该用户的曝光数（impressions），阿里的 DIN 论文（Zhou et al., 2018）就按曝光数加权；也有按点击数加权的。只有一类样本的用户（全没点或全点了）AUC 无定义，直接跳过。用上面的 `auc_rank`：

```python
def gauc(user_ids: list, labels: list, scores: list, weight: str = "impression") -> float:
    # 1. 按用户分组，记下每个用户的样本下标
    groups = {}
    for i, u in enumerate(user_ids):
        groups.setdefault(u, []).append(i)
    num, den = 0.0, 0.0
    for idx in groups.values():
        y = [labels[i] for i in idx]
        n_pos = sum(1 for v in y if v)
        if n_pos == 0 or n_pos == len(y):          # 2. 只有一类：跳过
            continue
        w = len(idx) if weight == "impression" else n_pos   # 3. 曝光数或点击数加权
        num += w * auc_rank(y, [scores[i] for i in idx])
        den += w
    if den == 0:
        raise ValueError("no user has both positive and negative samples")
    return num / den


users = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
labels = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]            # 用户 c 没有点击，会被跳过
scores = [0.8, 0.9, 0.7, 0.6, 0.3, 0.2, 0.1, 0.4, 0.5, 0.35]
print(auc_rank(labels, scores), gauc(users, labels, scores))   # 0.6190476190476191 0.41666666666666663
```

全局 AUC 是 0.619，看着还行。单独算的话用户 a 是 0.5、用户 b 是 0.3333，都不比随机好，GAUC 只有 0.4167：模型只学会了「用户 a 的分数整体偏高」。

### 自测

模仿 CodeSignal 的 hidden tests：随机数据（分数只有 5 种取值，平局很多），另外三个实现都和暴力版对拍，再检查单调不变性、标签类型和只有一类的情况。用上面定义的所有函数：

```python
import numpy as np


def run_tests() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 40))
        y = np.concatenate([[0, 1], rng.integers(0, 2, n - 2)])   # (n,)，前两个保证两类都在
        s = rng.integers(0, 5, n) / 4.0             # (n,)，只有 5 种取值
        ref = solution(y.tolist(), s.tolist())
        trap = auc_trapezoid(*roc_curve(y, s)[:2])
        for got in (auc_rank(y.tolist(), s.tolist()), auc_rank_np(y, s), trap):
            assert abs(got - ref) < 1e-12
        # 严格递增变换和 bool 标签不改变 AUC；分数取反得到 1 - AUC
        assert abs(auc_rank_np(y.astype(bool), np.exp(3 * s)) - ref) < 1e-12
        assert abs(auc_rank_np(y, -s) - (1 - ref)) < 1e-12
    for f in (solution, auc_rank, auc_rank_np, roc_curve):   # 只有一类必须报错
        try:
            f([1, 1, 1], [0.1, 0.2, 0.3])
        except ValueError:
            continue
        raise AssertionError(f"{f.__name__} should raise ValueError")
    print("all tests passed")


if __name__ == "__main__":
    run_tests()
```

本地另外和 `sklearn.metrics.roc_auc_score`、`sklearn.metrics.roc_curve(drop_intermediate=False)` 对拍过（随机连续分数、大量平局、n = 2 的极小输入、bool 标签），结果一致，误差在 1e-12 以内。写法就是 `assert abs(auc_rank(y, s) - roc_auc_score(y, s)) < 1e-12`。

### 关键追问

- **类别不平衡时，为什么看 AUC 不看 accuracy？** accuracy 依赖阈值和正例比例：1% 正例时全判负就有 99%。AUC 的 TPR、FPR 各自在本类内部归一化，正负比例变了 ROC 曲线不变；常数分数的 AUC 恰好是 0.5，随机打分的期望也是 0.5，不会因为不平衡虚高。
- **ROC-AUC 和 PR-AUC 怎么选？** 负样本极多时，FPR 的分母 $N$ 很大，多出一堆误报 FPR 也几乎不动，ROC-AUC 看起来偏乐观；precision 直接受误报影响，所以 PR-AUC 对不平衡更敏感。随机模型的 ROC-AUC 是 0.5，PR-AUC 约等于正例比例。PR 曲线见 8.2 节和 [02. 模型评估与指标](02-evaluation.md)。
- **为什么平局这么重要？** 树模型、分桶后的分数、量化后的模型常常输出大量相同的分数。平局不按 0.5 记，AUC 会随样本顺序变化（上面同一组常数分数算出 0.25 和 0.75）。平均名次和「同分合成一个阈值」两种做法都是对的。
- **复杂度？** 暴力 $O(PN)$；排名公式和 ROC 曲线都是 $O(n\log n)$，瓶颈在排序。数据量大到无法全量排序时（分布式、流式），把分数分成固定个数的桶，只累计每个桶的正负样本数，就能 $O(n)$ 近似计算，`tf.keras.metrics.AUC` 就是这么做的（默认 200 个阈值）。
- **AUC = 0.5 和 AUC < 0.5 说明什么？** 0.5 是随机排序或常数分数。小于 0.5 说明排序系统性地反了，把分数取反就得到 1 - AUC；实际中多半是 bug，例如标签取反，或者用了 `predict_proba[:, 0]`（负类概率）。
- **AUC 高，概率就准吗？** 不一定。AUC 只看排序，把所有概率除以 2 AUC 不变，校准却全错了。CTR 预估要拿概率去出价，还得看 logloss 和校准（见 02. 模型评估与指标 的第 4 节）。
- **能直接拿 AUC 当 loss 训练吗？** 不能。它由指示函数 $\mathbf{1}[s_i>s_j]$ 组成，几乎处处梯度为 0。常用可导的 pairwise 替代：对每个（正, 负）对最小化 $-\log\sigma(s_i-s_j)$，就是 RankNet / BPR loss。

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
- **SGD 和 Adam 怎么选？** Adam 给每个参数自适应步长，对学习率不太敏感、收敛快，Transformer 和 LLM 基本都用它的变体 AdamW。SGD + Momentum 有时泛化更好（Wilson et al., 2017, *The Marginal Value of Adaptive Gradient Methods in Machine Learning*），而且省显存：Adam 每个参数要多存 $m$、$v$ 两份状态，SGD + Momentum 只多存一份速度。
- **AdamW 改了什么？** Adam 把 L2 正则加进梯度后，正则项也会被 $\sqrt{\hat v}$ 除掉，梯度大的参数被衰减得少。AdamW 把 weight decay 从梯度里拿出来，单独做一步 `p -= lr * wd * p`。

---

## 10.1 PyTorch Training-Validation Loop

### 原理与直觉

第 11 节用 NumPy 手写了前向、反向，再用第 10 节的 `SGD` 更新参数。换到 PyTorch 以后，反向由 autograd 自动完成，更新交给 `torch.optim`，剩下的训练循环是一个固定套路。每个 epoch 分成两半：

- **训练**：遍历训练集的每个 batch，做五步：清梯度、前向、算 loss、反向、更新参数。
- **验证**：在验证集上只做前向，算 loss 和 accuracy，不更新参数。验证集的结果用来挑模型、决定什么时候停，所以它必须是训练时没见过的数据；也正因为它参与了挑模型，最终效果要在另一份从没用过的测试集上报告。

五步和第 11 节手写版一一对应：

| PyTorch                        | 第 11 节手写版                                                               |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `optimizer.zero_grad()`        | 手写版不需要；PyTorch 的 `.grad` 默认**累加**，每次 `backward()` 都加到旧值上 |
| `logits = model(xb)`           | `linear_forward` → `relu_forward` → `linear_forward`                         |
| `loss = criterion(logits, yb)` | `softmax_cross_entropy` 算出的 loss                                          |
| `loss.backward()`              | 倒着调用每层的 backward，梯度存进每个参数的 `.grad`                          |
| `optimizer.step()`             | 第 10 节的 `opt.step(grads)`                                                 |

验证时 `model.eval()` 和 `with torch.no_grad():` 两个都要写：前者让 Dropout、BatchNorm（11.1 节）切到推理时的行为，后者只是不建计算图，不会关掉 Dropout。区别见关键追问第一条。

### 先写核心版

数据用代码合成：二维点，一三象限为 1、二四象限为 0，和第 11 节是同一个任务（一条直线分不开）。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X = torch.randn(1000, 2)                                   # (N, 2)
y = (X[:, 0] * X[:, 1] > 0).long()                         # (N,) 类别标签必须是 long
train_loader = DataLoader(TensorDataset(X[:800], y[:800]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[800:], y[800:]), batch_size=64, shuffle=False)

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
criterion = nn.CrossEntropyLoss()                          # 吃 logits，内部做 log-softmax
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    model.train()                                          # 训练模式
    train_loss = 0.0
    for xb, yb in train_loader:                            # xb: (B, 2), yb: (B,)
        optimizer.zero_grad()                              # 1. 清掉上一步留下的梯度
        logits = model(xb)                                 # 2. 前向 (B, 2)
        loss = criterion(logits, yb)                       # 3. 标量，batch 内取平均
        loss.backward()                                    # 4. 反向，梯度写进每个参数的 .grad
        optimizer.step()                                   # 5. 用 .grad 更新参数
        train_loss += loss.item() * xb.size(0)             # .item() 取 Python float，按样本数加权
    train_loss /= len(train_loader.dataset)

    model.eval()                                           # 评估模式
    val_loss, correct = 0.0, 0
    with torch.no_grad():                                  # 不建计算图
        for xb, yb in val_loader:
            logits = model(xb)                             # (B, 2)
            val_loss += criterion(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    print(f"Epoch {epoch + 1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.3f}")
# Epoch 1, train loss: 0.5898, val loss: 0.4655, val acc: 0.915
# ...
# Epoch 10, train loss: 0.0866, val loss: 0.1082, val acc: 0.950
```

注释里的输出在 torch 1.13.1、2.4.1、2.8.0 上完全一致（本节其余输出也一样），换别的版本或机器，数字可能略有不同。面试时先写这一版，边写边念五步。写完再主动说：「接下来我会拆成函数，加上 early stopping、保存最好的权重、梯度裁剪和学习率调度。」

### 面试版实现

拆成三个函数：`train_one_epoch` 训练一遍返回平均 loss，`evaluate` 返回 `(loss, accuracy)`，`fit` 负责多个 epoch、early stopping 和记录 history。模型里放了 Dropout，这样 `train()` 和 `eval()` 的区别是真实存在的。

```python
import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                                 # (B, in_dim) -> (B, num_classes)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    max_grad_norm: Optional[float] = None) -> float:
    """
    Args:
        loader: 每次给出 xb (B, D) 和 yb (B,)
        max_grad_norm: 不为 None 时梯度裁剪：所有参数的梯度拼成一个向量，L2 范数超过它就整体等比例缩小，方向不变
    Returns:
        这个 epoch 按样本数加权的平均训练 loss
    """
    model.train()
    total_loss, total_count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)              # (B, D), (B,)
        optimizer.zero_grad()                              # 1. 清梯度
        logits = model(xb)                                 # 2. 前向 (B, C)
        loss = criterion(logits, yb)                       # 3. 标量
        loss.backward()                                    # 4. 反向
        if max_grad_norm is not None:                      # 裁剪改的是 .grad，放在 backward 之后、step 之前
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()                                   # 5. 更新
        total_loss += loss.item() * xb.size(0)             # 最后一个 batch 可能更小，按样本数加权
        total_count += xb.size(0)
    return total_loss / total_count


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """
    在 loader 上评估，不更新参数。
    Returns:
        (按样本数加权的平均 loss, accuracy)
    """
    model.eval()                                           # 关 Dropout，BatchNorm 用 running 统计量
    total_loss, correct, total_count = 0.0, 0, 0
    with torch.no_grad():                                  # 不建计算图
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)                             # (B, C)
            total_loss += criterion(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total_count += xb.size(0)
    return total_loss / total_count, correct / total_count


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
        epochs: int = 100, patience: int = 10, scheduler: Optional[Any] = None,
        max_grad_norm: Optional[float] = None, verbose: bool = True) -> Dict[str, List[float]]:
    """
    多 epoch 训练。val loss 连续 patience 个 epoch 没有变好就停，最后恢复最好那个 epoch 的权重。
    Returns:
        history: 每个 epoch 的 train_loss、val_loss、val_acc、lr
    """
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    for epoch in range(epochs):
        lr = optimizer.param_groups[0]["lr"]               # 这个 epoch 实际用的学习率
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)
        if verbose:
            print(f"Epoch {epoch + 1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.3f}")

        # 1. 学习率调度：按 epoch 调的 scheduler，放在这个 epoch 所有 optimizer.step() 之后
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)                       # 这一种要传入它监控的指标
        elif scheduler is not None:
            scheduler.step()

        # 2. early stopping：val loss 变好就存一份权重，否则计数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())   # 必须深拷贝，见关键追问
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)                      # 3. 恢复最好的权重
    return history
```

### Example：自测

数据还是象限任务，加了两处让它更像真实数据：10% 的标签随机翻转（模型有机会过拟合，early stopping 才有事做），两个特征尺度差很多（标准化才有意义）。标准化的 mean、std **只用训练集算**，验证集用同一组数，否则验证集的信息就泄露进了训练。

```python
def make_data(n: int, noise: float = 0.1, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """象限分类：一三象限为 1；noise 比例的标签随机翻转；两个特征尺度不同"""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, 2, generator=g)                     # (n, 2)
    y = (z[:, 0] * z[:, 1] > 0).long()                     # (n,)
    flip = torch.rand(n, generator=g) < noise              # (n,) 标签噪声
    y[flip] = 1 - y[flip]
    X = z * torch.tensor([10.0, 0.1]) + torch.tensor([50.0, -3.0])   # (n, 2) 故意拉开尺度
    return X, y


def test_training_loop() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")                           # 有 GPU 时换成 "cuda"
    X, y = make_data(1300)
    X_train, y_train, X_val, y_val = X[:1000], y[:1000], X[1000:], y[1000:]

    # 1. 标准化只用训练集的统计量
    mean, std = X_train.mean(dim=0), X_train.std(dim=0)    # (2,), (2,)
    X_train, X_val = (X_train - mean) / std, (X_val - mean) / std

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)  # 300 条，最后一批 44 条
    model = MLP(2, 64, 2).to(device)                       # 习惯上先 .to(device)，再建 optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    history = fit(model, train_loader, val_loader, criterion, optimizer, device,
                  epochs=100, patience=8, scheduler=scheduler, max_grad_norm=1.0, verbose=False)

    # 2. 恢复的是最好那个 epoch 的权重
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    assert abs(val_loss - min(history["val_loss"])) < 1e-7
    # 3. 按样本数加权的平均，和一次性在整个验证集上算的 loss 一致
    with torch.no_grad():
        full_loss = criterion(model(X_val.to(device)), y_val.to(device)).item()
    assert abs(val_loss - full_loss) < 1e-5
    # 4. 提前停了，而且正好停在最好 epoch 之后的第 patience 个 epoch
    best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
    assert len(history["val_loss"]) == best_epoch + 8 < 100
    # 5. loss 下降；准确率明显高于随机猜的 0.5
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert val_acc > 0.8
    # 6. eval 模式关了 Dropout，评估两次结果完全一样
    assert evaluate(model, val_loader, criterion, device) == (val_loss, val_acc)
    # 7. StepLR 每 10 个 epoch 减半：第 10 个 epoch 还是 0.01，第 11 个变成 0.005
    assert history["lr"][9:11] == [0.01, 0.005]
    print(f"stopped at epoch {len(history['val_loss'])}, best epoch {best_epoch}, "
          f"train loss {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}, val acc {val_acc:.3f}")
    print("all tests passed")


if __name__ == "__main__":
    test_training_loop()
# stopped at epoch 34, best epoch 26, train loss 0.5745 -> 0.3799, val acc 0.913
# all tests passed
```

val loss 在第 26 个 epoch 最低，之后连续 8 个 epoch 没再变好，第 34 个 epoch 停下，模型恢复成第 26 个 epoch 的权重。验证集 300 条里有 24 条标签被翻转过，一个完全学会象限规则的模型在这里也只能拿到 0.920，所以 0.913 已经接近上限。

### 先过拟合一个小 batch

正式训练前先做个冒烟测试：只拿一个 batch 反复训练几百步，loss 应该能压到接近 0。压不下去，说明模型、loss 或训练循环本身有 bug，这时跑完整训练只是浪费时间。做这个检查时先关掉 Dropout、weight decay 这类正则。

比如在下面的例子里删掉 `zero_grad`，试了 5 个种子，loss 都停在 0.4 以上；对 logits 又做一次 softmax，二分类的 loss 最低只能到 $\log(1+e^{-1})\approx0.313$，永远压不到 0。这个检查查不出标签错位：把 32 个标签随机打乱，网络照样能背下来，loss 一样压到 0.01 以下，数据问题要另外查。

```python
def overfit_one_batch(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor,
                      steps: int = 300, lr: float = 0.01) -> float:
    """在同一个 batch 上反复训练，返回最后一步的 loss"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)                    # xb: (B, D), yb: (B,)
        loss.backward()
        optimizer.step()
    return loss.item()


torch.manual_seed(0)
X, y = make_data(32)                                       # 用上面的 make_data，一个 batch：(32, 2), (32,)
X = (X - X.mean(dim=0)) / X.std(dim=0)                     # 只看能不能记住这 32 条，直接用自身统计量
final_loss = overfit_one_batch(MLP(2, 64, 2, dropout=0.0), X, y)
assert final_loss < 0.01
print("overfit one batch ok")
```

### 梯度累积

显存只够 batch 为 8，但想要 batch 为 32 的效果：连续做 4 次前向和 `backward()`，中间不清梯度，攒够 4 次再 `step()`。因为 `CrossEntropyLoss` 默认对 batch 取平均，每个小 batch 的 loss 要先除以累积步数 4，攒起来的梯度才等于大 batch 的平均梯度。记第 $k$ 块 8 个样本的下标集合为 $B_k$（下面的代码直接对拍）：

$$
\nabla\Big(\frac{1}{32}\sum_{i=1}^{32}\ell_i\Big)=\sum_{k=1}^{4}\nabla\Big(\frac{1}{4}\cdot\frac{1}{8}\sum_{i\in B_k}\ell_i\Big)
$$

```python
torch.manual_seed(0)
model = MLP(2, 16, 2, dropout=0.0)                         # 关掉 Dropout，两次前向才可比
criterion = nn.CrossEntropyLoss()
xb, yb = torch.randn(32, 2), torch.randint(0, 2, (32,))    # (32, 2), (32,)
# 1. 一个大 batch
model.zero_grad()
criterion(model(xb), yb).backward()
big_grads = [p.grad.clone() for p in model.parameters()]
# 2. 梯度累积：4 个小 batch，loss 先除以 4，中间不清梯度
accum_steps = 4
model.zero_grad()
for x_small, y_small in zip(xb.chunk(accum_steps), yb.chunk(accum_steps)):   # 每块 (8, 2), (8,)
    loss = criterion(model(x_small), y_small) / accum_steps
    loss.backward()                                        # .grad 在 4 次之间累加
acc_grads = [p.grad.clone() for p in model.parameters()]
print(all(torch.allclose(a, b, atol=1e-6) for a, b in zip(big_grads, acc_grads)))   # True
```

放进训练循环时，`zero_grad` 和 `step` 都改成每 `accum_steps` 个 batch 做一次。上面的等式要求每块样本数相同，模型里也没有 BatchNorm（BN 的统计量按小 batch 算）。另外两种情况也不完全等价：

- 最后不满 `accum_steps` 的尾巴也要 step 一次，它的梯度只有正常的几分之一（尾巴 1 块就是 1/4），一般可以接受。
- 序列任务里每块的真实 token 数不同，各块取平均再除以 4 会有偏差。严格的写法是用 `reduction="sum"`，最后除以总 token 数（6.1 节）。

```python
def train_one_epoch_accum(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                          optimizer: torch.optim.Optimizer, accum_steps: int) -> None:
    """每 accum_steps 个 batch 更新一次参数；loader 每次给出 xb (B, D) 和 yb (B,)"""
    model.train()
    optimizer.zero_grad()
    for i, (xb, yb) in enumerate(loader):
        loss = criterion(model(xb), yb) / accum_steps      # 除以累积步数
        loss.backward()                                    # 梯度累加，先不清
        if (i + 1) % accum_steps == 0 or i + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad()                          # step 之后才清
```

### 改错题：修一个坏掉的训练脚本

面试里有一类改错题：给一段「能跑完、不报错，但结果不对」的训练脚本，让你找 bug。下面这段藏了 8 个 bug，另有一处不影响结果的浪费。先自己找，再看注释：

```python
torch.manual_seed(0)
X = torch.randn(600, 2) * torch.tensor([10.0, 0.1])
y = (X[:, 0] * X[:, 1] > 0).long()
X = (X - X.mean(dim=0)) / X.std(dim=0)                     # BUG 1：用全部数据算 mean/std，验证集信息泄露
train_loader = DataLoader(TensorDataset(X[:500], y[:500]), batch_size=32, shuffle=False)  # BUG 2：训练集没打乱
val_loader = DataLoader(TensorDataset(X[500:], y[500:]), batch_size=64)
model = MLP(2, 64, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
for epoch in range(3):
    train_loss = 0
    for xb, yb in train_loader:                            # BUG 3：没调 model.train()，第 2 轮起 Dropout 是关的
        logits = model(xb)
        loss = criterion(torch.softmax(logits, dim=1), yb)   # BUG 4：CrossEntropyLoss 要 logits，又做了一次 softmax
        loss.backward()                                    # BUG 5：没有 zero_grad，梯度越攒越大
        optimizer.step()
        scheduler.step()                                   # BUG 6：按 epoch 设计的 scheduler 放进了 batch 循环
        train_loss += loss                                 # BUG 7：累加的是张量，所有计算图都留在内存里（基础篇 0.3 节）
    model.eval()
    correct = 0
    for xb, yb in val_loader:                              # 少了 torch.no_grad()（不影响结果，但浪费内存）
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
    print(f"Epoch {epoch + 1}, train loss: {train_loss / len(train_loader):.4f}, "
          f"val acc: {correct / len(val_loader):.3f}")    # BUG 8：都除以了 batch 数，准确率会大于 1
# Epoch 1, train loss: 0.6369, val acc: 37.500
# Epoch 2, train loss: 0.5962, val acc: 43.000
# Epoch 3, train loss: 0.6518, val acc: 27.500
```

输出里有两个一眼能看出的信号：准确率大于 1，train loss 不降。改好的写法就是上面的 `train_one_epoch` 和 `evaluate`。拿到这类题，按这个清单逐行扫：

1. **数据**：标准化、填缺失值、建词表是否只用了训练集；训练集 `shuffle=True`，验证集 `shuffle=False`；特征和标签有没有错位。
2. **标签和 loss 的输入**：`CrossEntropyLoss` 吃 logits，标签是 `long` 型类别下标，形状 `(B,)`，取值 `0..C-1`；`BCEWithLogitsLoss` 也吃 logits，标签是 float，形状和 logits 一样。模型最后都不要再接 softmax / sigmoid。
3. **五步顺序**：`zero_grad` → 前向 → loss → `backward` → `step`，一步不能少，顺序不能乱。
4. **模式切换**：每个 epoch 训练前 `model.train()`，验证前 `model.eval()`，验证包在 `torch.no_grad()` 里。
5. **scheduler**：放在 `optimizer.step()` 之后；按 epoch 调的放在 batch 循环外；`ReduceLROnPlateau` 要传 val loss。
6. **统计量**：累加 `loss.item()`；平均按样本数；准确率除以样本数；结果应该落在合理范围（准确率在 0 到 1 之间）。
7. **保存模型**：最好的权重用 `copy.deepcopy(model.state_dict())`。

### 关键追问

- **`model.eval()` 和 `torch.no_grad()` 有什么区别？** `eval()` 改变层的计算方式：Dropout 不丢神经元，BatchNorm 用 running 统计量。`no_grad()` 只是不记录计算图，前向结果不变，但省掉了为反向保存的中间激活，内存更少、速度更快。验证只用 `no_grad()`，Dropout 照样在随机丢，结果每次都不一样；只用 `eval()`，结果对，但白白建了计算图。纯推理还可以用 `torch.inference_mode()`，开销比 `no_grad()` 更小，但它算出的张量之后不能再参与反向。
- **`zero_grad` 放在哪里？** 放在上一次 `step()` 之后、这一次 `backward()` 之前都行，常见写法是每个 batch 一开始。忘了写，每一步的梯度都混进了之前所有 batch 的梯度，越攒越大，更新方向也被旧梯度带偏。torch 2.x 的 `zero_grad()` 默认把 `.grad` 设成 `None`，1.13 默认填 0，对正常训练没有区别（基础篇 0.3 节）。
- **平均 loss 为什么要按样本数加权？** `criterion` 返回的是 batch 内的平均。最后一个 batch 往往更小（上面 300 条验证集、batch 64，最后一批只有 44 条），直接对每个 batch 的平均再求平均，等于给这 44 条更大的权重。正确做法是 `loss.item() * batch_size` 累加，最后除以总样本数，结果和在整个数据集上一次性算的一样（自测里已对拍）。为什么累加的是 `loss.item()`，见基础篇 0.3 节。
- **为什么 val loss 有时比 train loss 还低？** 核心版第 1 个 epoch 就是这样（train 0.5898，val 0.4655）。train loss 是一个 epoch 里边训练边累加的平均，前面的 batch 用的是还没训好的参数；val loss 在 epoch 结束后用最新的参数算。模型有 Dropout 时，train loss 还是在随机丢神经元的状态下算的，也会偏高。
- **为什么要 `copy.deepcopy(model.state_dict())`？** `state_dict()` 返回的张量和模型参数共享内存。只写 `best_state = model.state_dict()`，之后每次 `optimizer.step()` 都会改到它，最后「恢复」的其实是最后一个 epoch 的权重。
- **`scheduler.step()` 放在哪？** `optimizer.step()` 之后，这是 PyTorch 1.1 以后的约定。写反了，第一次调用时会给出 UserWarning，而且第一次 `optimizer.step()` 用的已经是表里的第二个学习率，第一个值被跳过。`StepLR`、`CosineAnnealingLR` 一般每个 epoch 调一次；`OneCycleLR` 和 warmup 一般每个 batch 调一次，要放进 `train_one_epoch` 的 batch 循环，总步数按 batch 数算。
- **序列任务的循环要改什么？** loss 的写法见 6.1 节：`(B, T, V)` 的 logits 先 reshape 成 `(B*T, V)`，用 `nn.CrossEntropyLoss(ignore_index=pad_id)` 跳过 padding。统计量跟着改：每个 batch 的 loss 是真实 token 上的平均，累加时乘这个 batch 的真实 token 数 `(yb != pad_id).sum().item()`，最后除以总 token 数；token 准确率也只在 `yb != pad_id` 的位置上算。
- **混合精度怎么加？** GPU 上用 `torch.autocast(device_type="cuda", dtype=torch.float16)` 包住前向和 loss，再用 GradScaler 把 loss 放大，防止 fp16 的小梯度下溢成 0：`scaler.scale(loss).backward()`、`scaler.step(optimizer)`、`scaler.update()`；要裁剪梯度，先 `scaler.unscale_(optimizer)`。torch 2.4、2.8 上写 `torch.amp.GradScaler("cuda")`，旧写法 `torch.cuda.amp.GradScaler()` 会给出 FutureWarning；torch 1.13 只有旧写法。用 bf16 时不需要 GradScaler，它的指数范围和 fp32 一样，不容易下溢。

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

第 6 节提过结论，这里补上推导：

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

公式和 2.2 节的 LayerNorm 完全一样，区别只在统计量沿哪个轴算：BatchNorm 沿 **batch 维**（`axis=0`，每个特征跨样本求均值），LayerNorm 沿**特征维**（`axis=-1`，每个样本跨特征求均值）。

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

- **BatchNorm 沿哪个轴？** 输入是 $(N,D)$ 时沿 `axis=0`，每个特征跨样本算一组统计量。序列输入 $(B,T,D)$ 一般不用 BatchNorm，改用沿 `axis=-1` 的 LayerNorm（见 2.2 节）。
- **训练和推理有什么不同？** 训练用当前 batch 的统计量，推理用 running 统计量。PyTorch 里靠 `model.train()` 和 `model.eval()` 切换，推理前忘了调 `eval()` 是经典 bug：结果会随 batch 里其他样本变化。
- **为什么要 $\gamma$ 和 $\beta$？** 强行标准化会限制表达能力，比如 sigmoid 前的输入被压到 0 附近，只剩近似线性的一段。有了 $\gamma,\beta$，网络可以学回需要的尺度和偏移；$\gamma=\sqrt{\sigma^2+\epsilon}$、$\beta=\mu$ 时完全还原原始输入。
- **running_var 用有偏还是无偏方差？** 标准化当前 batch 用有偏方差（除以 $N$）。PyTorch 更新 `running_var` 时用的是无偏方差（除以 $N-1$），上面的实现为了简单没有区分。
- **batch 很小时会怎样？** 统计量噪声很大，效果明显变差。batch 为 1 时方差恒为 0，输出全是 $\beta$，PyTorch 在训练模式下会直接报错。这时改用 LayerNorm 或 GroupNorm，它们不依赖 batch。
- **BatchNorm 为什么有效？** 原论文的解释是减少 internal covariate shift。后来的研究（Santurkar et al., 2018, *How Does Batch Normalization Help Optimization?*）认为主要原因是让 loss 地形更平滑，允许用更大的学习率。

---

## 11.2 Softmax 单独的反向传播

### 原理与直觉

第 11 节只推了 softmax 和交叉熵**合在一起**的梯度 $(p-\text{onehot})/N$。很多时候 softmax 后面接的是别的运算：注意力里 $P=\operatorname{softmax}(QK^\top/\sqrt{d_k})$ 后面是乘 $V$（第 1 节），也可能接 MSE、KL 或自定义的 loss。这时 softmax 是一个独立的层，backward 收到任意的上游梯度 $dy=\partial\mathcal L/\partial y$，要算出 $dx=\partial\mathcal L/\partial x$。

对一个样本，$y_i=e^{x_i}/\sum_k e^{x_k}$，用商的求导法则对 $x_j$ 求导：

$$
\frac{\partial y_i}{\partial x_j}=y_i(\delta_{ij}-y_j),\qquad J=\operatorname{diag}(y)-yy^\top
$$

$i=j$ 时是 $y_i(1-y_i)$，$i\ne j$ 时是 $-y_iy_j$。backward 要算的是上游梯度乘 Jacobian（vector-Jacobian product）：

$$
dx_j=\sum_i dy_i\,\frac{\partial y_i}{\partial x_j}=y_j\,dy_j-y_j\sum_i dy_i\,y_i
\qquad\Longrightarrow\qquad
dx=y\odot\Big(dy-\sum_i dy_i\,y_i\Big)
$$

括号里的求和是 $dy$ 和 $y$ 的点积，每行一个标量。

**为什么不把 Jacobian 建出来？** 每个样本的 $J$ 是 $(C,C)$，一个 batch 就是 $(N,C,C)$，内存和计算都是 $O(NC^2)$。上面的公式只要一次逐元素乘和一次按行求和，是 $O(NC)$。词表 $C=32000$（LLaMA 的词表大小）时，一个 token 的 $J$ 就有 $1.024\times10^9$ 个元素，float32 约 4 GB。

log-softmax 同理。$z=x-\log\sum_k e^{x_k}$，所以 $\partial z_i/\partial x_j=\delta_{ij}-y_j$，代入得：

$$
dx=dz-\operatorname{softmax}(x)\sum_i dz_i
$$

### 先写核心版

两个 backward 都只用**前向输出**：softmax 缓存 $y$，log-softmax 缓存 $z$（因为 $\operatorname{softmax}(x)=e^{z}$），都不需要输入 $x$。`axis=-1` 加 `keepdims=True` 让任意前导维都能用，注意力的 `(B, H, T, T)` 也一样。

```python
import numpy as np


def softmax_forward(x: np.ndarray) -> np.ndarray:
    """x: (N, C) -> y: (N, C)，每行和为 1"""
    e = np.exp(x - x.max(axis=-1, keepdims=True))   # 1. (N, C) 减每行最大值，防上溢
    return e / e.sum(axis=-1, keepdims=True)        # 2. (N, C) / (N, 1) -> (N, C)


def softmax_backward(dy: np.ndarray, y: np.ndarray) -> np.ndarray:
    """dy: (N, C) 上游梯度；y: (N, C) 前向输出。返回 dx: (N, C)"""
    dot = (dy * y).sum(axis=-1, keepdims=True)      # 1. (N, 1) 每行的 Σ_i dy_i y_i
    return y * (dy - dot)                           # 2. (N, C) 广播相减，再逐元素乘 y


def log_softmax_forward(x: np.ndarray) -> np.ndarray:
    """x: (N, C) -> z: (N, C)，z = x - logsumexp(x)"""
    shifted = x - x.max(axis=-1, keepdims=True)                          # (N, C)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def log_softmax_backward(dz: np.ndarray, z: np.ndarray) -> np.ndarray:
    """dz: (N, C) 上游梯度；z: (N, C) 前向输出。返回 dx: (N, C)"""
    return dz - np.exp(z) * dz.sum(axis=-1, keepdims=True)   # exp(z) 就是 softmax(x)
```

### 自测

`numerical_grad` 和 `rel_error` 与第 11 节相同，这里重写一遍，让本节单独也能跑。检查任意上游梯度用第 11 节的技巧：令 $\mathcal L=\sum(y\odot dy)$，则 $\partial\mathcal L/\partial y=dy$。

```python
import numpy as np
import torch


def numerical_grad(f, x, h=1e-5):
    """和第 11 节相同：中心差分。f 是无参函数，返回标量；x 会被临时扰动"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        old = x.flat[i]
        x.flat[i] = old + h
        f_plus = f()
        x.flat[i] = old - h
        f_minus = f()
        x.flat[i] = old                                  # 一定要还原
        grad.flat[i] = (f_plus - f_minus) / (2 * h)
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def run_softmax_tests() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5)) * 3                  # (N, C)
    dy = rng.standard_normal((4, 5))                     # (N, C) 随便造的上游梯度

    # 1. 梯度检查：softmax 和 log-softmax
    dx = softmax_backward(dy, softmax_forward(x))
    err = rel_error(dx, numerical_grad(lambda: np.sum(softmax_forward(x) * dy), x))
    dx_log = log_softmax_backward(dy, log_softmax_forward(x))
    err_log = rel_error(dx_log, numerical_grad(lambda: np.sum(log_softmax_forward(x) * dy), x))
    print(f"softmax {err:.1e}, log_softmax {err_log:.1e}")   # softmax 1.4e-08, log_softmax 2.2e-10
    assert err < 1e-7 and err_log < 1e-7
    assert np.allclose(dx.sum(axis=-1), 0)               # 每行梯度之和为 0（见关键追问）

    # 2. 和 torch autograd 对拍：float64，用注意力形状 (B, H, T, T)
    s, ds = rng.standard_normal((2, 3, 4, 4)), rng.standard_normal((2, 3, 4, 4))
    pairs = [(softmax_forward, softmax_backward, torch.softmax),
             (log_softmax_forward, log_softmax_backward, torch.log_softmax)]
    for fwd, bwd, ref in pairs:
        st = torch.tensor(s, requires_grad=True)         # float64，和 numpy 精度一致
        ref(st, dim=-1).backward(torch.tensor(ds))       # 把 ds 当上游梯度传进去
        assert np.allclose(bwd(ds, fwd(s)), st.grad.numpy(), atol=1e-12)

    # 3. 接上交叉熵：dy = -onehot / (N p) 时，得到第 11 节的 (p - onehot) / N
    N, C = x.shape
    labels = np.array([0, 2, 4, 1])                      # (N,)
    onehot = np.eye(C)[labels]                           # (N, C)
    p = softmax_forward(x)                               # (N, C)
    dp = -onehot / (N * p)                               # (N, C) loss = -mean(log p_y) 对 p 的梯度
    assert np.allclose(softmax_backward(dp, p), (p - onehot) / N)
    print("all tests passed")


if __name__ == "__main__":
    run_softmax_tests()
```

第 3 步的代数：$\sum_i dp_i\,p_i=-1/N$，所以 $dx=p\odot(dp+1/N)=(p-\text{onehot})/N$，和第 11 节的结论一致。和 torch 对拍时，两个 torch 版本（1.13 和 2.8）的最大绝对误差都不到 $10^{-15}$，只剩 float64 的舍入误差。

### 关键追问

- **backward 为什么只缓存输出 y？** Jacobian $\operatorname{diag}(y)-yy^\top$ 只依赖 $y$，有了 $y$ 就不用重算 exp。PyTorch 的 softmax backward 也是用前向输出算的。
- **为什么每行的 dx 加起来等于 0？** 每行 logits 同时加一个常数，softmax 不变，所以梯度在「全 1 方向」上的分量为 0。代数上 $J\mathbf{1}=y-y(y^\top\mathbf{1})=0$。这也是梯度检查之外一个便宜的 sanity check。
- **softmax 饱和时梯度怎样？** $y$ 接近 one-hot 时 $J$ 的每一项都接近 0，梯度几乎消失。例如 logits 为 (20, 0, 0)、上游梯度为 (1, -1, 0.5) 时，`softmax_backward` 输出的最大绝对值只有 5.15e-09。注意力里要除以 $\sqrt{d_k}$，原因之一就是防止点积太大把 softmax 推到饱和区。
- **注意力的 backward 怎么用这个公式？** $P=\operatorname{softmax}(S)$，$O=PV$。先得到 $dP=dO\,V^\top$，形状 $(B,H,T,T)$，再套公式 $dS=P\odot(dP-\text{rowsum}(dP\odot P))$。被 mask 的位置 $P=0$，梯度自动是 0。FlashAttention 不存 $P$，它利用 $\text{rowsum}(dP\odot P)=\text{rowsum}(dO\odot O)$，只用 $(B,H,T,d)$ 大小的张量就能算出这一项。
- **分类为什么用 log-softmax 配 NLL，不用 softmax 配 MSE？** softmax 配 MSE 时，模型错得很自信（$y$ 是错误类别的 one-hot）也会因为 $J\approx0$ 拿不到梯度，学不动。log-softmax 的 Jacobian 是 $I-\mathbf{1}y^\top$，饱和时不会消失：同样的 logits (20, 0, 0) 和上游梯度 (1, -1, 0.5)，`log_softmax_backward` 输出的最大绝对值约为 1.0。数值上 log-softmax 也更稳，前向不会出现 $\log 0$（第 6 节）。

---

## 11.3 Dropout：前向与反向

### 原理与直觉

训练时每个激活以概率 $p$ 被随机置 0。神经元不能指望某几个固定的同伴一起出现，就不容易形成只在特定组合下才有用的「共适应」（co-adaptation）。也可以把它看成每一步训练一个随机子网络，推理时用完整网络近似这些子网络的平均。

**p 的含义要先问清楚。** PyTorch 的 `nn.Dropout(p=0.5)` 里 $p$ 是**丢弃**概率，本节也用这个约定。原论文（Srivastava et al., 2014）和 TensorFlow 1.x 的 `tf.nn.dropout(x, keep_prob)` 用的是**保留**概率，两者互为 $1-p$。

PyTorch 的做法叫 **inverted dropout**：训练时把留下来的激活放大 $1/(1-p)$，推理时什么都不做。

$$
m\sim\operatorname{Bernoulli}(1-p),\qquad
\text{out}_{\text{train}}=\frac{m\odot x}{1-p},\qquad
\text{out}_{\text{eval}}=x
$$

放大的原因是保持期望不变：$\operatorname{E}[\text{out}_i]=(1-p)\cdot x_i/(1-p)+p\cdot0=x_i$。训练和推理时下一层看到的激活量级一致，推理代码不用改，直接把 dropout 当成恒等。原论文的版本反过来：训练时只乘 mask 不放大，推理时把输出（等价于权重）乘 $1-p$。两种写法期望相同，但原版的推理代码必须知道 $p$，所以主流框架都用 inverted 版本。

反向：前向乘了 $m/(1-p)$，反向就乘同一个 $m/(1-p)$，即 $dx=dout\odot m/(1-p)$。被丢掉的位置梯度为 0。关键是 backward 必须用**前向那一次**的 mask，所以要缓存它。

### 先写核心版

```python
from typing import Optional, Tuple

import numpy as np


def dropout_forward(x: np.ndarray, p: float, training: bool,
                    rng: np.random.Generator) -> Tuple[np.ndarray, Optional[tuple]]:
    """x: (N, D)；p: 丢弃概率；rng: np.random.default_rng(seed)。返回 out (N, D) 和 cache"""
    if not training:
        return x, None                               # 推理：恒等
    mask = rng.random(x.shape) >= p                  # 1. (N, D) bool，每个位置以 1-p 的概率为 True
    out = x * mask / (1 - p)                         # 2. (N, D) 丢掉的置 0，留下的放大 1/(1-p)
    return out, (mask, p)                            # 3. backward 必须用同一个 mask


def dropout_backward(dout: np.ndarray, cache: Optional[tuple]) -> np.ndarray:
    if cache is None:                                # 推理模式的前向是恒等
        return dout
    mask, p = cache
    return dout * mask / (1 - p)                     # (N, D) 前向乘了什么，反向就乘什么
```

`rng.random` 在 $[0,1)$ 上均匀分布，所以 `>= p` 为 True 的概率正好是 $1-p$。这个版本在 $p=1$ 时会除以 0，面试版处理掉。

### 面试版实现

写成和 11.1 节 `BatchNorm1d` 一样的类：`forward(x, training)` 把缩放后的 mask 存在 `self.mask` 里，`backward(dout)` 直接用。

```python
from typing import Optional

import numpy as np


class Dropout:
    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        """p: 丢弃概率（PyTorch 约定），0 <= p <= 1"""
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.mask = None                                  # 缩放后的 mask，backward 用

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """x: 任意形状，例如 (N, D)。返回同形状"""
        if not training or self.p == 0.0:
            self.mask = None                              # 推理模式或 p = 0：恒等
            return x
        if self.p == 1.0:
            self.mask = np.zeros_like(x)                  # 全部丢弃：直接给 0，不做除法
        else:
            keep = self.rng.random(x.shape) >= self.p     # 1. (N, D) bool
            # 2. (N, D) 把 1/(1-p) 并进 mask，值只有 0 和 1/(1-p)；astype 保持 x 的 dtype
            self.mask = keep.astype(x.dtype) / (1.0 - self.p)
        return x * self.mask                              # 3. (N, D)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: 和 forward 的输出同形状。返回 dx"""
        if self.mask is None:
            return dout                                   # 前向是恒等，梯度原样传回
        return dout * self.mask                           # (N, D)
```

### 自测

用上面的 `Dropout` 和 11.2 节的 `numerical_grad`、`rel_error`。梯度检查时 mask 必须固定，否则每次前向丢的位置不同，loss 就没法看成 $x$ 的确定函数。办法是每次都用**同一个种子**新建层，mask 就和第一次完全一样。和 PyTorch 只能比统计性质和缩放方式，两边的随机数生成器不同，mask 不会逐位相同。

```python
import numpy as np
import torch
import torch.nn.functional as F


def run_dropout_tests() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 20)) + 1.0             # (N, D)，均值约 1

    # 1. 推理模式是恒等
    drop = Dropout(p=0.3, seed=0)
    assert np.array_equal(drop.forward(x, training=False), x)

    # 2. 训练模式：约 30% 置 0；1000 次前向取平均，回到 x
    print(f"zero fraction {(drop.forward(x) == 0).mean():.4f}")      # zero fraction 0.3020
    avg = sum(drop.forward(x) for _ in range(1000)) / 1000           # (N, D)
    print(f"mean: input {x.mean():.4f}, output {avg.mean():.4f}")    # mean: input 0.9720, output 0.9724
    assert abs(avg.mean() - x.mean()) < 0.01
    z = rng.standard_normal(1_000_000)                   # 标准正态，方差约 1
    z_out = Dropout(p=0.5, seed=0).forward(z)
    print(f"var: input {z.var():.4f}, output {z_out.var():.4f}")     # var: input 1.0014, output 2.0017

    # 3. 梯度检查：每次用同一个种子新建层，mask 固定
    xs, ds = rng.standard_normal((4, 5)), rng.standard_normal((4, 5))
    layer = Dropout(p=0.4, seed=1)
    layer.forward(xs)
    dx = layer.backward(ds)
    err = rel_error(dx, numerical_grad(lambda: np.sum(Dropout(p=0.4, seed=1).forward(xs) * ds), xs))
    print(f"grad check {err:.1e}")                                   # grad check 4.0e-11
    assert err < 1e-7

    # 4. 和 PyTorch 比缩放方式与 train/eval 行为（float64）
    torch.manual_seed(0)
    xt = torch.tensor(x, requires_grad=True)             # (N, D)
    out = F.dropout(xt, p=0.3, training=True)
    kept = out.detach() != 0                             # (N, D) bool，留下的位置
    print(f"torch kept {kept.double().mean().item():.4f}")         # torch kept 0.7125
    assert torch.allclose(out[kept], xt[kept] / 0.7)     # 留下的正好是 x / (1 - p)
    out.backward(torch.ones_like(out))
    assert torch.allclose(xt.grad, kept.double() / 0.7)  # 反向：dout * mask / (1 - p)
    assert torch.equal(F.dropout(xt, p=0.3, training=False).detach(), xt.detach())

    # 5. 边界：p = 0 恒等，p = 1 全 0 且没有 nan，p 越界报错
    assert np.array_equal(Dropout(p=0.0).forward(x), x)
    all_drop = Dropout(p=1.0)
    assert not all_drop.forward(x).any() and not all_drop.backward(np.ones_like(x)).any()
    try:
        Dropout(p=1.5)
        raise AssertionError("p = 1.5 should raise")
    except ValueError:
        pass
    print("all tests passed")


if __name__ == "__main__":
    run_dropout_tests()
```

PyTorch 那一边留下的比例是 0.7125，和 $1-p=0.7$ 吻合。第 2 步还有一个值得记住的数：标准正态输入、$p=0.5$ 时，输出方差从 1.0014 变成 2.0017。期望不变，方差却变大了，这一点和下面 BatchNorm 的问题有关。

### 关键追问

- **为什么推理时不用 dropout？** 推理需要确定的输出，而且我们想要的是所有子网络的平均预测，用完整网络近似这个平均即可。inverted dropout 在训练时已经做了缩放，推理时 dropout 就是恒等，可以直接跳过。
- **dropout 放在哪里？** MLP 里放在激活函数之后：Linear → ReLU → Dropout → Linear，输出层的 logits 一般不加。Transformer 原论文（$P_{drop}=0.1$）放在每个子层的输出上、加残差之前（residual dropout），以及 embedding 加位置编码之后；常见实现还会在 attention 权重 softmax 之后加一个（`nn.MultiheadAttention` 的 `dropout` 参数），FFN 的中间激活之后也常加。不少大模型预训练直接把 dropout 设成 0。
- **dropout 和 BatchNorm 一起用有什么问题？** inverted dropout 保住了均值，但训练时的方差变大（上面实测约 2 倍）。BN 的 `running_var` 是在开着 dropout 的训练阶段统计的，推理时 dropout 关掉，方差变了，BN 用的统计量就对不上（Li et al., 2019 称为 variance shift）。常见做法是 dropout 只放在最后一个 BN 之后，或者有 BN 的网络干脆不用 dropout。
- **MC dropout 是什么？** 推理时故意开着 dropout，同一个输入跑 $T$ 次，均值作预测，方差作模型不确定性的估计（Gal & Ghahramani, 2016）。
- **忘了 `model.eval()` 会怎样？** dropout 继续随机丢，同一个输入每次输出不同，指标偏低还会抖动；BN 也会改用当前 batch 的统计量。`torch.no_grad()` 只关梯度、不关 dropout，两者要一起用。反过来，验证完忘了切回 `model.train()`，后面的训练就没有 dropout 了。
- **backward 为什么不重新采样 mask？** 前向丢掉的神经元对 loss 没有贡献，它的梯度必须是 0；重新采样等于对另一个网络求导。所以 mask 要缓存；也可以像 FlashAttention 那样只存随机数种子，backward 时用同一个种子重新生成同一个 mask，省下存 mask 的显存。

---

## 11.4 手写 LSTM 与参数量

LSTM 在 coding 面试里有两种考法：手写前向（NumPy 或 PyTorch），以及单独一道「给定结构，算参数量」。门控的理论、梯度消失的来龙去脉、GRU 的公式见 [05. NLP、RNN 与词向量](05-nlp-rnn.md)，这里只讲怎么写代码和怎么数参数。

### 原理与直觉

LSTM 每个时间步有四个部分，都由当前输入 $x_t$ 和上一步的 $h_{t-1}$ 算出来：

- **输入门 i**：新内容写进记忆多少，取值 0 到 1。
- **遗忘门 f**：旧记忆留下多少，取值 0 到 1。
- **候选内容 g**：这一步想写进去的新内容，取值 -1 到 1（就是 05 页里的 $\tilde c_t$）。
- **输出门 o**：从记忆里读出多少交给 $h_t$，取值 0 到 1。

两个状态的分工：**cell state $c$** 是一条传送带，每步只做「按 f 擦掉一部分、按 i 加上一部分」，更新是加法，信息和梯度都能沿它走很远；**hidden state $h$** 是这一步对外的输出，下一层、分类头、下一个时间步读的都是它。

记号：$x_t:(B,D)$，$h_{t-1},c_{t-1}:(B,H)$。四个门的权重按列拼成一个大矩阵，一次矩阵乘法算完：

$$
a_t=x_tW_x+h_{t-1}W_h+b=[x_t,\ h_{t-1}]\,W+b,\qquad W_x:(D,4H),\quad W_h:(H,4H),\quad W:(D+H,4H),\quad a_t:(B,4H)
$$

$W$ 就是 $W_x$ 和 $W_h$ 上下叠起来。把 $a_t$ 按列切成四块，每块 $(B,H)$，顺序 i、f、g、o（和 PyTorch 一致）：

$$
i=\sigma(a_t^{(1)}),\qquad f=\sigma(a_t^{(2)}),\qquad g=\tanh(a_t^{(3)}),\qquad o=\sigma(a_t^{(4)})
$$

$$
c_t=f\odot c_{t-1}+i\odot g,\qquad h_t=o\odot\tanh(c_t)
$$

三个门用 sigmoid，因为它们是 0 到 1 的「开关比例」；g 用 tanh，因为写入的内容要能正能负。

### 先写核心版（NumPy 前向）

序列用 `(T, B, D)` 布局，时间维在前。这也是 `nn.LSTM` 的默认布局；传 `batch_first=True` 时才是 `(B, T, D)`。

```python
from typing import Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def lstm_cell_forward(x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                      W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """一个时间步。x_t: (B, D)；h_prev, c_prev: (B, H)；W: (D + H, 4H)；b: (4H,)"""
    H = h_prev.shape[1]
    # 1. 一次矩阵乘法同时算四个门的预激活
    a = np.concatenate([x_t, h_prev], axis=1) @ W + b   # (B, D + H) @ (D + H, 4H) -> (B, 4H)
    # 2. 按列切成四块，顺序 i, f, g, o
    i = sigmoid(a[:, :H])                # (B, H) 输入门
    f = sigmoid(a[:, H:2 * H])           # (B, H) 遗忘门
    g = np.tanh(a[:, 2 * H:3 * H])       # (B, H) 候选内容
    o = sigmoid(a[:, 3 * H:])            # (B, H) 输出门
    # 3. 传送带：擦掉一部分旧的，加上一部分新的
    c = f * c_prev + i * g               # (B, H)
    h = o * np.tanh(c)                   # (B, H)
    return h, c


def lstm_forward(x: np.ndarray, h0: np.ndarray, c0: np.ndarray, W: np.ndarray,
                 b: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """整条序列。x: (T, B, D)；h0, c0: (B, H)。返回所有 h (T, B, H) 和最后的 (h, c)"""
    h, c = h0, c0
    hs = []
    for t in range(x.shape[0]):          # 只能按时间串行：h_t 依赖 h_{t-1}
        h, c = lstm_cell_forward(x[t], h, c, W, b)   # x[t]: (B, D)
        hs.append(h)
    return np.stack(hs), (h, c)          # (T, B, H)，(B, H)，(B, H)
```

跑一个小例子，再和 `nn.LSTM` 对拍。PyTorch 把权重存成 `weight_ih_l0: (4H, D)` 和 `weight_hh_l0: (4H, H)`，沿列拼起来再转置就是我们的 $W:(D+H,4H)$；它有两个 bias，相加就等于我们的 $b$。用 float64 比，误差在浮点精度以内：

```python
import torch
import torch.nn as nn

rng = np.random.default_rng(0)
T, B, D, H = 5, 2, 3, 4
x = rng.standard_normal((T, B, D))                       # (T, B, D)
h0, c0 = np.zeros((B, H)), np.zeros((B, H))              # (B, H)

torch.manual_seed(0)
ref = nn.LSTM(D, H).double()                             # 默认 batch_first=False，吃 (T, B, D)
with torch.no_grad():
    W = torch.cat([ref.weight_ih_l0, ref.weight_hh_l0], dim=1).T.numpy()   # (4H, D + H) -> (D + H, 4H)
    b = (ref.bias_ih_l0 + ref.bias_hh_l0).numpy()                         # (4H,)
    out_ref, (h_n, c_n) = ref(torch.from_numpy(x))       # 不传状态时 h0 = c0 = 0

hs, (h_T, c_T) = lstm_forward(x, h0, c0, W, b)
print(hs.shape, h_T.shape)                               # (5, 2, 4) (2, 4)
print(np.allclose(hs, out_ref.numpy()), np.allclose(h_T, h_n[0].numpy()),
      np.allclose(c_T, c_n[0].numpy()))                  # True True True
```

`h_n` 的形状是 `(num_layers * num_directions, B, H)`，单层单向时第一维是 1，所以取 `h_n[0]`。

### 面试版实现（PyTorch nn.Module）

面试官说「用 PyTorch 写一个 LSTM，不许用 `nn.LSTM`」，就写下面这个。两个 `nn.Linear` 各自把四个门拼在一起：`x2h` 对应 `weight_ih_l0` 和 `bias_ih_l0`，`h2h` 对应 `weight_hh_l0` 和 `bias_hh_l0`。`nn.Linear.weight` 的形状本来就是 `(out, in)`，和 PyTorch 的布局一模一样，对拍时直接拷贝，不用转置。

```python
from typing import Optional, Tuple

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """单层单向 LSTM，batch_first=True。参数布局和 nn.LSTM 一致，门的顺序 i, f, g, o。"""

    def __init__(self, input_size: int, hidden_size: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)    # weight (4H, D)，bias (4H,)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)   # weight (4H, H)，bias (4H,)

    def forward(self, x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, D)
            state: 初始 (h0, c0)，各 (B, H)；None 表示全 0
        Returns:
            output: (B, T, H)，每一步的 h
            (h_n, c_n): 最后一步的状态，各 (B, H)
        """
        batch_size, seq_len, _ = x.size()
        if state is None:
            zeros = x.new_zeros(batch_size, self.hidden_size)
            state = (zeros, zeros)
        h, c = state                                          # (B, H), (B, H)
        # 1. 输入投影和 h 无关，整条序列一次算完
        x_proj = self.x2h(x)                                  # (B, T, 4H)
        outputs = []
        for t in range(seq_len):
            # 2. 只有 h2h 这一项必须按时间一步一步算
            gates = x_proj[:, t] + self.h2h(h)                # (B, 4H)
            i, f, g, o = gates.chunk(4, dim=1)                # 各 (B, H)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            # 3. 更新 cell 和 hidden
            c = f * c + i * g                                 # (B, H)
            h = o * torch.tanh(c)                             # (B, H)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)            # (B, T, H)，(B, H)，(B, H)


def test_lstm_matches_pytorch() -> None:
    torch.manual_seed(42)
    batch_size, seq_len, input_size, hidden_size = 3, 7, 5, 8
    model = LSTM(input_size, hidden_size)
    ref = nn.LSTM(input_size, hidden_size, batch_first=True)
    # 1. 拷贝权重：形状一一对应
    with torch.no_grad():
        ref.weight_ih_l0.copy_(model.x2h.weight)              # (4H, D)
        ref.weight_hh_l0.copy_(model.h2h.weight)              # (4H, H)
        ref.bias_ih_l0.copy_(model.x2h.bias)                  # (4H,)
        ref.bias_hh_l0.copy_(model.h2h.bias)                  # (4H,)
    x = torch.randn(batch_size, seq_len, input_size)          # (B, T, D)
    h0 = torch.randn(batch_size, hidden_size)                 # (B, H)
    c0 = torch.randn(batch_size, hidden_size)                 # (B, H)
    # 2. 前向对拍：nn.LSTM 的状态多一维 num_layers * num_directions = 1
    out, (h_n, c_n) = model(x, (h0, c0))
    ref_out, (ref_h, ref_c) = ref(x, (h0.unsqueeze(0), c0.unsqueeze(0)))
    assert out.shape == (batch_size, seq_len, hidden_size)
    assert torch.allclose(out, ref_out, atol=1e-6)
    assert torch.allclose(h_n, ref_h[0], atol=1e-6) and torch.allclose(c_n, ref_c[0], atol=1e-6)
    assert torch.allclose(model(x)[0], ref(x)[0], atol=1e-6)  # 不传状态：两边都从 0 开始
    # 3. 反向对拍：梯度由 autograd 沿时间反传（BPTT），也要一致
    (out.sum() + c_n.sum()).backward()
    (ref_out.sum() + ref_c.sum()).backward()
    assert torch.allclose(model.x2h.weight.grad, ref.weight_ih_l0.grad, atol=1e-5)
    assert torch.allclose(model.h2h.weight.grad, ref.weight_hh_l0.grad, atol=1e-5)
    # 4. 参数量一致：4 * (H * (D + H) + 2H)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == sum(p.numel() for p in ref.parameters())
    assert n_params == 4 * (hidden_size * (input_size + hidden_size) + 2 * hidden_size)
    print("all tests passed")


if __name__ == "__main__":
    test_lstm_matches_pytorch()   # all tests passed
```

反向不用手写。autograd 记下了循环里每一步的计算图，`backward()` 时沿时间从后往前把梯度传回去，这就是 BPTT（Backpropagation Through Time）。同一个 `h2h.weight` 在每一步都被用到，它的梯度是所有时间步贡献的和，和第 11 节「前向广播，反向求和」是同一个道理。

### 参数量

这道题经常单独出。每层每个方向有 k 组变换（LSTM 的 k = 4），每组都是一个从 $[x_t,h_{t-1}]$ 到 $H$ 维的线性层：权重 $H(D_{in}+H)$ 个，偏置 $H$ 个。

$$
P_{\text{教科书}}=4\,\big(H(D_{in}+H)+H\big),\qquad P_{\text{PyTorch}}=4\,\big(H(D_{in}+H)+2H\big)
$$

PyTorch 多出的 $4H$ 来自两组 bias（`b_ih` 和 `b_hh`），两者在前向里直接相加，数学上是冗余的，但确实占参数。面试时先问清是哪种约定，没说就两个都报。

堆叠和双向的规则：

- **第 1 层**：$D_{in}=D$。
- **第 2 层及以后**：输入是上一层的输出，单向时 $D_{in}=H$，双向时两个方向的输出拼起来，$D_{in}=2H$。
- **双向**：每层有正、反两套独立参数，每层乘 2。
- **换模型只换 k**：LSTM 4 组（i、f、g、o），GRU 3 组（r、z、n），vanilla RNN 1 组。`bias=False` 时两组 bias 都去掉。

算例（$D=100$，$H=256$，PyTorch 约定）：

| 模型和结构 | 计算 | 参数量 |
| --- | --- | --- |
| LSTM，1 层单向 | 4 × (256 × 356 + 2 × 256) | 366,592 |
| 同上，只算一个 bias | 4 × (256 × 356 + 256) | 365,568 |
| GRU，1 层单向 | 3 × (256 × 356 + 2 × 256) | 274,944 |
| RNN，1 层单向 | 1 × (256 × 356 + 2 × 256) | 91,648 |
| LSTM，2 层双向 | 第 1 层 2 × 366,592 = 733,184；第 2 层 2 × 4 × (256 × 768 + 512) = 1,576,960 | 2,310,144 |

2 层双向里第 2 层的 768 = 2H + H：输入是 512 维的拼接结果。`nn.LSTM(100, 256, num_layers=2, bidirectional=True)` 的 `weight_ih_l1` 形状正是 `(1024, 512)`。

写成 CodeSignal 单函数题的样子：入口叫 `solution`，return 答案，通用的计数逻辑放在顶层辅助函数里，不 import 任何库。

```python
def rnn_param_count(input_size: int, hidden_size: int, num_layers: int = 1,
                    bidirectional: bool = False, num_gates: int = 4,
                    num_biases: int = 2) -> int:
    """num_gates: LSTM 4、GRU 3、RNN 1；num_biases: PyTorch 是 2，教科书写法是 1，bias=False 是 0"""
    num_directions = 2 if bidirectional else 1
    total = 0
    for layer in range(num_layers):
        # 第 1 层吃原始输入；之后吃上一层的输出，双向时是两个方向拼起来的 2H
        layer_input = input_size if layer == 0 else hidden_size * num_directions
        weights = num_gates * hidden_size * (layer_input + hidden_size)    # W_ih 和 W_hh
        biases = num_gates * hidden_size * num_biases                      # b_ih 和 b_hh
        total += num_directions * (weights + biases)
    return total


def solution(input_size: int, hidden_size: int, num_layers: int, bidirectional: bool) -> int:
    """nn.LSTM 的参数量（PyTorch 约定，两组 bias）"""
    return rnn_param_count(input_size, hidden_size, num_layers, bidirectional)
```

用 `sum(p.numel() for p in module.parameters())` 对拍 `nn.LSTM`、`nn.GRU`、`nn.RNN`，多层、双向、`bias=False` 都覆盖到：

```python
import torch.nn as nn

configs = [(100, 256, 1, False), (100, 256, 2, True), (10, 20, 3, False), (7, 5, 3, True)]
for module_cls, num_gates in [(nn.LSTM, 4), (nn.GRU, 3), (nn.RNN, 1)]:
    for D, H, L, bi in configs:
        for use_bias in (True, False):
            module = module_cls(D, H, num_layers=L, bidirectional=bi, bias=use_bias)
            expected = sum(p.numel() for p in module.parameters())
            assert rnn_param_count(D, H, L, bi, num_gates, 2 if use_bias else 0) == expected

print(solution(100, 256, 1, False))                   # 366592
print(solution(100, 256, 2, True))                    # 2310144
print(rnn_param_count(100, 256, num_biases=1))        # 365568
print(rnn_param_count(100, 256, num_gates=3), rnn_param_count(100, 256, num_gates=1))   # 274944 91648
```

### 关键追问

- **LSTM 为什么能缓解梯度消失？** 沿 cell 这条路，$c_t=f\odot c_{t-1}+i\odot g$ 是加法，只看这条路时 $\partial c_t/\partial c_{t-1}=f$，梯度每往回走一步乘一个 f，不像 vanilla RNN 那样每步乘一次 $W_h$ 和 tanh 的导数。f 接近 1 时梯度几乎无损地传回很早的时间步。这只是缓解：序列很长时 f 的连乘照样会衰减。完整推导见 [05. NLP、RNN 与词向量](05-nlp-rnn.md)。
- **为什么常把遗忘门的 bias 初始化成 1？** 权重很小时 $f\approx\sigma(b_f)$。$b_f=0$ 时 f 约为 0.5，沿 cell 往回 20 步，梯度只剩 $0.5^{20}\approx9.5\times10^{-7}$；$b_f=1$ 时 $\sigma(1)\approx0.731$，20 步后约 $1.9\times10^{-3}$，大了约 2000 倍。模型一开始就倾向「记住」（Jozefowicz et al., 2015）。PyTorch 里遗忘门是 bias 的第二块 `[H:2H]`，两个 bias 相加才是有效 bias，所以只改其中一个，比如 `bias_ih_l0.data[H:2 * H].fill_(1.0)`、`bias_hh_l0.data[H:2 * H].zero_()`。
- **LSTM 和 GRU 怎么选？** GRU 只有 3 组变换，没有单独的 c，参数是 LSTM 的 3/4，多数任务效果接近，数据少或想要快时先试 GRU。手写 GRU 去对拍 `nn.GRU` 时注意两点：PyTorch 的候选状态是 $n=\tanh(W_{in}x+b_{in}+r\odot(W_{hn}h+b_{hn}))$，r 乘在矩阵乘法之后；更新写成 $h_t=(1-z)\odot n+z\odot h_{t-1}$，z 的含义和 05 页的公式正好相反。
- **计算量多大，为什么不能沿时间并行？** 矩阵乘法部分每步每个样本做 $4H(D+H)$ 次乘加，和权重个数相同；$D=100$，$H=256$ 时是 364,544 次，长度 T 的序列再乘 T。$h_t$ 依赖 $h_{t-1}$，T 步必须串行，GPU 再多核也只能并行 batch 和 H 这两维。面试版里把 `x2h` 提到循环外，就是把能并行的部分（输入投影）一次做完。Attention 的每个位置直接看所有位置，T 步一次矩阵乘法算完，代价是 $O(T^2)$ 的计算和显存。
- **h 和 c 分别拿来干什么？** h 是对外的输出：`output` 就是每一步的 h，给下一层或分类头用；c 是内部记忆，只在步与步之间传递。做序列分类时取最后一层的 `h_n[-1]`，单层单向时它等于 `output[:, -1]`。双向时 `output[:, -1]` 里只有正向的最终状态，反向的最终状态在 `output[:, 0]` 的后半段，所以通常拼 `h_n[-2]` 和 `h_n[-1]`。
- **batch 里序列长短不一怎么办？** 补齐 padding 后用 `pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)` 再喂给 `nn.LSTM`，每条序列只跑到自己的真实长度，`h_n` 就是最后一个真实 token 处的状态；不 pack 的话 `h_n` 会把 padding 也算进去。
- **序列很长时怎么训练？** 截断 BPTT：每 k 步做一次反向，把状态 `h.detach()`、`c.detach()` 之后传给下一段，梯度只回传 k 步，显存从和 T 成正比降到和 k 成正比。梯度爆炸用 `torch.nn.utils.clip_grad_norm_` 裁剪。

---

## 12. Top-K 问题（堆）

这一节是算法题：从 $n$ 个元素里找最大的 $k$ 个。它和 2.4 节解码时的 Top-K 采样是两个不同的问题。

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
- **数据全在内存里，能更快吗？** 能。快速选择平均 $O(n)$，NumPy 的 `np.argpartition` 就是这么做的（2.4 节和第 9 节 KNN 都用了它）。
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

## 14. 手写 N-gram 语言模型

N-gram 的理论（链式法则、Markov 假设、平滑方法的名字）在 [05. NLP、RNN 与词向量](05-nlp-rnn.md) 第 7、8 节，这里只讲怎么写。这道题考的是三件事：分子分母数对，`<s>` 和 `</s>` 补对，概率连乘改成 log 相加。

### 原理与直觉

语言模型给一句话打分。链式法则把句子概率拆成每个词在前文下的条件概率之积，再做 Markov 假设：第 $t$ 个词只看前 $n-1$ 个词。

$$
P(w_1,\dots,w_m)\approx\prod_{t=1}^{m+1}P(w_t\mid w_{t-n+1},\dots,w_{t-1})
$$

条件概率直接数出来（最大似然估计）。记上下文 $h=(w_{t-n+1},\dots,w_{t-1})$，$c(\cdot)$ 是训练集里的出现次数：

$$
P(w\mid h)=\frac{c(h,w)}{c(h)}
$$

分母 $c(h)$ 是 $h$ **作为上下文**出现的次数，也就是 $\sum_w c(h,w)$。

两个边界 token：

- 句首补 $n-1$ 个 `<s>`，第一个词也有完整的上下文。`<s>` 只当上下文，永远不需要被预测。
- 句尾补一个 `</s>`，它和普通词一样要被预测，所以上面连乘到 $m+1$。没有 `</s>` 时，所有长度为 1 的句子概率和是 1，长度为 2 的也是 1，加起来不是一个合法分布；有了它，模型还能学到句子在哪里结束。

几百个小于 1 的数连乘就会下溢（`0.01 ** 400` 在 Python 里就是 `0.0`），所以句子概率一律用 log 相加。Perplexity（PPL）是平均每个 token 的负 log 概率再取指数，$N$ 是被预测的 token 数（每句的词数加 1 个 `</s>`）：

$$
\log P(\text{句子})=\sum_{t=1}^{m+1}\log P(w_t\mid h_t),\qquad \text{PPL}=\exp\Big(-\frac{1}{N}\sum_t\log P(w_t\mid h_t)\Big)
$$

### 先写核心版

先写 bigram（$n=2$）、不平滑的版本，十几行就能证明你懂计数。题目如果不让 import，把 `Counter` 换成 `d[key] = d.get(key, 0) + 1` 即可。

```python
import math
from collections import Counter
from typing import Dict, List, Tuple


def train_bigram(sentences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """MLE bigram：P(w | prev) = c(prev, w) / c(prev)，不做平滑"""
    pair_counts, prev_counts = Counter(), Counter()
    for sent in sentences:
        tokens = ["<s>"] + sent + ["</s>"]                  # 1. 补句首、句尾
        for prev, word in zip(tokens, tokens[1:]):          # 2. 遍历所有相邻对
            pair_counts[(prev, word)] += 1
            prev_counts[prev] += 1                          # 3. prev 作为上下文的次数
    return {pair: c / prev_counts[pair[0]] for pair, c in pair_counts.items()}


corpus = [["i", "like", "nlp"], ["i", "like", "deep", "learning"], ["you", "like", "nlp"]]
probs = train_bigram(corpus)
print(probs[("<s>", "i")], probs[("like", "nlp")], probs[("nlp", "</s>")])   # 0.666... 0.666... 1.0

tokens = ["<s>", "i", "like", "nlp", "</s>"]                # 4 个 bigram，含 </s>
print(sum(math.log(probs[(a, b)]) for a, b in zip(tokens, tokens[1:])))       # -0.8109，即 2 log(2/3)
print(("like", "learning") in probs)                        # False：没见过，MLE 概率为 0
```

最后一行就是 MLE 的问题：测试句里只要有一个没见过的 bigram（比如「i like learning」），整句概率就是 0，log 是负无穷。面试版用 add-k 平滑解决。

### 面试版实现

在核心版上加四样东西：任意 $n$、add-k 平滑、`<unk>`、评估和生成。平滑公式：

$$
P_{\text{add-}k}(w\mid h)=\frac{c(h,w)+k}{c(h)+kV}
$$

$V$ 必须等于**所有可能被预测的 token 数**，否则对 $w$ 求和不等于 1。这里 $V$ = 训练集里的词 + `</s>` + `<unk>`；`<s>` 不算，因为它从不被预测。测试时没见过的词一律映射成 `<unk>`。没见过的上下文 $c(h)=0$，公式自动给出均匀分布 $1/V$。

```python
import math
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NGramLM:
    def __init__(self, n: int = 2, k: float = 1.0, min_count: int = 1):
        """
        Args:
            n: 阶数，n=2 是 bigram
            k: add-k 平滑，k=1 是 Laplace，k=0 退化成 MLE
            min_count: 训练集里出现次数少于它的词也当成 <unk>
        """
        assert n >= 1 and k >= 0
        self.n, self.k, self.min_count = n, k, min_count
        self.vocab: Set[str] = set()                         # 可被预测的 token：训练词 + </s> + <unk>
        self.ngram_counts: Counter = Counter()               # key (h, w) -> c(h, w)，h 是 n-1 元 tuple
        self.context_counts: Counter = Counter()             # key h -> c(h)；没见过的 key 返回 0

    def _map(self, word: str) -> str:
        """OOV 词映射成 <unk>；<s> 只会出现在上下文里，原样保留"""
        return word if word in self.vocab or word == BOS else UNK

    def train(self, sentences: List[List[str]]) -> "NGramLM":
        # 1. 先定词表，计数时 OOV（或低频词）才能映射成 <unk>
        word_counts = Counter(w for sent in sentences for w in sent)
        self.vocab = {w for w, c in word_counts.items() if c >= self.min_count} | {EOS, UNK}
        # 2. 每个要预测的位置（第一个词到 </s>）记一次 c(h, w) 和 c(h)
        self.ngram_counts, self.context_counts = Counter(), Counter()
        for sent in sentences:
            tokens = [BOS] * (self.n - 1) + [self._map(w) for w in sent] + [EOS]
            for t in range(self.n - 1, len(tokens)):
                h = tuple(tokens[t - self.n + 1:t])          # 前 n-1 个 token
                self.ngram_counts[(h, tokens[t])] += 1
                self.context_counts[h] += 1
        return self

    def prob(self, word: str, context: Sequence[str]) -> float:
        """
        Args:
            word: 要预测的词，OOV 按 <unk> 算
            context: 前文，只用最后 n-1 个词，不够长时左边补 <s>
        Returns:
            (c(h, w) + k) / (c(h) + k * V)
        """
        padded = [BOS] * (self.n - 1) + [self._map(w) for w in context]
        h = tuple(padded[len(padded) - self.n + 1:])         # n=1 时是空 tuple
        numerator = self.ngram_counts[(h, self._map(word))] + self.k
        denominator = self.context_counts[h] + self.k * len(self.vocab)
        return numerator / denominator if denominator > 0 else 0.0   # 只有 k=0 且 h 没见过才是 0

    def sentence_logprob(self, sentence: List[str]) -> float:
        """log P(sentence)：每个位置（含 </s>）的 log 条件概率相加"""
        tokens = [BOS] * (self.n - 1) + sentence + [EOS]
        total = 0.0
        for t in range(self.n - 1, len(tokens)):
            p = self.prob(tokens[t], tokens[t - self.n + 1:t])
            if p == 0.0:                                     # k=0 时遇到没见过的 n-gram
                return -math.inf
            total += math.log(p)
        return total

    def perplexity(self, sentences: List[List[str]]) -> float:
        """exp(-总 log 概率 / N)，N = 每句词数 + 1（</s> 也算，<s> 不算）"""
        total_logprob = sum(self.sentence_logprob(s) for s in sentences)
        num_tokens = sum(len(s) + 1 for s in sentences)
        return math.exp(-total_logprob / num_tokens)

    def next_word_distribution(self, context: Sequence[str]) -> Dict[str, float]:
        return {w: self.prob(w, context) for w in sorted(self.vocab)}

    def predict_next(self, context: Sequence[str]) -> str:
        dist = self.next_word_distribution(context)
        return max(dist, key=dist.get)                       # 平局时取字典序最小的词

    def generate(self, max_len: int = 20) -> List[str]:
        """greedy 生成：每步取概率最大的词，遇到 </s> 停"""
        words: List[str] = []
        while len(words) < max_len:
            next_word = self.predict_next(words)
            if next_word == EOS:
                break
            words.append(next_word)
        return words
```

### Example

用上面的 `corpus` 和 `NGramLM`。训练词有 6 个，$V=6+2=8$。

```python
lm = NGramLM(n=2, k=1.0).train(corpus)
print(len(lm.vocab), lm.prob("nlp", ["like"]), lm.prob("xyz", ["like"]))    # 8 0.2727... 0.0909...

test = [["i", "like", "nlp"], ["i", "like", "learning"]]     # 第二句有没见过的 bigram
print(lm.perplexity(test))                                   # 4.163963996609268
print(NGramLM(n=2, k=0.0).train(corpus).perplexity(test))    # inf：MLE 遇到没见过的 bigram
print(lm.generate(), NGramLM(n=3, k=1.0).train(corpus).generate())
# ['i', 'like', 'nlp'] ['i', 'like', 'deep', 'learning']

low = NGramLM(n=2, k=1.0, min_count=2).train(corpus)         # deep、learning、you 变成 <unk>
print(sorted(low.vocab), low.prob("deep", ["like"]))
# ['</s>', '<unk>', 'i', 'like', 'nlp'] 0.25
```

逐个手算对一下：$P(\text{nlp}\mid\text{like})=(2+1)/(3+8)=3/11$，MLE 是 $2/3$，平滑把它压低了一大截，让出的概率分给了没见过的词。`xyz` 按 `<unk>` 算，计数为 0，得到 $1/11$。trigram 生成时，上下文 (i, like) 后面 nlp 和 deep 各出现 1 次，平局取字典序小的 deep。`min_count=2` 时 like 后面的 deep 变成 `<unk>`，like 后接 `<unk>` 的计数是 1，$V=5$，所以是 $(1+1)/(3+5)=0.25$。

### 自测

用上面的 `corpus`、`train_bigram` 和 `NGramLM`：

```python
import math
from typing import List


def solution(train_sentences: List[List[str]], test_sentences: List[List[str]],
             n: int, k: float) -> float:
    """CodeSignal 单函数题的样子：返回测试集 perplexity，保留 4 位小数"""
    return round(NGramLM(n=n, k=k).train(train_sentences).perplexity(test_sentences), 4)


def test_ngram() -> None:
    lm = NGramLM(n=2, k=1.0).train(corpus)
    # 1. 手算：词表 8 个；P(nlp | like) = 3/11；OOV 按 <unk> 算
    assert len(lm.vocab) == 8
    assert math.isclose(lm.prob("nlp", ["like"]), 3 / 11)
    assert math.isclose(lm.prob("xyz", ["like"]), lm.prob(UNK, ["like"]))
    # 2. 归一化：不同 n、k，任意上下文（包括没见过的），整个词表上的概率和都是 1
    for n in [1, 2, 3]:
        for k in [0.1, 1.0]:
            model = NGramLM(n=n, k=k).train(corpus)
            for context in [[], ["i"], ["i", "like"], ["never", "seen"]]:
                assert math.isclose(sum(model.next_word_distribution(context).values()), 1.0)
    # 3. k=0 退化成 MLE，和核心版 train_bigram 完全一致
    mle = NGramLM(n=2, k=0.0).train(corpus)
    for (prev, word), p in train_bigram(corpus).items():
        assert math.isclose(mle.prob(word, [prev]), p)
    # 4. PPL 手算："i like nlp" 的 4 个概率是 3/11、3/10、3/11、3/10
    expected = (3 / 11 * 3 / 10 * 3 / 11 * 3 / 10) ** (-1 / 4)
    assert math.isclose(lm.perplexity([["i", "like", "nlp"]]), expected)
    assert solution(corpus, [["i", "like", "nlp"]], 2, 1.0) == round(expected, 4)
    print("all tests passed")


if __name__ == "__main__":
    test_ngram()
```

### 关键追问

- **为什么一定要平滑？** MLE 给没见过的 n-gram 概率 0。测试句里只要有一个，整句概率是 0，PPL 是无穷大（上面 k=0 的那一行）。语料再大也躲不开：词频是长尾分布，大量合法的 n-gram 在训练集里一次都没出现过。
- **add-k 有什么问题？还有哪些平滑？** add-k 给每个没见过的词都分 $k$ 份，$V$ 很大时会从见过的词那里抢走太多概率，$k$ 一般取小于 1 并在验证集上调。**Backoff**：高阶 n-gram 没见过就退到低阶（trigram 退到 bigram），Katz backoff 用折扣保证归一化，Google 的 stupid backoff 乘固定的 0.4、不归一化，胜在大规模下简单。**Interpolation**：总是把各阶混合，$\lambda_3P_3+\lambda_2P_2+\lambda_1P_1$，$\lambda$ 在 held-out 数据上调。**Kneser-Ney**：先对每个计数减一个固定折扣，低阶分布用「这个词跟在多少种不同的词后面」代替词频，「Francisco」很常见但几乎只跟在「San」后面，所以它作为新搭配出现的概率应该很低。Modified KN 是 n-gram 模型的事实标准（KenLM 的默认）。
- **n 越大越好吗？** 不一定。可能的 n-gram 有 $V^n$ 种，$n$ 越大，绝大多数组合从没出现过，出现过的也大多只有一两次，估计越来越不可靠；存储量随观测到的不同 n-gram 数增长。实践中常用 3~5 阶，配合 KN 平滑和大语料。
- **Perplexity 怎么理解？** 它是平均负 log 似然（也就是交叉熵）的指数，可以理解成「模型每一步平均在多少个词里犹豫」：对 $V$ 个词均匀猜，PPL 正好是 $V$。比较 PPL 的前提是同一个词表、同样处理 `</s>` 和 `<unk>`：把更多词映射成 `<unk>` 会让 PPL 变低，模型本身并没有变好。
- **`<unk>` 的概率从哪来？** add-k 下 `<unk>` 至少有 $k$ 份。更好的做法是 `min_count`：把训练集里的低频词替换成 `<unk>`，它就有了真实计数，学到「什么位置容易出现生词」。
- **和神经语言模型是什么关系？** 目标完全相同：按链式法则预测下一个 token，评估都用 PPL，神经 LM 训练时最小化的交叉熵就是 log PPL（见 [06. LLM 基础](06-llm-foundations.md)）。区别在怎么估计 $P(w\mid h)$：N-gram 是一张计数表，上下文固定 $n-1$ 个词，每个上下文各算各的；RNN、Transformer 用 embedding 共享参数，「i like nlp」学到的东西能迁移到「we love nlp」，上下文也长得多，所以不需要手工平滑。

---

## 15. 手写 BPE Tokenizer

[06. LLM 基础](06-llm-foundations.md) 第 8 节讲了 BPE 和 WordPiece 的区别，这里手写训练和编码。例子用 Sennrich et al.（2016）论文里的经典小语料。

### 原理与直觉

1. 每个词拆成字符，末尾加一个词尾标记 `</w>`：`low` 变成 `l o w </w>`。有了 `</w>`，词尾的 `est</w>`（newest 的结尾）和词中间的 `est`（estimate 的开头）是两个不同的 token，解码时也靠它还原空格。
2. 统计所有**相邻符号对**的出现次数，按词频加权：`newest` 出现 6 次，它里面的 `(e, s)` 就算 6 次。
3. 把次数最多的 pair 合并成一个新符号，追加到 merges 列表。在新的切分上重复第 2、3 步，直到合并了 `num_merges` 次。
4. 词表 = 初始符号（所有字符 + `</w>`）+ 每次合并产生的新符号，所以**词表大小 = 初始符号数 + 合并次数**（再加上 `<unk>` 这类特殊 token）。实际中先定目标词表大小，再反推合并次数。

为什么用子词：整词词表遇到新词只能输出 `<unk>`，纯字符序列又太长。BPE 训练完，高频词变成一个 token（`low` 变成 `low</w>`），罕见词和新词拆成学过的片段，最坏退回单个字符，所以只要字符见过就不会 OOV。

**平局规则**：最高频的 pair 可能有好几个，结果取决于怎么选。本节取「最先出现的那个」：词按输入顺序、词内从左到右扫描，`Counter` 按插入顺序迭代（Python 3.7+），`max` 遇到平局返回第一个。这和 Sennrich 论文里附的参考代码在 Python 3.7+ 上的行为一致。想让结果和输入顺序无关，可以改成 `min(pairs, key=lambda p: (-pairs[p], p))`，即平局取字典序最小的 pair。面试时先问清楚题目要哪一种。

### 先写核心版

```python
from collections import Counter
from typing import Dict, List, Tuple

EOW = "</w>"
Pair = Tuple[str, str]
Word = Tuple[str, ...]                       # 一个词当前的切分，如 ("l", "o", "w", "</w>")


def get_pair_counts(splits: Dict[str, Word], word_counts: Dict[str, int]) -> Dict[Pair, int]:
    """所有相邻符号对的出现次数，按词频加权"""
    pairs = Counter()
    for word, symbols in splits.items():
        for pair in zip(symbols, symbols[1:]):
            pairs[pair] += word_counts[word]
    return pairs


def merge_pair(symbols: Word, pair: Pair) -> Word:
    """从左到右扫一遍，把所有相邻的 (a, b) 换成 a + b"""
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(word_counts: Dict[str, int], num_merges: int) -> Tuple[List[Pair], Dict[str, Word]]:
    """
    Args:
        word_counts: 词 -> 词频
        num_merges: 合并次数
    Returns:
        merges（按学到的顺序）和每个训练词最后的切分
    """
    splits = {word: tuple(word) + (EOW,) for word in word_counts}       # 1. 拆成字符 + </w>
    merges: List[Pair] = []
    for _ in range(num_merges):
        pairs = get_pair_counts(splits, word_counts)                    # 2. 数 pair
        if not pairs:                                                    # 每个词都只剩一个符号了
            break
        best = max(pairs, key=pairs.get)                                 # 3. 最高频，平局取最先出现的
        splits = {w: merge_pair(s, best) for w, s in splits.items()}    # 4. 所有词里合并它
        merges.append(best)
    return merges, splits


word_counts = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges, splits = train_bpe(word_counts, num_merges=10)
print(merges)
# [('e', 's'), ('es', 't'), ('est', '</w>'), ('l', 'o'), ('lo', 'w'),
#  ('n', 'e'), ('ne', 'w'), ('new', 'est</w>'), ('low', '</w>'), ('w', 'i')]
print(splits)
# {'low': ('low</w>',), 'lower': ('low', 'e', 'r', '</w>'),
#  'newest': ('newest</w>',), 'widest': ('wi', 'd', 'est</w>')}
```

前三步都在合并 `e s t </w>`：它出现在 newest（6 次）和 widest（3 次）里，计数 9，最高。

### 面试版实现

训练直接用上面的 `train_bpe`（还有 `Pair`、`EOW`、`merge_pair`），类里多做三件事：预分词、建词表（token 和 id 互查）、按 rank 编码。

编码用 GPT-2 的循环：把 merges 的下标当成 rank（越小越早学到），每次在当前切分里找 rank 最小的相邻 pair，合并它的所有出现，直到剩下的 pair 都不在 merges 里。这等于把训练过程在这一个词上重放一遍。

```python
import math
import re
from collections import Counter
from typing import Dict, List

UNK_TOKEN = "<unk>"


class BPETokenizer:
    def __init__(self):
        self.merges: List[Pair] = []
        self.ranks: Dict[Pair, int] = {}                  # pair -> rank，越小越早学到
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    @staticmethod
    def pre_tokenize(text: str) -> List[str]:
        """连续的字母数字算一个词，每个标点单独成词，空白丢掉"""
        return re.findall(r"\w+|[^\w\s]", text)

    def train(self, corpus_text: str, num_merges: int) -> "BPETokenizer":
        # 1. 预分词、数词频，再用上面的 train_bpe 学 merges
        word_counts = Counter(self.pre_tokenize(corpus_text))
        self.merges, _ = train_bpe(word_counts, num_merges)
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        # 2. 词表 = <unk> + 初始符号 + 合并出的新符号；dict.fromkeys 去重并保持顺序
        initial = sorted({ch for word in word_counts for ch in word} | {EOW})
        tokens = [UNK_TOKEN] + initial + [a + b for a, b in self.merges]
        self.token_to_id = {tok: i for i, tok in enumerate(dict.fromkeys(tokens))}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        return self

    def encode_word(self, word: str) -> List[str]:
        """反复找 rank 最小的相邻 pair，合并它在词里的所有出现"""
        symbols = tuple(word) + (EOW,)
        while len(symbols) > 1:
            pairs = set(zip(symbols, symbols[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, math.inf))
            if best not in self.ranks:                     # 剩下的 pair 都没学过
                break
            symbols = merge_pair(symbols, best)            # 用上面的 merge_pair
        return list(symbols)

    def encode(self, text: str) -> List[int]:
        unk_id = self.token_to_id[UNK_TOKEN]
        return [self.token_to_id.get(tok, unk_id)          # 训练时没见过的字符变成 <unk>
                for word in self.pre_tokenize(text) for tok in self.encode_word(word)]

    def decode(self, ids: List[int]) -> str:
        text = "".join(self.id_to_token[i] for i in ids)
        return text.replace(EOW, " ").strip()              # </w> 还原成空格
```

### Example

用上面核心版的 `merges` 对一下：

```python
corpus_text = "low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3   # 和 word_counts 等价
tok = BPETokenizer().train(corpus_text, num_merges=10)
print(tok.merges == merges, len(tok.token_to_id))       # True 22

for word in ["lower", "widest", "lowest", "newer", "wider"]:   # 前两个是训练词
    print(word, tok.encode_word(word))
# lower ['low', 'e', 'r', '</w>']
# widest ['wi', 'd', 'est</w>']
# lowest ['low', 'est</w>']
# newer ['new', 'e', 'r', '</w>']
# wider ['wi', 'd', 'e', 'r', '</w>']

ids = tok.encode("lowest newer zoo")
print(ids)                                               # [16, 14, 18, 3, 8, 1, 0, 7, 7, 1]
print(tok.decode(ids))                                   # lowest newer <unk>oo
```

训练词的编码和训练结束时 `splits` 里的切分一模一样。没见过的词拆成学过的片段：`lowest` 由 `low` 和 `est</w>` 拼成。`z` 在训练语料里没出现过，只能变成 `<unk>`。

### 变体：byte-level BPE、WordPiece、Unigram

- **Byte-level BPE（GPT-2）**：初始符号是 256 个字节，文本先按 UTF-8 编码成字节，再跑同样的 BPE，所以任何字符串都能编码，不需要 `<unk>`。GPT-2 的词表 50257 = 256 个字节 + 50000 次合并 + 1 个 `<|endoftext|>`，正好是「初始符号 + 合并次数」。它不用 `</w>`，而是把词前面的空格并进 token（显示成 `Ġ`），decode 能原样还原空白。
- **WordPiece（BERT）**：训练也是逐步合并，但选 pair 的分数是 $c(ab)/\big(c(a)\,c(b)\big)$，两个片段各自都很常见时分数会被压低，更偏向「合在一起才常见」的组合，相当于选让语料似然涨得最多的合并。编码时不重放 merges，按词表从左到右做最长匹配，词中间的片段带 `##` 前缀。
- **Unigram LM（SentencePiece 常用）**：方向相反，先拿一个很大的候选词表，用 EM 估计每个 token 的概率，反复删掉「删了以后语料似然下降最少」的 token，直到词表缩到目标大小。编码用 Viterbi 找概率最大的切分，也可以按概率采样不同切分（subword regularization）。

### 自测

用上面的 `word_counts`、`splits`、`train_bpe` 和 `tok`：

```python
from typing import Dict, List


def solution(word_counts: Dict[str, int], num_merges: int) -> List[str]:
    """CodeSignal 单函数题的样子：返回学到的 merges，每个写成 "a b" """
    learned, _ = train_bpe(word_counts, num_merges)
    return [a + " " + b for a, b in learned]


def test_bpe() -> None:
    # 1. 前三次合并：e-s-t-</w> 计数 9，最高
    assert solution(word_counts, 3) == ["e s", "es t", "est </w>"]
    # 2. 训练词：按 rank 编码 == 训练结束时的切分
    for word, symbols in splits.items():
        assert tok.encode_word(word) == list(symbols)
    # 3. 词表大小 = <unk> + 11 个初始符号（10 个字母 + </w>）+ 10 次合并
    assert len(tok.token_to_id) == 1 + 11 + 10
    # 4. 往返：只含已知字符的文本，decode(encode(x)) == x
    for text in ["low lower", "newest widest", "lowest newer slow"]:
        assert tok.decode(tok.encode(text)) == text
    # 5. 未知字符变成 <unk>，其余照常编码
    assert tok.decode(tok.encode("zoo")) == UNK_TOKEN + "oo"
    # 6. num_merges 给得很大：合并到每个词只剩一个 token 就停
    _, all_splits = train_bpe(word_counts, num_merges=1000)
    assert all(len(s) == 1 for s in all_splits.values())
    print("all tests passed")


if __name__ == "__main__":
    test_bpe()
```

### 关键追问

- **训练的复杂度？真实实现怎么加速？** 朴素版每次合并都把所有词重扫一遍来数 pair。设去重后所有词的符号总数为 $N$、合并次数为 $M$，复杂度是 $O(MN)$，$M$ 动辄几万，太慢。第一步是按词去重（本节已经这么做：同一个词只存一份加词频）。真正的实现做增量更新：维护 pair 计数和「pair 到包含它的词」的倒排索引，合并 $(a,b)$ 时只改含有它的那些词，减掉左右邻居的旧 pair、加上新 pair；最高频 pair 用堆（lazy deletion）找。HF tokenizers 和 SentencePiece 都是这个思路。
- **encode 为什么必须按训练顺序合并？** 后面的 merge 建立在前面的 merge 之上：`(new, est</w>)` 要求 `ne`、`new`、`es`、`est`、`est</w>` 都已经合出来。顺序一乱，会得到训练时从没出现过的切分，模型没见过这样的 token 组合，同一个词在训练和推理时的 id 序列也对不上。按 rank 从小到大合并就是重放训练过程，训练词能复现训练时的切分（自测第 2 条）。
- **为什么用 GPT-2 的循环，不按 merges 逐条扫？** 逐条扫每个词要过一遍全部 $M$ 条规则；GPT-2 的循环只看词里实际存在的 pair，每轮 $O(L)$ 查字典，最多 $L-1$ 轮（$L$ 是词长），和 $M$ 无关。同一个词反复出现，再加一个缓存。
- **词表大小怎么权衡？** 词表大：同样的文本切出的 token 少，上下文窗口能装更多内容，生成步数也少；代价是 embedding 和输出层都是 $V\times d$ 的矩阵，softmax 更贵，罕见 token 训练不足。词表小则反过来，序列变长，注意力的计算量按长度平方增长。LLaMA 2 用 32000，LLaMA 3 扩到约 12.8 万，多语言和代码的压缩率更好。
- **中文怎么处理？** 中文没有空格，按空格预分词会把一整句当成一个词。byte-level 起点（GPT-2 系列）：一个汉字在 UTF-8 里是 3 个字节（「中」是 `e4 b8 ad`），常见字和常用词会被合并成一个 token，罕见字可能占 2~3 个 token。字符级起点（SentencePiece，LLaMA 1/2）：每个汉字是一个初始符号，再往上合并成词；罕见字用 byte fallback 退回字节，避免 `<unk>`。所以同一段中文在不同 tokenizer 下的 token 数可能差很多，直接影响成本和上下文能装多少内容。
- **为什么要先预分词？** 让合并不跨过词边界：否则会学出横跨空格的 `in the`、带标点的 `dog.` 这类 token，同一个词跟着不同标点就成了不同 token，浪费词表。预分词后 BPE 只在词内部跑，同一个词只算一次、可以缓存。GPT-2 用正则把文本切成缩写（`'s`、`'ll`）、字母串、数字串和标点串，空格归到后面的词；本节的 `\w+|[^\w\s]` 是最简版本，代价是 decode 只能还原成用单个空格分隔的词。

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