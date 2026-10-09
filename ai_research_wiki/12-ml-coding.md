# 12. ML Coding 专题

[返回目录](README.md)

本页按专题整理常见手写模型（Transformer 与解码、经典 ML、神经网络层、NLP）。CodeSignal 的题型、代码规范，NumPy、Pandas、PyTorch 的基本语法和纯 Python 矩阵运算在 [11. ML Coding 基础](11-ml-coding-basics.md)，正文里写作「基础篇 0.x 节」。面试时不只要写出能跑的代码，还要主动说明：

- 输入输出 shape。
- 时间/空间复杂度。
- 数值稳定性。
- 边界条件。
- 和框架实现的差异。

---

## 1. 手撸 Transformer（Encoder-Decoder，PyTorch）

原论文（Vaswani et al., 2017）的 Transformer 做翻译，分两半。**Encoder** 双向读完整个源句，输出每个位置的表示 `memory`；**Decoder** 一边用因果 mask 看已经生成的目标前缀，一边用 cross-attention 查 `memory`，预测下一个 token。两半用的是同一套零件：Multi-Head Attention、FFN、残差 + LayerNorm。记 B = batch，C = `d_model`，H = 头数，`d_k = C / H`，V = 目标词表大小，数据流是：

`src (B, T_src)` → Encoder → `memory (B, T_src, C)`；`tgt (B, T_tgt)` + `memory` → Decoder → `logits (B, T_tgt, V)`

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

$M$ 是 mask，屏蔽的位置为 $-\infty$（代码里用 `-1e9`），其余为 0。除以 $\sqrt{d_k}$ 让分数的方差不随维度变大，softmax 才不会饱和（推导见关键追问）。本节 mask 统一约定 1 = 可见、0 = 屏蔽；`nn.MultiheadAttention` 的 bool mask 正好相反（True = 屏蔽），两种约定混用是最常见的 bug。

### 实现

**第 1 步：attention 和多头。** 多头就是把 C 维切成 H 份，每份 `d_k` 维各算一次 attention，再拼回来过 `w_o`。同一个 `MultiHeadAttention` 既做 self-attention（`query = key = value = x`），也做 cross-attention（`query` 来自 decoder，`key = value = memory`），区别只在传参。

```python
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        q: (B, H, T_q, d_k)；k: (B, H, T_k, d_k)；v: (B, H, T_k, d_v)
        mask: 可广播到 (B, H, T_q, T_k)，1 = 可见，0 = 屏蔽
    Returns:
        (B, H, T_q, d_v)
    """
    # 1. 打分并缩放：(B, H, T_q, d_k) @ (B, H, d_k, T_k) -> (B, H, T_q, T_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    # 2. 屏蔽必须在 softmax 之前：填 -1e9，softmax 后权重为 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 3. 沿 T_k 归一化，再加权求和 V
    weights = F.softmax(scores, dim=-1)                    # (B, H, T_q, T_k)
    return torch.matmul(weights, v)                        # (B, H, T_q, d_v)


class MultiHeadAttention(nn.Module):
    """self-attention 传 (x, x, x)；cross-attention 传 (decoder 状态, memory, memory)。"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, H, T, d_k)"""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_k) -> (B, T, C)；transpose 后内存不连续，先 contiguous 再 view"""
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """query: (B, T_q, C)；key、value: (B, T_k, C)；返回 (B, T_q, C)"""
        # 1. 线性投影后切头
        q = self.split_heads(self.w_q(query))              # (B, H, T_q, d_k)
        k = self.split_heads(self.w_k(key))                # (B, H, T_k, d_k)
        v = self.split_heads(self.w_v(value))              # (B, H, T_k, d_k)
        # 2. H 个头并行算 attention（H 维和 B 维一样当 batch 维）
        out = scaled_dot_product_attention(q, k, v, mask)  # (B, H, T_q, d_k)
        # 3. 拼回 (B, T_q, C)，再过 w_o 融合各头
        return self.w_o(self.combine_heads(out))
```

**第 2 步：FFN、位置编码、Encoder 层和 Decoder 层。** FFN 对每个位置单独做 `C → d_ff → C`（原论文 512 → 2048 → 512，ReLU）。位置编码用原论文的正弦编码，原理见 1.1 节。层结构按原论文的 Post-LN：每个子层都是 `x = LayerNorm(x + Dropout(Sublayer(x)))`。三个 attention 子层的分工：

| 子层 | 所在 | Q 来自 | K、V 来自 | mask |
| --- | --- | --- | --- | --- |
| self-attention | Encoder | 源句 | 源句 | `src_mask`：源端 padding，`(B, 1, 1, T_src)` |
| masked self-attention | Decoder | 目标前缀 | 目标前缀 | `tgt_mask`：目标端 padding 且因果，`(B, 1, T_tgt, T_tgt)` |
| cross-attention | Decoder | Decoder 上一子层 | Encoder 输出 `memory` | `src_mask`（源句完整已知，不需要因果） |

```python
class PositionwiseFeedForward(nn.Module):
    """Linear(C -> d_ff) -> ReLU -> Linear(d_ff -> C)，每个位置独立计算"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)"""
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """PE[pos, 2i] = sin(pos / 10000^(2i/C))，PE[pos, 2i+1] = cos(同上)，加到 embedding 上再 dropout"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1).float()                    # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (C/2,)
        pe = torch.zeros(max_len, d_model)                                       # (max_len, C)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, C)：不训练，但跟着 .to(device) 走、存进 state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)"""
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderLayer(nn.Module):
    """self-attention -> FFN，每个子层 Post-LN：LN(x + Dropout(Sublayer(x)))"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T_src, C)；src_mask: (B, 1, 1, T_src)。返回 (B, T_src, C)"""
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderLayer(nn.Module):
    """masked self-attention -> cross-attention -> FFN，每个子层 Post-LN"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T_tgt, C)；memory: (B, T_src, C)。返回 (B, T_tgt, C)"""
        # 1. masked self-attention：Q = K = V = decoder，tgt_mask 挡住未来和 padding
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # 2. cross-attention：Q 来自 decoder，K、V 来自 memory，注意力矩阵 (B, H, T_tgt, T_src)
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        # 3. FFN
        return self.norm3(x + self.dropout(self.ffn(x)))
```

**第 3 步：拼成 Transformer。** N 个层要放进 `nn.ModuleList`，放在普通 list 里不会被注册，`parameters()` 找不到，优化器也不会更新它们。`encode` 和 `decode` 分开写，是因为生成时源句只编码一次，`decode` 每生成一个 token 调用一次（见 1.3 节）。原论文把 embedding 乘 $\sqrt{d_{\text{model}}}$，让词向量和取值在 -1 到 1 之间的位置编码量级相当。

```python
def make_src_mask(src: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """(B, T_src) -> (B, 1, 1, T_src)，True = 真实 token；广播到每个头、每个 query"""
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """(B, T_tgt) -> (B, 1, T_tgt, T_tgt)：padding mask 与因果 mask 同时为 True 才可见"""
    seq_len = tgt.size(1)
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)                                # (B, 1, 1, T)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).bool()      # (T, T)，下三角
    return pad_mask & causal                                                          # (B, 1, T, T)


class Transformer(nn.Module):
    """Encoder-Decoder Transformer：forward(src, tgt) -> logits (B, T_tgt, V)"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048,
                 max_len: int = 512, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """src: (B, T_src) -> memory: (B, T_src, C)"""
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """tgt: (B, T_tgt)，memory: (B, T_src, C) -> logits: (B, T_tgt, V)"""
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src: (B, T_src)；tgt: (B, T_tgt)，以 BOS 开头的目标前缀（teacher forcing）"""
        src_mask = make_src_mask(src, self.pad_id)
        tgt_mask = make_tgt_mask(tgt, self.pad_id)
        return self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask)
```

训练时 `tgt` 喂 `[BOS, y1, ..., y_{n-1}]`，标签是 `[y1, ..., y_n]`（右移一位），logits reshape 成 `(B*T_tgt, V)` 后用 `nn.CrossEntropyLoss(ignore_index=pad_id)`。

### 自测

用上面的 `Transformer`。检查三件事：输出形状；decoder 的因果性（改掉位置 2 的目标 token，位置 0、1 的 logits 不变）；源端 padding 不影响结果（带 PAD 的第 2 句和去掉 PAD 单独跑一致）。`model.eval()` 关掉 dropout，结果才确定。

```python
def test_transformer() -> None:
    torch.manual_seed(42)
    model = Transformer(11, 13, d_model=32, num_heads=4, num_layers=2, d_ff=64).eval()
    src = torch.tensor([[5, 6, 7, 8, 9], [5, 6, 7, 0, 0]])    # (B=2, T_src=5)，第 2 句后两个是 PAD
    tgt = torch.tensor([[1, 3, 4, 5], [1, 6, 7, 8]])           # (B=2, T_tgt=4)，1 是 BOS
    logits = model(src, tgt)
    assert logits.shape == (2, 4, 13)                          # 1. (B, T_tgt, V)
    logits2 = model(src, torch.tensor([[1, 3, 9, 5], [1, 6, 9, 8]]))   # 2. 只改位置 2
    assert torch.allclose(logits[:, :2], logits2[:, :2], atol=1e-6)
    assert not torch.allclose(logits[:, 2], logits2[:, 2])
    assert torch.allclose(model(src[1:, :3], tgt[1:]), logits[1:], atol=1e-5)   # 3. padding 无影响
    print("all tests passed")


if __name__ == "__main__":
    test_transformer()   # all tests passed
```

### 关键追问

- **为什么除以 $\sqrt{d_k}$？** 若 $q,k$ 各维独立、均值 0、方差 1，点积 $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ 的方差是 $d_k$；$d_k=64$ 时分数的标准差是 8，softmax 接近 one-hot，梯度几乎为 0。除以 $\sqrt{d_k}$ 把方差拉回 1，多头时 $d_k$ 是每个头的维度 C/H。
- **为什么要多头？** 单头的 softmax 只能给出一种加权平均，多个关系（指代、句法、相邻词）会被混在一起；H 个头在各自 C/H 维的子空间里学不同的注意力模式，总计算量和单头差不多。
- **Encoder、Decoder、cross-attention 各管什么？** Encoder 是双向 self-attention，只屏蔽 padding；Decoder 的 self-attention 加因果 mask，保证训练时位置 t 看不到答案；cross-attention 的 Q 来自 decoder、K/V 来自 `memory`，是两半之间唯一的连接。BERT 只用 Encoder 堆叠（encoder-only），GPT 只用去掉 cross-attention 的 Decoder 层（decoder-only）。
- **Post-LN 和 Pre-LN？** Post-LN（原论文，本节写法）是 `LN(x + Sublayer(x))`，深层时靠近输出的梯度大，需要学习率 warmup。Pre-LN 是 `x + Sublayer(LN(x))`，残差主干直通，深层更好训，最后要补一个 LayerNorm；GPT-2 之后的 LLM 基本都用 Pre-LN（LLaMA 用 RMSNorm）。
- **复杂度？** 每层 attention 打分和加权求和是 $O(T^2 d)$，投影和 FFN 是 $O(T d^2)$，序列长时 $T^2$ 项占主导；cross-attention 是 $O(T_{\text{tgt}}T_{\text{src}}d)$。注意力矩阵 `(B, H, T, T)` 的显存是 $O(T^2)$，FlashAttention 分块计算、不存整个矩阵，额外显存降到 $O(T)$，计算量不变。

---

## 1.1 位置编码（白板讲解）

白板三步：1）写「猫追狗」「狗追猫」说明为什么需要；2）写正弦公式、画时钟指针；3）画对比表讲相对位置、ALiBi、RoPE。

attention 只算内容的两两点积，打乱输入顺序，输出只会跟着打乱（置换等变，$\operatorname{Attn}(PX)=P\,\operatorname{Attn}(X)$），FFN、LayerNorm 也逐位置计算。所以不加位置信息时，「猫追狗」和「狗追猫」里「猫」的表示完全一样，模型看到的只是词袋。

### 正弦编码（原始 Transformer）

$$
PE_{(pos,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad
PE_{(pos,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

记 $\omega_i=10000^{-2i/d}$。**时钟指针**：每对 $(\sin(pos\,\omega_i), \cos(pos\,\omega_i))$ 是单位圆上的一根指针，位置加 1 就转 $\omega_i$ 弧度。$i=0$ 每步转 1 弧度，像秒针；$i$ 越大转得越慢，像时针。一个位置就是所有指针读数的组合，和词向量相加后作为输入。原论文试过可学习的位置表，效果几乎一样，选正弦是希望能外推到更长的序列。

**平移 = 旋转**：由和角公式，$PE_{pos+k}=M_k\,PE_{pos}$，$M_k$ 是把第 $i$ 对指针转 $k\omega_i$ 的块对角旋转矩阵，只和 $k$ 有关。所以模型用一个固定的线性变换就能表达「往后数 $k$ 个」，而且 $PE_{pos}\cdot PE_{pos+k}=\sum_i\cos(k\omega_i)$ 只依赖距离。PyTorch 模块见第 1 节。

### 实现

```python
import numpy as np


def sinusoidal_encoding_np(max_len: int, d_model: int) -> np.ndarray:
    """Returns: (max_len, d_model)，第 pos 行是位置 pos 的编码。d_model 须为偶数。"""
    pos = np.arange(max_len)[:, None]                        # (L, 1)
    two_i = np.arange(0, d_model, 2)[None, :]                # (1, C/2)，公式里的 2i
    angle = pos / np.power(10000.0, two_i / d_model)         # (L, C/2)，广播
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle)                              # 偶数维放 sin
    pe[:, 1::2] = np.cos(angle)                              # 奇数维放 cos
    return pe


def test_sinusoidal_encoding_np() -> None:
    pe = sinusoidal_encoding_np(64, 16)                      # (64, 16)
    # 1. 位置 0 是 sin(0)=0 和 cos(0)=1 交替
    assert pe.shape == (64, 16) and np.allclose(pe[0], np.tile([0.0, 1.0], 8))
    # 2. 相似度只看距离 k：起点 10 和起点 30 结果相同，k=0 时等于 d/2，越远越小
    sims = np.array([pe[10] @ pe[10 + k] for k in (0, 1, 4, 16)])
    assert np.allclose(sims, [pe[30] @ pe[30 + k] for k in (0, 1, 4, 16)])
    print(np.round(sims, 4))                                 # [8.     7.4852 5.5597 4.214 ]
    print("all tests passed")


if __name__ == "__main__":
    test_sinusoidal_encoding_np()
```

### 对比表

相对位置的出发点是「隔了几个词」往往比「排在第几」更重要，所以不改输入，在每层 attention 打分时加一项只依赖距离的量：$s_{ij}=q_i^\top k_j/\sqrt{d_k}+b(i-j)$。

| 方法 | 做法 | 加在哪里 | 可学习 | 长度外推 | 代表模型 |
| --- | --- | --- | --- | --- | --- |
| 正弦 | 上面的公式 | 输入 embedding | 固定 | 公式能算任意位置，没训练过的长度效果不保证 | 原始 Transformer |
| 可学习绝对位置 | $(L_{\max}, d)$ 的表，`tok_emb[ids] + pos_emb[:T]` | 输入 embedding | 可学习 | 不能超过表长 $L_{\max}$ | BERT、GPT-2 |
| 相对位置（T5） | 每个 head 一个标量 $b(i-j)$，距离分桶（远处按对数合并） | attention 分数 | 可学习 | 远距离共用一个桶，任意长度都能算 | T5 |
| ALiBi | $b(i-j)=-m_h(i-j)$，每个 head 固定斜率（8 个 head 取 1/2 到 1/256） | attention 分数 | 固定 | 好，这是它的设计目标 | BLOOM |
| RoPE | 把 q、k 按各自位置旋转，点积只剩 $m-n$（1.2 节） | Q、K | 固定 | 原版一般，靠位置插值等方法扩长 | LLaMA |

### 关键追问

- **为什么相加而不是拼接？** 拼接会让维度变大，而且进第一个线性层时 $W[x; p]=W_x x+W_p p$，本来就是各自投影再相加。高维空间里语义和位置可以占近似正交的子空间，相加后仍能用线性投影分开。
- **为什么 sin 和 cos 都要？为什么是 10000？** 单靠 sin 定不了相位（$\sin a=\sin(\pi-a)$），平移性质用到的和角公式也要同时有 sin 和 cos。10000 是经验值，让最慢的指针在常见长度内转不满一圈；它是超参数，Llama 3 把 RoPE 的 base 调到 500000。
- **位置信息在哪一层注入？** 绝对位置（可学习、正弦）只在最底层加一次，靠残差传上去。T5 偏置、ALiBi、RoPE 在每一层的 attention 打分里都重新作用，而且只影响打分（RoPE 不碰 V）。
- **长度外推怎么办？** 可学习位置直接不行，正弦能算但模型没见过，ALiBi 为外推而设计。RoPE 常用位置插值：目标长度 $L'$ 上的位置 $m$ 缩成 $mL/L'$，让转角落回训练时见过的范围，再少量微调（NTK-aware、YaRN 是改进版）。

---

## 1.2 RoPE（旋转位置编码）

绝对位置编码把位置向量**加**到词向量上，长文本外推表现不佳。RoPE 用**旋转**表达位置：把 q、k 的每两个通道看作平面上的一个箭头，位置 $m$ 就把它转 $m\theta_i$（$\theta_i=10000^{-2i/d}$，低维转得快、高维转得慢），相当于把 1.1 节「平移 = 旋转」直接用在 Q、K 上。点积时绝对角度抵消，**只剩相对距离 $m-n$**，LLaMA、Qwen、ChatGLM 等主流开源模型都用它：

$$
\langle R_m q,\ R_n k\rangle = q^\top R_{n-m} k
$$

对每一对通道 $(x_{2i},x_{2i+1})$ 展开就是标准的二维旋转：

$$
x'_{2i}=x_{2i}\cos(m\theta_i)-x_{2i+1}\sin(m\theta_i),\qquad
x'_{2i+1}=x_{2i+1}\cos(m\theta_i)+x_{2i}\sin(m\theta_i)
$$

### 实现

```python
import numpy as np


def apply_rope(x: np.ndarray, freq_base: float = 10000.0) -> np.ndarray:
    """x: (B, T, H, d_k)，d_k 为偶数（相邻两维一对）。Returns: 同形状，位置 m 的第 i 对通道转 m * theta_i。"""
    _, seq_len, _, d_k = x.shape
    assert d_k % 2 == 0, "d_k 必须是偶数才能两两旋转"
    # 1. 每对通道的频率 theta_i = base^(-2i/d_k)
    theta = 1.0 / (freq_base ** (np.arange(0, d_k, 2) / d_k))          # (d_k/2,)
    # 2. 角度 m * theta_i：位置和频率的外积
    freqs = np.outer(np.arange(seq_len), theta)                       # (T, d_k/2)
    # 3. cos/sin 每个复制两次，对齐交错排列的 d_k 维
    cos_val = np.repeat(np.cos(freqs), 2, axis=-1)[None, :, None, :]  # (1, T, 1, d_k)
    sin_val = np.repeat(np.sin(freqs), 2, axis=-1)[None, :, None, :]  # (1, T, 1, d_k)
    # 4. 旋转：每对 (x_even, x_odd) 变成 x * cos + (-x_odd, x_even) * sin
    swapped = np.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)  # (B, T, H, d_k)
    return x * cos_val + swapped * sin_val


def test_rope() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 8, 2, 16))                                # (B, T, H, d_k)
    # 1. 旋转是正交变换，保范数
    assert np.allclose(np.linalg.norm(apply_rope(x), axis=-1), np.linalg.norm(x, axis=-1))
    # 2. 同一对 q、k 放到不同位置：点积只依赖 m - n
    q, k = rng.normal(size=16), rng.normal(size=16)
    rq = apply_rope(np.tile(q, (1, 8, 1, 1)))[0, :, 0]                # (T, d_k)，第 m 行是 R_m q
    rk = apply_rope(np.tile(k, (1, 8, 1, 1)))[0, :, 0]                # (T, d_k)
    assert np.isclose(rq[3] @ rk[1], rq[7] @ rk[5])                   # m - n 都是 2
    assert not np.isclose(rq[3] @ rk[1], rq[3] @ rk[2])               # 距离变了，点积跟着变
    print("all tests passed")


if __name__ == "__main__":
    test_rope()
```

### 关键追问

- **为什么 `np.repeat(..., 2)` 而不是 `np.tile`？** 这份实现是**交错**配对（相邻两维一对），每个角度要连续复制两次才能对齐 $(x_{2i},x_{2i+1})$。HuggingFace 常见的 `rotate_half` 是「前半 / 后半」配对，要用 `np.tile`，两种约定不能混用。
- **作用在哪？** 只转 **Q 和 K**（只有它们进点积），不碰 V，每一层 attention 都要做。
- **怎么扩长？** 位置插值把每步转角调小，让同样的角度范围装下更长序列；NTK-aware、YaRN 等是在它基础上的改进。

---

## 1.3 解码：Greedy、Beam Search 与采样

训练时 decoder 一次看到整句答案（teacher forcing）；推理时只能自回归地一个一个生成：源句先 `encode` 一次得到 `memory`，再从 `[BOS]` 开始，每步把已生成的前缀喂给 `decode`，只取最后一个位置的 logits `(B, V)` 挑出下一个 token 拼到末尾，直到 EOS 或 `max_len`。各种方法只在「怎么挑」上不同：

| 方法 | 怎么挑下一个 token | 输出 | 常见用途 |
| --- | --- | --- | --- |
| Greedy | argmax | 确定 | 抽取、代码、评测 |
| Beam Search | 保留累计 log-prob 最高的 k 条路径 | 确定 | 翻译、摘要、语音识别（k 常取 4 到 5） |
| 采样 | 温度缩放、top-k / top-p 截断后按概率抽 | 随机 | 对话、创意写作 |

Beam 的分数是累计 log-prob（连乘变连加，防止下溢），排名时除以长度惩罚 $L^{\alpha}$（$L$ 是已生成的长度）：

$$
\text{score}(y_{1:L})=\sum_{t=1}^{L}\log p(y_t\mid y_{<t},x),\qquad \text{rank}=\frac{\text{score}}{L^{\alpha}}
$$

采样先做温度缩放 $p_i\propto\exp(z_i/T)$（$T<1$ 更尖，$T>1$ 更平），再截断：**Top-K** 只留最大的 K 个；**Top-P**（nucleus）按概率降序，取累计概率刚好达到 p 的最小集合，候选数随分布自动伸缩。

### 实现

下面的函数只依赖第 1 节 `Transformer` 的两个方法：`memory = model.encode(src, src_mask)` 和 `logits = model.decode(tgt, memory, src_mask, tgt_mask)`，第 1 节的模型可以直接传进来。`greedy_decode` 是批量版：每行生成 EOS 的时间不同，用 `(B,)` 的布尔向量 `finished` 记录，结束的行之后只补 PAD。把 `pick_fn` 换成采样函数，同一个循环就变成采样解码。

```python
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def greedy_decode(model: nn.Module, src: torch.Tensor, max_len: int, bos_id: int, eos_id: int,
                  pad_id: int = 0, pick_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                  ) -> torch.Tensor:
    """
    Args:
        src: (B, T_src)；pick_fn: logits (B, V) -> token (B,)，默认 argmax（greedy）
    Returns:
        (B, 1 + n)，以 BOS 开头，n <= max_len；每行 EOS 之后补 pad_id
    """
    model.eval()                                                          # 关掉 dropout
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)                  # (B, 1, 1, T_src)
    memory = model.encode(src, src_mask)                                  # 1. 源句只编码一次，(B, T_src, C)
    ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)   # (B, 1)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)        # (B,)
    for _ in range(max_len):
        tgt_mask = torch.tril(torch.ones(ys.size(1), ys.size(1), device=src.device))  # (T, T) 因果
        logits = model.decode(ys, memory, src_mask, tgt_mask)[:, -1, :]   # 2. 只取最后一个位置，(B, V)
        next_token = logits.argmax(dim=-1) if pick_fn is None else pick_fn(logits)   # (B,)
        next_token = next_token.masked_fill(finished, pad_id)             # 3. 已结束的行只补 PAD
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)              # (B, T + 1)
        finished = finished | (next_token == eos_id)                      # 4. 用 | 累积，PAD 不会把它改回去
        if finished.all():
            break
    return ys


def filter_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0,
                  top_p: float = 1.0) -> torch.Tensor:
    """logits: (B, V) -> (B, V)；被过滤的位置置 -inf，softmax 后概率正好是 0，剩下的自动重新归一化"""
    logits = logits / temperature                                         # 1. 温度
    if top_k > 0:                                                         # 2. top-k：比每行第 k 大还小的屏蔽
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]   # (B, 1)
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:                                                       # 3. top-p：降序排序后算累计概率
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)        # (B, V)
        probs = F.softmax(sorted_logits, dim=-1)
        sorted_remove = (probs.cumsum(dim=-1) - probs) >= top_p           # 前面已凑够 p 的都多余；第 1 个永远保留
        remove = sorted_remove.scatter(1, sorted_idx, sorted_remove)      # 排序后的顺序 -> 原顺序
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """logits: (B, V) -> (B,)；temperature = 0 是 greedy 的极限，单独处理避免除零"""
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)    # (B, V)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def beam_search(model: nn.Module, src: torch.Tensor, max_len: int, bos_id: int, eos_id: int,
                beam_width: int = 3, alpha: float = 0.0, pad_id: int = 0) -> Dict:
    """
    Args:
        src: (1, T_src)，一条源句；alpha: 长度惩罚，排名用 score / L ** alpha
    Returns:
        {"tokens": [BOS, ...], "score": 累计 log-prob}
    """
    model.eval()
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)                  # (1, 1, 1, T_src)
    memory = model.encode(src, src_mask)                                  # (1, T_src, C)

    def rank(beam: Dict) -> float:
        return beam["score"] / max(len(beam["tokens"]) - 1, 1) ** alpha

    beams = [{"tokens": [bos_id], "score": 0.0}]
    for _ in range(max_len):
        candidates = []
        for beam in beams:
            if beam["tokens"][-1] == eos_id:      # 1. 已结束：原样带到下一轮，用自己的分数继续竞争
                candidates.append(beam)
                continue
            tgt = torch.tensor([beam["tokens"]], device=src.device)                       # (1, T)
            tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1), device=src.device))
            log_probs = F.log_softmax(model.decode(tgt, memory, src_mask, tgt_mask)[0, -1], dim=-1)  # (V,)
            top_lp, top_ids = torch.topk(log_probs, min(beam_width, log_probs.numel()))  # 2. 每条只展开前 k 个
            for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                candidates.append({"tokens": beam["tokens"] + [tok], "score": beam["score"] + lp})
        beams = sorted(candidates, key=rank, reverse=True)[:beam_width]  # 3. 全局剪枝回 k 条
        if all(b["tokens"][-1] == eos_id for b in beams):                 # 4. 全部结束就停
            break
    return beams[0]                                                       # 已按 rank 降序
```

采样解码的调用方式：`greedy_decode(model, src, 20, bos_id, eos_id, pick_fn=lambda z: sample_next(z, 0.8, top_k=50, top_p=0.9))`。

### 自测

为了本节能单独运行，用一个只有 `encode` / `decode` 两个方法的随机小模型代替第 1 节的 `Transformer`：decoder 状态是前缀 embedding 的累加和，位置 t 只依赖不晚于 t 的 token，效果和因果 mask 相同。约定 PAD = 0、BOS = 1、EOS = 2。

```python
import itertools


class StubSeq2Seq(nn.Module):
    """只为测试解码：随机初始化，接口同第 1 节的 Transformer，mask 参数接收但不用"""

    def __init__(self, vocab_size: int = 8, d_model: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask):
        return self.embed(src)                                                    # (B, T_src, C)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        h = self.embed(tgt).cumsum(dim=1) + memory.mean(dim=1, keepdim=True)     # (B, T_tgt, C)
        logits = self.fc_out(torch.tanh(h))                                       # (B, T_tgt, V)
        return logits.index_fill(-1, torch.tensor([0, 1]), -1e9)                  # 不生成 PAD、BOS


def test_decoding() -> None:
    torch.manual_seed(45)
    model, bos, eos = StubSeq2Seq(), 1, 2
    src = torch.tensor([[3, 4, 5], [6, 7, 0]])                                    # (B=2, T_src=3)
    out = greedy_decode(model, src, 6, bos, eos)
    print(out.tolist())
    # 1. 采样时 top_k = 1 等于 greedy；top-p 小例子：p = 0.7 只留 0.5 和 0.3
    assert torch.equal(greedy_decode(model, src, 6, bos, eos, pick_fn=lambda z: sample_next(z, top_k=1)), out)
    toy = torch.log(torch.tensor([[0.15, 0.5, 0.05, 0.3]]))
    assert torch.isfinite(filter_logits(toy, top_p=0.7)).tolist() == [[False, True, False, True]]
    # 2. beam：宽度 1 等于 greedy；宽度 64 时前两步一条都不剪（8 条、7 * 8 + 1 条），结果等于暴力枚举
    src1, memory = src[:1], model.encode(src[:1], None)
    beams = {w: beam_search(model, src1, 3, bos, eos, beam_width=w) for w in [1, 3, 64]}
    for w, beam in beams.items():
        print(w, beam["tokens"], round(beam["score"], 4))
    assert beams[1]["tokens"] == greedy_decode(model, src1, 3, bos, eos)[0].tolist()

    def full_score(tokens: List[int]) -> float:   # 一次完整前向给整条序列打分，和 beam 的逐步累加相互独立
        lp = F.log_softmax(model.decode(torch.tensor([tokens]), memory, None, None)[0], dim=-1)   # (T, V)
        return sum(lp[t, tokens[t + 1]].item() for t in range(len(tokens) - 1))
    conts = {c[:c.index(eos) + 1] if eos in c else c for c in itertools.product(range(8), repeat=3)}
    best = max(([bos] + list(c) for c in conts), key=full_score)
    assert beams[64]["tokens"] == best and abs(beams[64]["score"] - full_score(best)) < 1e-5
    print(beam_search(model, src1, 3, bos, eos, beam_width=64, alpha=1.0)["tokens"])
    print("all tests passed")


if __name__ == "__main__":
    test_decoding()
    # [[1, 6, 6, 6, 2, 0, 0], [1, 6, 6, 6, 6, 4, 2]]
    # 1 [1, 6, 6, 6] -4.4276
    # 3 [1, 7, 4, 4] -3.3934
    # 64 [1, 2] -2.0994
    # [1, 7, 4, 4]
    # all tests passed
```

第 1 行在第 4 步生成 EOS，之后只补 PAD。宽度 1 就是 greedy，第一步选了 6 就只能沿着它走；宽度 3 留住了第一步只排第 3 的 7，总分反而更高；宽度 64 找到了全局最优 `[1, 2]`，也就是直接输出 EOS。累计 log-prob 每多一个 token 就多加一个负数，所以不加长度惩罚会偏爱短句；`alpha=1.0` 之后换成了 `[1, 7, 4, 4]`。

### 关键追问

- **KV Cache 是什么，占多少显存？** 因果 mask 下旧位置的 K、V 不会再变，所以每层把它们缓存起来，每步只为新 token 算 $q,k,v$ 并把 $k,v$ 拼到缓存末尾，不再重算整个前缀；cross-attention 的 K、V 来自 `memory`，整段生成只算一次。每条序列的缓存是 $2\times n_{\text{layers}}\times n_{\text{kv}}\times d_{\text{head}}\times T\times\text{bytes}$（2 是 K 和 V），Llama-3-8B（32 层、8 个 KV 头、$d_{\text{head}}=128$、fp16）每个 token 占 $2\times32\times8\times128\times2$ 字节，即 128 KiB。
- **重复惩罚怎么做？** CTRL 论文把已出现过的 token 的 logit 除以 $\theta$（$\theta\approx1.2$），HuggingFace 的实现补了一条：logit 为负时改成乘以 $\theta$，否则负 logit 反而变大。OpenAI API 的 frequency / presence penalty 用减法：$z_i\leftarrow z_i-\alpha_f c_i-\alpha_p\mathbf{1}[c_i>0]$，$c_i$ 是已出现次数。
- **为什么要长度惩罚？Beam Search 保证最优吗？** log-prob 恒为负，越长分越低，不加惩罚偏爱短句（上面宽度 64 的结果就是空输出），常用 $\text{score}/L^{\alpha}$，$\alpha$ 取 0.6 到 1.0。不保证最优：每步只留 k 条，可能剪掉「当下分低、后面翻盘」的路径；$k=1$ 就是 greedy，每步复杂度 $O(kV)$。
- **Top-K 和 Top-P 怎么选，温度趋于 0 会怎样？** Top-K 的 K 固定，分布尖时放进垃圾、分布平时砍掉合理选项；Top-P 的候选数随分布伸缩。工业实现常一起用，顺序是温度 → top-k → top-p → softmax → 采样；$T\to0$ 退化成 greedy，代码里要特判防除零。
- **开放式生成为什么不用 beam search？** 概率最高的序列往往是最常见、最平淡的说法，还容易复读，所以对话、写作改用 top-p 采样；翻译的输出被源句约束，beam search 仍然好用。批量实现时把 B 个样本的 k 条 beam 摊平成 `(B * k, T)` 一起前向，带 KV Cache 时要按父 beam 的下标用 `index_select` 重排缓存。

---

## 2. KMeans

两步循环：**分配**（每个点归到最近的中心）和**更新**（每个簇的均值当新中心），直到中心不再移动。它在最小化 inertia，也就是每个点到所属中心的平方距离之和 $J=\sum_{i}\min_{j}\|x_i-\mu_j\|^2$。

**算距离那行最常考。** `((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)`：$(n,1,d)$ 减 $(1,k,d)$ 广播成 $(n,k,d)$，即每个点减每个中心；沿坐标维 `axis=2` 求和得到 $(n,k)$ 的平方距离矩阵，再 `argmin(axis=1)` 给每个点挑最近的中心。一行代替两层 for 循环。中间数组是 $(n,k,d)$，$k$ 小所以没问题；KNN 的训练点很多，改用第 3 节的展开公式。

**k-means++ 初始化。** 纯随机初始化可能把中心挤在一起。k-means++ 第一个中心随机挑，之后每个点以正比于 $D(x)^2$（到最近已选中心的平方距离）的概率被选为下一个中心，让初始中心互相远离。它只改初始化，是 sklearn 的默认。

### 实现（CodeSignal 风格）

接口仿 sklearn：`fit` 返回自己，`predict` 返回簇下标，结果存在 `centroids`、`labels_`、`inertia_`。边界处理：空簇保留旧中心；收敛用 `tol` 判断，浮点数不用 `==`；跑 `n_init` 次留 inertia 最小的一次，因为 k-means 只保证局部最优。

```python
from typing import Optional, Union

import numpy as np


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4,
                 init: Union[str, np.ndarray] = "k-means++", n_init: int = 10,
                 random_state: Optional[int] = None):
        """init: "k-means++"，或直接给定初始中心 (k, d)；n_init: 重启几次"""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None    # (k, d)
        self.labels_: Optional[np.ndarray] = None      # (n,)
        self.inertia_: Optional[float] = None          # 每个点到所属中心的平方距离之和

    @staticmethod
    def _sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """X: (n, d)，C: (k, d)。返回平方距离 (n, k)；不开根号，argmin 不变"""
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """X: (n, d)。返回初始中心 (k, d)"""
        n = X.shape[0]
        if self.n_clusters > n:
            raise ValueError(f"n_clusters={self.n_clusters} > n_samples={n}")
        if not isinstance(self.init, str):                   # 1. 直接给定初始中心
            return np.array(self.init, dtype=float)
        centroids = [X[rng.integers(n)]]                     # 2. k-means++：第一个中心随机挑
        for _ in range(1, self.n_clusters):
            d2 = self._sq_dist(X, np.array(centroids)).min(axis=1)    # (n,) 到最近已选中心
            probs = d2 / d2.sum() if d2.sum() > 0 else None           # 3. 正比于 D(x)^2；全部重合时均匀抽
            centroids.append(X[rng.choice(n, p=probs)])
        return np.array(centroids)

    def _run_once(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """跑一次完整的 k-means。X: (n, d)。返回最终中心 (k, d)"""
        centroids = self._init_centroids(X, rng)                      # (k, d)
        for _ in range(self.max_iter):
            labels = self._sq_dist(X, centroids).argmin(axis=1)       # 1. 分配 (n,)
            new_centroids = centroids.copy()
            for j in range(self.n_clusters):                          # 2. 更新：取簇均值，空簇保留旧中心
                members = X[labels == j]                              # (n_j, d)
                if len(members) > 0:
                    new_centroids[j] = members.mean(axis=0)
            if np.linalg.norm(new_centroids - centroids) <= self.tol:   # 3. 中心几乎不动就停
                return new_centroids
            centroids = new_centroids
        return centroids

    def fit(self, X: np.ndarray) -> "KMeans":
        """X: (n, d)"""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best_inertia, best_centroids = np.inf, None
        for _ in range(self.n_init if isinstance(self.init, str) else 1):   # 给定初始中心时只跑一次
            centroids = self._run_once(X, rng)
            inertia = float(self._sq_dist(X, centroids).min(axis=1).sum())
            if inertia < best_inertia:                                # 4. 留 inertia 最小的一次
                best_inertia, best_centroids = inertia, centroids
        self.centroids, self.inertia_ = best_centroids, best_inertia
        self.labels_ = self.predict(X)                                # 5. 用最终中心算 labels_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (m, d)。返回每个点最近中心的下标 (m,)"""
        if self.centroids is None:
            raise RuntimeError("call fit before predict")
        return self._sq_dist(np.asarray(X, dtype=float), self.centroids).argmin(axis=1)
```

### CodeSignal ML Core 版（纯 Python）

ML Core 的 k-Means 题不许 import（见基础篇第 0 节）：数据是 list of lists，辅助函数放顶层，入口叫 `solution`。初始中心以题面为准，这里取前 k 个点；平局取下标小的中心，和 `np.argmin` 一致。

```python
def squared_distance(p: list, q: list) -> float:
    return sum((a - b) ** 2 for a, b in zip(p, q))


def assign_clusters(data: list, centroids: list) -> list:
    """每个点最近中心的下标；min 遇到平局返回第一个，即下标小的"""
    k = len(centroids)
    return [min(range(k), key=lambda c: squared_distance(p, centroids[c])) for p in data]


def update_centroids(data: list, labels: list, centroids: list) -> list:
    """每个簇取均值；空簇保留旧中心"""
    new_centroids = []
    for c in range(len(centroids)):
        members = [p for p, label in zip(data, labels) if label == c]
        mean = [sum(col) / len(members) for col in zip(*members)]       # zip(*members) 按列取
        new_centroids.append(mean if members else list(centroids[c]))
    return new_centroids


def solution(data: list, k: int, max_iter: int = 100) -> list:
    """data: n 个点，每个点是 d 个 float 的 list。返回每个点的簇下标（长度 n 的 list）"""
    centroids = [list(p) for p in data[:k]]                    # 1. 初始化：前 k 个点
    labels = assign_clusters(data, centroids)
    for _ in range(max_iter):
        centroids = update_centroids(data, labels, centroids)  # 2. 更新中心
        new_labels = assign_clusters(data, centroids)          # 3. 重新分配
        if new_labels == labels:                               # 4. 分配不变，均值也不变，收敛
            break
        labels = new_labels
    return labels
```

### Example

```python
np.random.seed(42)
X = np.random.rand(100, 2)                       # (100, 2)
model = KMeans(n_clusters=3, random_state=42).fit(X)
print(model.centroids.shape, model.labels_.shape, round(model.inertia_, 4))   # (3, 2) (100,) 5.8103
print(solution([[0.0, 0.0], [10.0, 10.0], [0.5, 0.0], [10.0, 9.5]], k=2))     # [0, 1, 0, 1]
rng = np.random.default_rng(0)                   # 三团分得很开的点 (150, 2)
blobs = np.vstack([np.array(c) + rng.normal(scale=0.5, size=(50, 2)) for c in [(0, 0), (5, 5), (0, 5)]])
for n_init in (1, 10):
    m = KMeans(n_clusters=3, n_init=n_init, random_state=0).fit(blobs)
    print(n_init, round(m.inertia_, 2), sorted(np.bincount(m.labels_).tolist()))
# 1 719.32 [23, 27, 100]：只跑一次就陷进局部最优，两团被合成了一团
# 10 75.58 [50, 50, 50]
```

### 自测

```python
def test_kmeans() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([np.array(c) + rng.normal(scale=0.5, size=(50, 2)) for c in [(0, 0), (5, 5), (0, 5)]])
    model = KMeans(n_clusters=3, random_state=0).fit(X)
    pairs = set(zip(np.repeat(np.arange(3), 50), model.labels_))                 # (真实簇, 预测簇)
    assert len(pairs) == 3 and len(set(model.labels_)) == 3                      # 1. 一一对应，编号可以不同
    assert np.isclose(model.inertia_, KMeans._sq_dist(X, model.centroids).min(axis=1).sum())   # 2. 和中心一致
    Xs = X[rng.permutation(len(X))]           # 3. 纯 Python 版 == NumPy 版：初始中心相同，tol=0 跑到完全不动
    assert solution(Xs.tolist(), 3) == KMeans(n_clusters=3, init=Xs[:3], tol=0.0).fit(Xs).labels_.tolist()
    print("all tests passed")


if __name__ == "__main__":
    test_kmeans()
```

### 关键追问

- **收敛到全局最优吗？** 不保证。每轮 inertia 单调不增，只收敛到局部最优，所以要 k-means++ 加多次重启（Example 里 `n_init=1` 就把两团合成了一团）。
- **空簇怎么办？** 保留旧中心（上面的写法），或者把它重置到离自己中心最远的那个点。
- **复杂度？** 每轮 $O(nkd)$，$n$ 是样本数，$k$ 是簇数，$d$ 是维度。距离对特征尺度敏感，先标准化。
- **k 怎么选？** inertia 随 k 增大单调下降，不能直接取最小；画 inertia 对 k 的曲线找拐点（肘部法），或者比较 silhouette score。
- **CodeSignal 上最常见的丢分点？** 初始中心、平局规则、最大轮数没按题面写。hidden tests 比对的是确定的输出，题面说取前 k 个点当初始中心就必须照做。

---

## 3. KNN

KNN 没有训练过程：`fit` 只把训练数据存起来，预测时找离新点最近的 $k$ 个训练点，分类做多数投票，回归取标签均值。训练几乎免费，推理很贵。

$m$ 个测试点对 $n$ 个训练点，第 2 节的广播写法会生成 $(m,n,d)$ 的中间数组，$n$ 一大就撑爆内存。把平方距离展开成 $\|a-b\|^2=\|a\|^2-2\,a\cdot b+\|b\|^2$，中间项对所有点对一起算就是一次矩阵乘法 $AB^\top$，内存只要 $(m,n)$。

### 实现

```python
from collections import Counter

import numpy as np


def pairwise_sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (m, d)，B: (n, d)。返回 (m, n) 的平方距离矩阵"""
    A_sq = (A ** 2).sum(axis=1, keepdims=True)    # (m, 1)
    B_sq = (B ** 2).sum(axis=1)                   # (n,)，相加时广播成 (1, n)
    d2 = A_sq - 2 * A @ B.T + B_sq                # (m, n)
    return np.maximum(d2, 0)                      # 两个大数相减可能得到 -1e-12，开根号会变 nan


class KNN:
    def __init__(self, k: int = 3, task: str = "classification"):
        self.k = k
        self.task = task                          # "classification" 或 "regression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """X: (n, d)，y: (n,)。lazy learning：只存数据"""
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (m, d)。返回 (m,)"""
        k = min(self.k, len(self.X_train))
        d2 = pairwise_sq_dist(np.asarray(X, dtype=float), self.X_train)   # (m, n)，找最近邻不用开根号
        # 1. argpartition 只把每行最小的 k 个挪到前面，O(n)，这 k 个之间无序
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]                     # (m, k)
        # 2. 这 k 个按距离排好，平票时 Counter 选先出现的类别，即离得更近的那一类
        order = np.argsort(np.take_along_axis(d2, idx, axis=1), axis=1)     # (m, k)
        neighbor_y = self.y_train[np.take_along_axis(idx, order, axis=1)]   # (m, k)
        if self.task == "regression":
            return neighbor_y.mean(axis=1)                                  # 3. 回归：取均值
        return np.array([Counter(row).most_common(1)[0][0] for row in neighbor_y])   # 3. 分类：投票
```

### 自测

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def test_knn() -> None:
    rng = np.random.default_rng(0)
    X_train = np.vstack([rng.normal(0, 1, (50, 2)), rng.normal(3, 1, (50, 2))])   # (100, 2)
    y_train = np.array([0] * 50 + [1] * 50)
    X_test = rng.normal(1.5, 2, (40, 2))                                          # (40, 2)
    brute = ((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)        # (40, 100)
    assert np.allclose(pairwise_sq_dist(X_test, X_train), brute)                 # 1. 展开公式 == 广播
    pred = KNN(k=5).fit(X_train, y_train).predict(X_test)                        # 2. 二分类、k 奇数不会平票
    assert np.array_equal(pred, KNeighborsClassifier(5).fit(X_train, y_train).predict(X_test))
    reg = KNN(k=5, task="regression").fit(X_train, X_train[:, 0]).predict(X_test)
    assert np.allclose(reg, KNeighborsRegressor(5).fit(X_train, X_train[:, 0]).predict(X_test))
    print("all tests passed")


if __name__ == "__main__":
    test_knn()
```

### 关键追问

- **复杂度？** `fit` 是 $O(1)$，但要 $O(nd)$ 内存；预测 $m$ 个点 $O(mnd)$，每行选 top-k 用 `argpartition` 是 $O(n)$。
- **k 怎么选？** k 太小对噪声敏感、容易过拟合，k 太大边界过平、容易欠拟合；用交叉验证选，二分类取奇数避免平票。
- **为什么要标准化？** 距离会被尺度最大的特征主导，比如年收入（0 到 $10^6$）会盖住年龄（0 到 100）。
- **数据量大或者高维怎么办？** 低维用 KD-Tree、Ball Tree 剪枝；高维时最近和最远的距离趋于相同（维度灾难），先降维或用 embedding，再上近似最近邻索引（FAISS 的 IVF、HNSW）。RAG 的向量检索就是 KNN，距离常用第 8 节的余弦相似度。
- **ML Core 不许 import 怎么写？** 和第 2 节纯 Python 版同一个套路：顶层写 `euc_dist` 辅助函数，`solution(train_data, test_data, k)` 里对每个测试点按距离排序取前 k 个，再用 dict 计票。

---

## 4. Logistic Regression

名字叫回归，其实是二分类：线性打分 $z=w^\top x+b$，用 Sigmoid $\sigma(z)=1/(1+e^{-z})$ 压成正类概率 $p$，损失是二元交叉熵（BCE），梯度下降更新。BCE 对 logits 的梯度很干净，就是 $p-y$，所以

$$
\nabla_w\mathcal L=\frac{1}{n}X^\top(p-y),\qquad \nabla_b\mathcal L=\frac{1}{n}\sum_i(p_i-y_i)
$$

**为什么要写稳定版。**

- Sigmoid：`1 / (1 + np.exp(-z))` 在 $z=-1000$ 时 `np.exp(1000)` 溢出并报 RuntimeWarning；改写成 `np.exp(z) / (1 + np.exp(z))` 则在 $z=1000$ 时得到 inf/inf = nan。两式数学上相等，所以 $z\ge0$ 用前者、$z<0$ 用后者，`exp` 的参数永远不大于 0。
- BCE：先算 $p$ 再取 log，$z\ge37$ 时 $p$ 在 float64 里已经舍入成 1.0，标签为 0 时 $-\log(1-p)=-\log 0$，loss 变成 inf。把 $p=\sigma(z)$ 代进去化简成只用 logits 的形式：

$$
\mathcal L=\max(z,0)-zy+\log\left(1+e^{-|z|}\right)
$$

指数项永远是 $e^{-|z|}\in(0,1]$，不会溢出；`np.log1p(t)` 在 t 很小时比 `np.log(1 + t)` 精确。`nn.BCEWithLogitsLoss` 内部也是这种只吃 logits 的稳定写法。

### 实现

```python
from typing import List, Optional

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。z: 任意形状，返回同形状"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))      # 1. z >= 0：exp(-z) <= 1
    exp_z = np.exp(z[~pos])                        # 2. z < 0：exp(z) < 1
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def bce_with_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """logits, y: (n,)。返回平均 loss：max(z, 0) - z*y + log(1 + exp(-|z|))"""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)
    loss = np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return float(loss.mean())


class LogisticRegression:
    def __init__(self, lr: float = 0.1, max_iter: int = 1000, l2: float = 0.0):
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2                                       # L2 正则强度，截距不参与
        self.w: Optional[np.ndarray] = None                # (d,)
        self.b = 0.0
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """X: (n, d)，y: (n,)，取值 0/1"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)         # 拉平成 (n,)，防止 (n, 1) 广播成 (n, n)
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0                  # 凸问题，零初始化即可
        self.loss_history = []
        for _ in range(self.max_iter):
            logits = X @ self.w + self.b                   # 1. 前向只算到 logits (n,)
            reg = 0.5 * self.l2 * float(self.w @ self.w)
            self.loss_history.append(bce_with_logits(logits, y) + reg)
            err = sigmoid(logits) - y                      # 2. dL/dlogits = p - y，(n,)
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)   # 3. 加上 L2 的梯度 λw
            self.b -= self.lr * float(err.mean())          # 截距只管整体基准，不正则
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (m, d)。返回正类概率 (m,)"""
        if self.w is None:
            raise RuntimeError("call fit before predict")
        return sigmoid(np.asarray(X, dtype=float) @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """返回 0/1 标签 (m,)"""
        return (self.predict_proba(X) >= threshold).astype(int)
```

### 自测

和 sklearn 对拍时要换算正则系数：我们的目标是「平均 loss $+\frac{\lambda}{2}\|w\|^2$」，sklearn 是「$C\cdot$ 总 loss $+\frac12\|w\|^2$」，所以 $C=1/(n\lambda)$；两边都不正则截距。

```python
from sklearn.linear_model import LogisticRegression as SkLogisticRegression


def test_logistic_regression() -> None:
    z = np.linspace(-30, 30, 61)
    assert np.allclose(sigmoid(z), 1 / (1 + np.exp(-z)))                     # 1. 正常范围和原公式一致
    assert sigmoid(np.array([-1000.0, 1000.0])).tolist() == [0.0, 1.0]        # 2. 极端值无溢出、无 warning
    assert np.isclose(bce_with_logits(np.array([40.0]), np.array([0.0])), 40.0)   # 原公式这里是 inf
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))                                             # (200, 3)
    y = (X @ np.array([1.0, -2.0, 0.5]) + 0.3 + rng.normal(size=200) > 0).astype(int)
    model = LogisticRegression(lr=0.5, max_iter=2000, l2=0.01).fit(X, y)
    ref = SkLogisticRegression(C=1 / (200 * 0.01), tol=1e-8).fit(X, y)       # 3. 和 sklearn 对拍
    assert np.allclose(model.w, ref.coef_[0], atol=1e-4)
    assert np.isclose(model.b, ref.intercept_[0], atol=1e-4)
    assert np.all(np.diff(model.loss_history) <= 1e-12)                       # 4. 凸问题，loss 单调下降
    print("all tests passed")


if __name__ == "__main__":
    test_logistic_regression()
```

### 关键追问

- **为什么不用 MSE？** sigmoid 加 MSE 对 $w$ 非凸，而且梯度里多一个 $p(1-p)$，预测错得很离谱时 $p$ 饱和，梯度反而接近 0；BCE 加线性 logits 是凸的，梯度就是 $p-y$。
- **数据线性可分会怎样？** 不加正则时把 $\|w\|$ 放大总能让 loss 继续下降，权重发散到无穷；加 L2 或早停。
- **阈值一定是 0.5 吗？** 不一定，由误报和漏报的业务成本决定，按 precision、recall 选（8.1 节）。
- **多分类怎么办？** 换成 softmax 加交叉熵（第 9 节），梯度形式同样是 $P-Y$；或者训练 K 个一对多（one-vs-rest）的二分类器。
- **输出的概率能直接用吗？** 用 BCE 训练的 LR 本身校准得比较好；加了强正则或者类别重采样后，概率会偏，需要重新校准。

---

## 5. 线性回归与多项式回归

多元线性回归 $\hat y=Xw+b$ 有闭式解，不用迭代。$X$ 左边拼一列 1，截距变成 $\theta_0$，对均方误差求导令其为 0：

$$
\theta=(X^\top X)^{-1}X^\top y
$$

**代码里不写 `inv(X.T @ X)`。** 显式求逆最不稳，而且 $X^\top X$ 的条件数是 $X$ 的平方，特征一共线就会炸。`np.linalg.lstsq` 直接对 $X$ 做 SVD 求最小二乘，不构造 $X^\top X$，秩亏时还能给出最小范数解。面试时先写公式证明会推导，再说实现调 `lstsq`。

**Ridge** 的闭式解是 $\theta=(X^\top X+\alpha I)^{-1}X^\top y$。对角线加上 $\alpha I$ 把最小特征值至少抬到 $\alpha$，原本奇异的矩阵也可逆，所以能处理共线性；实现时用 `np.linalg.solve` 解方程，不求逆。

**多项式回归**就是特征扩展加线性回归：把 $x$ 展开成 $[x,x^2,\dots,x^p]$ 当新特征，套同一个求解器。注意区分「多元」（多个特征）和「多项式」（高次项）。

### 实现

```python
from typing import Optional

import numpy as np


def add_intercept(X: np.ndarray) -> np.ndarray:
    """X: (n, d) -> (n, d + 1)，最左边拼一列 1，截距变成 theta[0]"""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """x: (n,) -> (n, degree)，列依次是 x, x^2, ..., x^degree；截距由模型加"""
    return np.column_stack([x ** p for p in range(1, degree + 1)])


class LinearRegression:
    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha                                  # 0 是普通最小二乘，> 0 是 Ridge
        self.theta: Optional[np.ndarray] = None             # (d + 1,)，theta[0] 是截距

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """X: (n, d)，y: (n,)"""
        Xb = add_intercept(np.asarray(X, dtype=float))      # (n, d + 1)
        y = np.asarray(y, dtype=float)
        if self.alpha == 0:
            # 1. lstsq 返回 (解, 残差平方和, 秩, 奇异值)，只要解
            self.theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        else:
            # 2. Ridge：解 (X^T X + αI) θ = X^T y，截距不正则
            reg = self.alpha * np.eye(Xb.shape[1])
            reg[0, 0] = 0.0
            self.theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (m, d)。返回 (m,)"""
        return add_intercept(np.asarray(X, dtype=float)) @ self.theta
```

### 自测

```python
from sklearn.linear_model import LinearRegression as SkLinearRegression, Ridge


def test_linear_regression() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))                                               # (50, 3)
    y = X @ np.array([2.0, -1.0, 0.5]) + 3.0 + 0.1 * rng.normal(size=50)       # (50,)
    pairs = [(LinearRegression(), SkLinearRegression()), (LinearRegression(alpha=5.0), Ridge(alpha=5.0))]
    for ours, ref in pairs:                                                    # 1. OLS 和 Ridge 都和 sklearn 对拍
        ref.fit(X, y)
        assert np.allclose(ours.fit(X, y).theta, np.r_[ref.intercept_, ref.coef_])
    x = np.linspace(-2, 2, 20)                                                 # 2. 多项式：精确恢复 1 + 2x - 3x^2
    theta = LinearRegression().fit(polynomial_features(x, 2), 1 + 2 * x - 3 * x ** 2).theta
    assert np.allclose(theta, [1.0, 2.0, -3.0])
    x1 = np.arange(1.0, 6.0)                       # 3. 完全共线：第二列 = 第一列 + 1，加上截距列秩只有 2
    X_col = np.column_stack([x1, x1 + 1])          # inv(X^T X) 在这里报 LinAlgError，lstsq 照样拟合
    assert np.allclose(LinearRegression().fit(X_col, 2 * x1 + 3).predict(X_col), 2 * x1 + 3)
    print("all tests passed")


if __name__ == "__main__":
    test_linear_regression()
```

### 关键追问

- **什么时候不用闭式解？** 闭式解要 $O(nd^2+d^3)$；特征维度 $d$ 很大，或者数据放不进内存时，改用（小批量）梯度下降。
- **Ridge 和 Lasso 的区别？** L2 把所有权重往 0 收缩，有闭式解；L1 会把一部分权重压成正好 0，相当于特征选择，没有闭式解，用坐标下降求。
- **多项式回归的坑？** degree 高了会剧烈震荡去穿过每个训练点，严重过拟合，用 Ridge 压高次项、用验证集选 degree；多个特征展开时交叉项数量爆炸；$x^p$ 的量级差别很大，先标准化。

---

## 6. 决策树：信息熵与信息增益

- **熵** $H(y)=-\sum_c p_c\log_2 p_c$：全是同一类时为 0，各类均匀时最大。
- **信息增益** $IG=H(\text{父})-\sum_{\text{子}}\frac{n_{\text{子}}}{n}H(\text{子})$：切分前的熵减去切分后子集熵的加权平均。每一步选 IG 最大（切完最纯）的特征和阈值。
- **Gini 不纯度** $1-\sum_c p_c^2$ 是 CART 和 sklearn 的默认，不用算 log，结果通常和熵差不多。

### 实现

```python
import numpy as np


def entropy(y: np.ndarray) -> float:
    """y: (n,) 标签。返回以 2 为底的熵"""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)       # 各类别频次
    p = counts / len(y)                                 # np.unique 保证 p > 0，不会 log(0)
    return float(-np.sum(p * np.log2(p)))


def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """父节点熵减去子节点熵的加权平均；不加权的话，只有 2 个样本的纯净子节点会把分数骗高"""
    n = len(y_parent)
    child = len(y_left) / n * entropy(y_left) + len(y_right) / n * entropy(y_right)
    return entropy(y_parent) - child


if __name__ == "__main__":
    y = np.array([0, 0, 1, 1])
    assert entropy(y) == 1.0 and entropy(np.array([1, 1, 1])) == 0.0
    assert information_gain(y, y[:2], y[2:]) == 1.0                    # 完美切分，增益等于父节点的熵
    assert np.isclose(information_gain(y, y[[0, 2]], y[[1, 3]]), 0.0)  # 两边仍是一半一半，没有增益
    print("all tests passed")
```

### 关键追问

- **怎么找一个特征的最优阈值？** 把该特征的取值排序，在相邻两个不同取值的中点逐个试，取 IG 最大的；每个节点 $O(d\,n\log n)$。
- **怎么防过拟合？** 限制 `max_depth`、`min_samples_leaf`，或者事后剪枝；树按阈值切分，特征不需要标准化。

---

## 7. SVM

面试很少让手写 SMO，重点讲清几何直觉，代码直接调库：

- **最大间隔**：能分开两类的超平面有无数条，SVM 选离最近点（支持向量）最远的那条，泛化更好。只有支持向量决定边界。
- **软间隔与 C**：等价于 hinge loss $\max(0,1-y\,f(x))$ 加 L2 正则。`C` 越大越不容忍误分类，间隔越窄，越容易过拟合。
- **核技巧**：线性不可分时用核函数（如 RBF）隐式映射到高维再切，不用显式算高维坐标。RBF 的 `gamma` 越大，边界越弯。
- **要先标准化**：间隔由距离定义，和 KNN 一样受特征尺度影响。

```python
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 同心圆：一类被另一类包围，线性不可分
X, y = make_circles(n_samples=300, factor=0.4, noise=0.1, random_state=0)   # (300, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
for kernel in ("linear", "rbf"):
    clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0)).fit(X_train, y_train)
    print(kernel, round(clf.score(X_test, y_test), 3), len(clf[-1].support_))   # 测试准确率、支持向量个数
# linear 0.411 198：一条直线切不开同心圆
# rbf 1.0 35：映射到高维后完全分开，边界只由 35 个支持向量决定
```

---

## 8. Cosine Similarity

$$
\cos(a,b)=\frac{a\cdot b}{\|a\|\,\|b\|}
$$

余弦相似度只看方向、不看长度，取值在 $[-1,1]$。检索和 RAG 用它，是因为长文档的 embedding 模长往往更大，直接比内积会让长文档占便宜。向量做过 L2 归一化后 $\|a-b\|^2=2-2\cos(a,b)$，余弦和欧氏距离给出同样的排序，所以向量库通常先把所有向量归一化，之后只算内积。

### 实现

```python
import numpy as np


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """A: (n, d)，B: (m, d)。返回 (n, m)，第 i 行第 j 列是 cos(A[i], B[j])；单个向量传 a[None]"""
    # 1. 每行除以自己的 L2 模长，(n, d) 和 (m, d)；eps 防零向量除出 nan
    A_unit = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_unit = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    # 2. 归一化之后内积就是余弦，一次矩阵乘拿到全部 n * m 对
    return A_unit @ B_unit.T
```

### 自测

```python
if __name__ == "__main__":
    from sklearn.metrics.pairwise import cosine_similarity

    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(4, 8)), rng.normal(size=(3, 8))
    S = cosine_similarity_matrix(A, B)
    assert S.shape == (4, 3) and np.allclose(S, cosine_similarity(A, B))
    assert np.allclose(np.diag(cosine_similarity_matrix(A, 5 * A)), 1.0)    # 只看方向，不看长度
    assert np.allclose(cosine_similarity_matrix(np.zeros((1, 8)), B), 0.0)  # 零向量得到 0，不出 nan
    print("all tests passed")
```

### 关键追问

- **为什么要 `eps`？** 零向量的模长是 0，直接除得到 `nan`；空文本、全 padding 的行都可能产生零向量。
- **批量为什么先归一化再矩阵乘？** 归一化只要 $O((n+m)d)$，之后一次矩阵乘交给 BLAS 算完全部 $n\times m$ 对；逐对的 Python 循环要调用 $nm$ 次，慢几个数量级。
- **余弦相似度是距离吗？** 它越大越相似，方向和距离相反。$1-\cos$ 常叫余弦距离，但它不满足三角不等式，不是严格的度量。
- **什么时候不该用？** 模长本身有意义时（词频强度、推荐里的置信度），归一化会把这部分信息丢掉。

---

## 8.1 分类指标：Precision、Recall、F1 与 AUC

### Precision、Recall、F1

$$
\text{Precision}=\frac{TP}{TP+FP},\qquad
\text{Recall}=\frac{TP}{TP+FN},\qquad
F_1=\frac{2PR}{P+R}
$$

Precision（查准率）是预测为正的样本里真正为正的比例，管**别误报**；Recall（查全率）是真正的正样本里被抓到的比例，管**别漏报**。阈值调低，recall 升、precision 降；调高则相反。F1 是两者的调和平均，对偏科更狠：0.9 和 0.1 的算术平均是 0.5，F1 只有 0.18。

```python
from typing import List, Tuple


def precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    """二分类，y_true / y_pred 是长度 n 的 0/1 列表。分母为 0 时返回 0（同 sklearn 的 zero_division=0）。"""
    # 1. 数混淆矩阵里的 TP、FP、FN
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    # 2. 一个正类都没预测出来，或者没有正样本时，分母为 0，显式返回 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1
```

多分类：**macro** 对每个类别 c 用 `y == c` 做一次 one-vs-rest，算出 F1 再取平均，小类和大类等权；**micro** 先把所有类别的 TP、FP、FN 加总再算，被大类主导，单标签多分类下 micro-F1 就等于 accuracy。关心稀有类就看 macro。

### ROC-AUC

ROC 曲线把阈值从高到低扫一遍，横轴 $\text{FPR}=FP/N$，纵轴 $\text{TPR}=TP/P$（就是 recall），$P$、$N$ 是正、负样本数，AUC 是曲线下面积。面试最常考它的概率含义：随机抽一个正样本和一个负样本，正样本分数更高的概率，平局算一半：

$$
\text{AUC}=\frac{1}{PN}\sum_{i\in\text{pos}}\sum_{j\in\text{neg}}\Big(\mathbf{1}[s_i>s_j]+\tfrac12\,\mathbf{1}[s_i=s_j]\Big)
$$

AUC 不需要选阈值，而且只看排序：分数做任何严格递增变换（sigmoid、log、乘正数），AUC 都不变，所以拿 logits 和拿概率算结果一样。**先写暴力版**：照公式数对，$O(PN)$，最不容易错，之后拿它当对拍的标准答案。

```python
def solution(labels: list, scores: list) -> float:
    """labels: 长度 n，1/True 为正类；scores: 长度 n，概率或 logits 都行。返回 [0, 1] 的 AUC。"""
    # 1. 按标签分成两堆；if y 同时兼容 1/0 和 True/False
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    if not pos or not neg:      # 只有一类时 AUC 无定义：报错，别偷偷返回 0.5
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

手算：9 对里 0.9 赢 3 对；0.8 赢 2 对，和负样本 0.8 打平记 0.5；0.4 赢 2 对。合计 7.5 / 9。

**面试版：排名公式（Mann-Whitney U），O(n log n)。** 所有样本按分数升序排名（最小的是第 1 名），$R_+$ 是正样本的名次之和，则 $\text{AUC}=\big(R_+-P(P+1)/2\big)/(PN)$。没有平局时，正样本的名次 = 排在它下面的负样本数 + 下面的正样本数 + 1；对所有正样本求和，后两项合起来正好是 $1+2+\cdots+P=P(P+1)/2$，减掉它就只剩「赢的对数」。有平局时，同分的一组都给**平均名次**，相当于组内每个（正, 负）对各记 0.5。

```python
def auc_rank(labels: list, scores: list) -> float:
    """排名公式，纯 Python。labels / scores 同 solution。时间 O(n log n)。"""
    n, n_pos = len(scores), sum(1 for y in labels if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("only one class present, AUC is undefined")
    order = sorted(range(n), key=lambda i: scores[i])     # 1. 按分数升序排下标
    # 2. 同分组 order[i..j] 占名次 i+1 .. j+1，统一给平均名次
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    # 3. 正样本名次和减去 P(P+1)/2 得到赢的对数，再除以 P * N
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

平局必须按 0.5 记：如果同分样本按出现顺序拿名次，4 个同分样本（2 正 2 负）会随顺序算出 0.25 或 0.75，正确答案是 0.5。也可以先画 ROC 曲线（同分样本合成一个阈值点）再用梯形公式求面积，结果一样。数据大到无法全量排序时，把分数分进固定个数的桶、只累计每个桶的正负样本数，能 $O(n)$ 近似（`tf.keras.metrics.AUC` 默认 200 个阈值）。

### GAUC（推荐 / 广告常考）

全局 AUC 会拿用户 A 的正样本和用户 B 的负样本比，可线上排序只发生在同一个用户的候选之间。GAUC 先对每个用户单独算 AUC，再按曝光数（也有按点击数）加权平均，只有一类样本的用户跳过：

$$
\text{GAUC}=\frac{\sum_u w_u\,\text{AUC}_u}{\sum_u w_u}
$$

```python
def gauc(user_ids: list, labels: List[int], scores: List[float]) -> float:
    """按曝光数加权；用上面的 auc_rank。"""
    groups = {}                                         # 1. 用户 -> 样本下标
    for i, u in enumerate(user_ids):
        groups.setdefault(u, []).append(i)
    num, den = 0.0, 0.0
    for idx in groups.values():
        y = [labels[i] for i in idx]
        if 0 < sum(y) < len(y):                         # 2. 两类都有才算
            num += len(idx) * auc_rank(y, [scores[i] for i in idx])   # 3. 权重 = 曝光数
            den += len(idx)
    return num / den


users = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
labels = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]                # 用户 c 没有点击，跳过
scores = [0.8, 0.9, 0.7, 0.6, 0.3, 0.2, 0.1, 0.4, 0.5, 0.35]
print(auc_rank(labels, scores), gauc(users, labels, scores))   # 0.6190476190476191 0.41666666666666663
```

全局 AUC 0.619 看着还行，可用户 a、b 各自只有 0.5 和 0.3333：模型只学会了「用户 a 的分数整体偏高」。

### 自测

```python
if __name__ == "__main__":
    import random
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    random.seed(0)
    for _ in range(200):
        n = random.randint(2, 30)
        y = [0, 1] + [random.randint(0, 1) for _ in range(n - 2)]   # 保证两类都在
        s = [random.randint(0, 4) / 4 for _ in range(n)]             # 只有 5 种分数，平局很多
        pred = [int(v >= 0.5) for v in s]
        ref = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)[:3]
        assert all(abs(a - b) < 1e-12 for a, b in zip(precision_recall_f1(y, pred), ref))
        assert all(abs(f(y, s) - roc_auc_score(y, s)) < 1e-12 for f in (solution, auc_rank))
    print("all tests passed")
```

### 关键追问

- **类别不平衡时为什么看 AUC，不看 accuracy？** 1% 正例时全判负就有 99% accuracy，recall 却是 0。AUC 的 TPR、FPR 各在本类内部归一化，正负比例变了曲线不变，常数分数恰好是 0.5。
- **precision 和 recall 谁更重要？** 看错误的代价：垃圾邮件过滤怕误杀正常邮件，重 precision；欺诈识别、疾病筛查怕漏，重 recall。也可以用 $F_\beta$，$\beta>1$ 偏 recall。
- **ROC-AUC 和 PR-AUC 怎么选？** 负样本极多时 FPR 的分母 $N$ 很大，多出一堆误报 FPR 也几乎不动，ROC-AUC 偏乐观。precision 直接受误报影响，所以 PR-AUC 对不平衡更敏感，随机模型的 PR-AUC 约等于正例比例。
- **AUC 高，概率就准吗？** 不一定：AUC 只看排序，把所有概率除以 2，AUC 不变，校准却全错了。CTR 预估要拿概率出价，还得看 logloss 和校准。
- **能直接拿 AUC 当 loss 吗？** 不能，它由指示函数组成，几乎处处梯度为 0。常用可导的 pairwise 替代：对每个（正, 负）对最小化 $-\log\sigma(s_i-s_j)$，即 RankNet / BPR loss。

---

## 9. Softmax、交叉熵与常见 Loss

Softmax 把 logits 变成和为 1 的概率分布。实现时分子分母同乘 $e^{-\max_j z_j}$，数学上等价，但最大项变成 $e^0=1$，不会上溢：

$$
\operatorname{softmax}(z)_i=\frac{e^{z_i-\max_j z_j}}{\sum_k e^{z_k-\max_j z_j}}
$$

交叉熵只看真实类别那一项的概率，$\mathcal L=-\frac1N\sum_i\log p_{i,y_i}$。实现时把 softmax 和 log 合并成 log-softmax（log-sum-exp），除法变成减法，也不会出现 $\log 0$：

$$
\log\operatorname{softmax}(z)_k=z_k-\max_j z_j-\log\sum_{m}e^{z_m-\max_j z_j}
$$

### NumPy 实现

```python
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """x: 任意形状，例如 (B, C)；沿 axis 归一化，返回同形状，每行和为 1"""
    # 1. 减每行最大值：数学上等价，最大项变成 e^0 = 1，不会上溢
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    # 2. keepdims=True 让 (B, 1) 和 (B, C) 广播，每行各自归一化
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """logits: (B, C) 未经 softmax 的分数；y: (B,) 整数类别（不是 one-hot）。返回 batch 平均的交叉熵"""
    # 1. log-softmax = (z - max) - log Σ e^(z - max)，(B, C)；没有除法，也不会 log(0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    # 2. 花式索引：第 i 行取第 y[i] 列，(B,)
    return float(-log_probs[np.arange(len(y)), y].mean())


z = np.array([[1000.0, 0.0], [1.0, 2.0]])            # (2, 2)，第 0 行直接 np.exp 会溢出成 inf
print(softmax(z).round(4).tolist())                  # [[1.0, 0.0], [0.2689, 0.7311]]
print(f"{cross_entropy_from_logits(z, np.array([1, 1])):.4f}")   # 500.1566：p 下溢成 0 也不会得到 inf
```

带 batch 时必须写 `axis=-1, keepdims=True`：省掉 axis 的 `np.max(x)`、`e.sum()` 会对整个矩阵取最大、求和，每行不再各自归一化。合并后对 logits 的梯度是 $(p-\text{onehot})/N$，推导见第 11 节。手上只有概率时，先 `np.clip(p, 1e-15, 1)` 再取 log。

### PyTorch 实现

面试更常见的问法是「用 PyTorch 实现这个 loss」。只写 forward，backward 交给 autograd，注意三点：输入一律用 logits（多分类 `F.log_softmax`，二分类 `F.softplus`）；不要先 softmax 再 log，概率下溢成 0 时 loss 是 `inf`、梯度是 `nan`；中途别用 `.item()`、`.numpy()`，否则计算图断开。证明自己写对了，要和内置实现比两样东西：loss 值和梯度（下面的 `check`）。

#### 1. Cross-Entropy：ignore_index 与 label smoothing

target 等于 `ignore_index`（padding）的位置不计入 loss，`mean` 的分母是**有效位置数**。拿 -100 直接做下标会出错（`gather` 报越界，花式索引在类别数 ≥ 100 时还会静默读到倒数第 100 列），所以先换成合法下标 0，算完再清零。PyTorch 的 label smoothing 把 ε 均匀分给全部 $C$ 个类（包括真实类别），交叉熵拆成两项，第二项就是代码里的 `-log_probs.mean(dim=1)`：

$$
q_k=(1-\varepsilon)\,\mathbf{1}[k=y]+\frac{\varepsilon}{C},\qquad \mathcal L_i=(1-\varepsilon)\,(-\log p_{y_i})+\frac{\varepsilon}{C}\sum_{k=1}^{C}(-\log p_k)
$$

```python
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """loss: 逐元素 loss；reduction: 'none' | 'sum' | 'mean'"""
    if reduction == "none":
        return loss
    return loss.sum() if reduction == "sum" else loss.mean()


def check(my_fn: Callable, ref_fn: Callable, x: torch.Tensor, atol: float = 1e-6) -> bool:
    """对拍：my_fn、ref_fn 输入 x、返回标量 loss；loss 值和对 x 的梯度都一致才返回 True"""
    x1, x2 = x.detach().clone().requires_grad_(True), x.detach().clone().requires_grad_(True)
    loss1, loss2 = my_fn(x1), ref_fn(x2)
    g1, g2 = torch.autograd.grad(loss1, x1)[0], torch.autograd.grad(loss2, x2)[0]
    return torch.allclose(loss1, loss2, atol=atol) and torch.allclose(g1, g2, atol=atol)


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100,
                       label_smoothing: float = 0.0, reduction: str = "mean") -> torch.Tensor:
    """logits: (N, C) 未经 softmax 的分数；target: (N,) 整数类别，等于 ignore_index 的位置不计入 loss。
    返回：'none' 时 (N,)，忽略位置为 0；否则标量（全部被忽略时 mean 是 nan，和内置一致）"""
    log_probs = F.log_softmax(logits, dim=1)                         # 1. (N, C)
    valid = target != ignore_index                                    # 2. (N,) True 表示要算
    safe_target = target.masked_fill(~valid, 0)                       # 3. 忽略位置换成合法下标 0
    nll = -log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)   # 4. (N,) 真实类别的 -log p
    smooth = -log_probs.mean(dim=1)                                   # 5. (N,) 对均匀分布的交叉熵
    loss = (1 - label_smoothing) * nll + label_smoothing * smooth     # 6. (N,)
    loss = loss.masked_fill(~valid, 0.0)                              # 7. 忽略位置清零
    if reduction == "mean":
        return loss.sum() / valid.sum()                               # 分母 = 有效位置数
    return reduce_loss(loss, reduction)
```

语言模型的 `(B, T, V)` logits 先错开一位再展平，`logits[:, :-1].reshape(-1, V)` 配 `tokens[:, 1:].reshape(-1)`；把 `(B, T, V)` 直接传给 `F.cross_entropy`，它会把第 1 维 T 当成类别数。

#### 2. BCE with logits 与 pos_weight

二分类和多标签用 sigmoid。`pos_weight`（记作 $w_p$）给正样本项加权，负:正 = 9:1 时常设 9。利用 $-\log\sigma(x)=\operatorname{softplus}(-x)$ 和 $-\log(1-\sigma(x))=\operatorname{softplus}(x)$：

$$
\mathcal L=-\big[w_p\,y\log\sigma(x)+(1-y)\log(1-\sigma(x))\big]=w_p\,y\operatorname{softplus}(-x)+(1-y)\operatorname{softplus}(x)
$$

这里要用 `F.softplus` 写：NumPy 里常见的稳定式 $\max(x,0)-xy+\log(1+e^{-|x|})$ 搬进 autograd 后 forward 没错，但 `clamp` 和 `abs` 在 x = 0 处取的梯度没有互相抵消，梯度变成 $1-y$（正确值是 $\sigma(0)-y=0.5-y$），而输出层零初始化时第一步的 logits 正好全是 0。

```python
def bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                    pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> torch.Tensor:
    """logits: (N,) 或 (N, C)；target: 同形状的 0/1（或软标签）；pos_weight: 标量或 (C,)"""
    w_p = 1.0 if pos_weight is None else pos_weight
    pos_term = w_p * target * F.softplus(-logits)    # 1. -w_p · y · log σ(x)
    neg_term = (1 - target) * F.softplus(logits)     # 2. -(1 - y) · log(1 - σ(x))
    return reduce_loss(pos_term + neg_term, reduction)
```

#### 3. Focal Loss（nn.Module 写法）

欺诈检测、CTR 预估里绝大多数样本是负样本，模型很快就能把它们分对（$p_t$ 接近 1），但数量巨大，加起来仍然主导 loss 和梯度。Focal loss 给每个样本乘 $(1-p_t)^\gamma$，分得越对权重越小，γ = 2 时 $p_t=0.9$ 的样本权重只有 0.01：

$$
\mathrm{FL}=-\alpha_t\,(1-p_t)^{\gamma}\log p_t,\qquad p_t=\begin{cases}p,&y=1\\1-p,&y=0\end{cases},\qquad \alpha_t=\begin{cases}\alpha,&y=1\\1-\alpha,&y=0\end{cases}
$$

$-\log p_t$ 就是逐元素的 BCE，所以复用 `bce_with_logits`，再用 $p_t=e^{-\text{BCE}}$ 拿到 $p_t$。PyTorch 没有内置 focal loss，检查方法是 γ = 0、不加 α 时结果必须等于 BCE。

```python
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: Optional[float] = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha          # 正样本权重，None 表示不加
        self.gamma = gamma          # 聚焦参数，0 时退化成 BCE
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (N,) 未经 sigmoid 的分数；target: (N,) 0/1。'none' 时返回 (N,)，否则标量"""
        ce = bce_with_logits(logits, target, reduction="none")    # 1. (N,) = -log p_t，数值稳定
        p_t = torch.exp(-ce)                                      # 2. (N,) 模型给真实类别的概率
        loss = (1 - p_t) ** self.gamma * ce                       # 3. 分得越对，权重越小
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)   # 4. (N,)
            loss = alpha_t * loss
        return reduce_loss(loss, self.reduction)
```

#### 4. InfoNCE / CLIP 对比损失

一个 batch 里有 $N$ 对（图, 文），第 $i$ 张图的正样本是第 $i$ 段文字，其余 $N-1$ 段都当负样本（in-batch negatives），所以对比学习通常要大 batch。L2 归一化后两两点积再除以温度 τ，每一行就是一个 $N$ 分类问题，答案在对角线上，所以 label 是 `arange(N)`；按列算就是对转置再调一次。τ 越小分布越尖，梯度越集中在最难的负样本上（CLIP 把它做成可学习参数，初始 0.07）：

$$
s_{ij}=\frac{\hat u_i^\top\hat v_j}{\tau},\qquad \mathcal L_{\text{img}}=\frac1N\sum_{i}-\log\frac{e^{s_{ii}}}{\sum_{j}e^{s_{ij}}},\qquad \mathcal L=\frac12\big(\mathcal L_{\text{img}}+\mathcal L_{\text{txt}}\big)
$$

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
```

PyTorch 没有内置的 CLIP loss，自测用能手算的情况对拍：4 个正交单位向量、τ = 0.5 时，对角线相似度是 2、其余是 0，每行 loss 都是 $\log(1+3e^{-2})$。

#### 5. 知识蒸馏的 KL Loss

两边的 logits 都除以温度 $T$ 再 softmax，$T>1$ 让分布变平，露出「这张 3 有点像 8」这类类别间的信息。loss 是老师分布到学生分布的 KL 散度，再乘 $T^2$：

$$
\mathcal L_{\text{KD}}=T^2\cdot\frac1N\sum_{i}\sum_{k}p^{t}_{ik}\big(\log p^{t}_{ik}-\log p^{s}_{ik}\big),\qquad p^{t}=\operatorname{softmax}(z^{t}/T),\quad p^{s}=\operatorname{softmax}(z^{s}/T)
$$

训练时再和硬标签加权：$\alpha\,\mathcal L_{\text{KD}}+(1-\alpha)\,\mathrm{CE}(z^{s},y)$，老师的 logits 要 `detach()`。内置写法 `F.kl_div(input, target, reduction="batchmean")` 里 `input` 是**学生的 log 概率**，`target` 是**老师的概率**，顺序和直觉相反，最容易写反；`reduction="mean"` 会再除以类别数 $C$。

```python
def distillation_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         temperature: float = 4.0) -> torch.Tensor:
    """student_logits, teacher_logits: (N, C)。返回 T^2 * KL(p_teacher || p_student)，按样本平均"""
    teacher_logits = teacher_logits.detach()                        # 老师不更新
    log_p_s = F.log_softmax(student_logits / temperature, dim=-1)   # 1. (N, C)
    log_p_t = F.log_softmax(teacher_logits / temperature, dim=-1)   # 2. (N, C) 老师也取 log，避免 0 * log 0
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)          # 3. (N,) 每个样本的 KL
    return kl.mean() * temperature ** 2                             # 4. 按样本平均，再乘 T^2
```

回归的 MSE 就是 `reduce_loss((pred - target) ** 2, reduction)`，前面先 `assert pred.shape == target.shape`，防 `(N, 1)` 减 `(N,)` 广播成 `(N, N)`；Huber 在 $|d|<\delta$ 时取 $\frac12 d^2$，否则取 $\delta(|d|-\frac12\delta)$，离群点的梯度被限制在 ±δ 以内。

### 自测

```python
if __name__ == "__main__":
    torch.manual_seed(42)
    z, t = torch.randn(6, 5), torch.tensor([1, -100, 3, -100, 0, 2])     # (N, C), (N,)，两个位置被忽略
    assert check(lambda a: cross_entropy_loss(a, t, label_smoothing=0.1),
                 lambda a: F.cross_entropy(a, t, label_smoothing=0.1), z)
    assert abs(cross_entropy_from_logits(z.numpy(), t.clamp(min=0).numpy()) - F.cross_entropy(z, t.clamp(min=0)).item()) < 1e-5
    x, y, pw = torch.randn(6, 4) * 4, torch.randint(0, 2, (6, 4)).float(), torch.rand(4) * 5
    x[0] = 0.0                                                              # 第 0 行全 0：查 x = 0 处的梯度
    assert check(lambda a: bce_with_logits(a, y, pw), lambda a: F.binary_cross_entropy_with_logits(a, y, pos_weight=pw), x)
    assert check(lambda a: BinaryFocalLoss(None, 0.0)(a, y), lambda a: F.binary_cross_entropy_with_logits(a, y), x)
    s, tea = torch.randn(8, 10), torch.randn(8, 10) * 3                     # 学生、老师的 logits，(N, C)
    assert check(lambda a: distillation_kl_loss(a, tea, 4.0), lambda a: 16 * F.kl_div(
        F.log_softmax(a / 4, dim=-1), F.softmax(tea / 4, dim=-1), reduction="batchmean"), s)
    assert abs(clip_loss(torch.eye(4), torch.eye(4), 0.5).item() - math.log(1 + 3 * math.exp(-2))) < 1e-6
    print("all tests passed")
```

### 关键追问

- **为什么 loss 吃 logits，不吃概率？** 先 softmax 再 log，概率下溢成 0 时 loss 是 `inf`、梯度是 `nan`，logsumexp 和 softplus 不会。模型最后接了 `nn.Softmax` 再喂给 `F.cross_entropy` 等于做两次 softmax：不报错，但 loss 降不到 0，梯度也被压扁。
- **有 ignore_index 时 mean 除以什么？** 除以有效位置数。梯度累积（10.1 节）时各 micro-batch 的有效 token 数不同，各自取 mean 再平均有偏，严格写法是用 `reduction="sum"`，最后除以总有效 token 数。
- **类别不均衡怎么处理？** 多分类给 `F.cross_entropy` 传 `weight`（形状 `(C,)`），二分类用 `pos_weight`，简单负样本特别多（欺诈、CTR）时用 focal loss。focal loss 默认 γ = 2、α = 0.25，α 小于 0.5 是因为大量简单负样本已经被 γ 压掉了。
- **Label smoothing 的直觉？** one-hot 目标要求真实类别概率等于 1，logits 只能无限拉大，模型越来越过度自信；软目标的最优 logits 有限，校准和泛化通常更好。副作用是 loss 降不到 0；也有定义把 ε 分给其余 $C-1$ 类，面试时先问清楚。
- **蒸馏为什么乘 T²？** 不乘时梯度约按 $1/T^2$ 缩小：链式法则从 $z/T$ 带出一个 $1/T$，两个分布变平后 $p^{s}-p^{t}$ 又约正比于 $1/T$。乘 $T^2$ 后梯度量级基本不随 $T$ 变，改 $T$ 时不用重调和 CE 的权重 α。

---

## 10. 优化器：SGD、Momentum、Adam

训练就是下山：梯度 $g_t=\nabla_\theta\mathcal L$ 指向上坡最陡的方向，每步往反方向走一小步，SGD 写成 $\theta_{t+1}=\theta_t-\eta\,g_t$。S（Stochastic）指 $g_t$ 只用一个 mini-batch 算，是全量梯度的带噪声估计。

Momentum 给参数加上惯性。速度 $v$ 展开后是 $g_t+\beta g_{t-1}+\beta^2 g_{t-2}+\cdots$，即过去梯度的指数加权和，$\beta$ 通常取 0.9：

$$
v_{t+1}=\beta\,v_t+g_t,\qquad
\theta_{t+1}=\theta_t-\eta\,v_{t+1}
$$

Adam 等于 Momentum（一阶矩 $m$，管方向）加 RMSProp（二阶矩 $v$ 是梯度平方的滑动平均，给每个参数单独定步长），再加偏差修正。这里的 $v$ 和 Momentum 的速度含义不同：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)\,g_t,\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^2
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

**为什么 Momentum 能加速？** 在又窄又长的山谷里（各方向曲率差很多），横跨山谷的梯度每步换号，SGD 在两侧来回弹；沿谷底的梯度小但方向一致，SGD 走得慢。加权求和让换号的分量互相抵消、一致的分量累加，梯度恒为 $g$ 时 $v\to g/(1-\beta)$，有效步长放大到 $\eta/(1-\beta)$，$\beta=0.9$ 时是 10 倍。在 $f(x,y)=\tfrac12(x^2+100y^2)$ 上从 $(10,1)$ 出发、$\eta=0.018$，GD 要 508 步才离最低点不到 $10^{-3}$，加上 $\beta=0.9$ 的 Momentum 只要 138 步。

**什么时候会震荡？** 在 $f(x)=\tfrac12\lambda x^2$ 上 GD 是 $x_{t+1}=(1-\eta\lambda)x_t$：$\eta\lambda$ 在 1 到 2 之间每步换号来回跳，超过 2 发散，所以学习率上限 $2/\lambda_{\max}$ 由最陡方向决定。另外两种：$\beta$ 太大时惯性太强，冲过最低点来回摆；mini-batch 噪声让参数在最低点附近抖，靠学习率衰减或加大 batch 解决。

### 实现

仿 PyTorch 接口：构造时传入参数数组的列表，`step(grads)` 原地更新。

```python
from typing import List, Tuple

import numpy as np


class SGD:
    def __init__(self, params: List[np.ndarray], lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        """params: 参数数组的列表（形状任意），step 里原地修改"""
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p) for p in params]   # 每个参数一个速度，形状相同

    def step(self, grads: List[np.ndarray]) -> None:
        """grads: 和 params 一一对应，形状相同"""
        for p, g, v in zip(self.params, grads, self.velocities):
            if self.weight_decay > 0:
                g = g + self.weight_decay * p   # 1. L2 正则的梯度是 λw（新数组，不改调用方的 grad）
            if self.momentum > 0:
                v *= self.momentum              # 2. 衰减旧速度，原地改 self.velocities 才记得住
                v += g                          #    再加上当前梯度
                g = v
            p -= self.lr * g                    # 3. 原地更新：模型持有的是同一个数组


class Adam:
    def __init__(self, params: List[np.ndarray], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]   # 一阶矩：管方向
        self.v = [np.zeros_like(p) for p in params]   # 二阶矩：管每个参数的步长
        self.t = 0                                    # 步数，偏差修正要用

    def step(self, grads: List[np.ndarray]) -> None:
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1                           # 1. m = β1·m + (1-β1)·g，原地
            m += (1 - self.beta1) * g
            v *= self.beta2                           # 2. v = β2·v + (1-β2)·g²，原地
            v += (1 - self.beta2) * g * g
            m_hat = m / (1 - self.beta1 ** self.t)    # 3. 偏差修正：m、v 从 0 起步，前几步偏小
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)   # 4. 原地更新
```

### 自测

和 `torch.optim.SGD`、`torch.optim.Adam` 对拍：同样的初值、同样的 5 步梯度，参数应该一致。

```python
import numpy as np
import torch

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cases = [(SGD, torch.optim.SGD, {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.01}),
             (Adam, torch.optim.Adam, {"lr": 0.1})]
    for ours_cls, torch_cls, kwargs in cases:
        w = rng.standard_normal((3, 4))                # 模型持有的参数数组
        w_t = torch.tensor(w, requires_grad=True)      # torch.tensor 会拷贝，两边互不影响
        opt, opt_t = ours_cls([w], **kwargs), torch_cls([w_t], **kwargs)
        for _ in range(5):
            g = rng.standard_normal((3, 4))
            opt.step([g])                              # 原地改 w
            w_t.grad = torch.tensor(g)
            opt_t.step()
        assert np.allclose(w, w_t.detach().numpy())
    print("all tests passed")
```

### 关键追问

- **为什么 `v *= momentum` 和 `p -= ...` 必须原地写？** 写成 `v = momentum * v + g` 会新建数组，`self.velocities` 里存的永远是 0，Momentum 悄悄退化成 SGD 而且不报错。`p` 同理，不原地改，模型持有的那个数组根本没变。
- **偏差修正修的是什么？** $m$、$v$ 从 0 起步，前几步被拉向 0：第一步 $m_1=0.1\,g_1$，除以 $1-\beta_1=0.1$ 正好还原成 $g_1$。不修正时第一步的更新量约是修正后的 3.16 倍（$0.1/\sqrt{0.001}$）。
- **SGD 和 Adam 怎么选？** Adam 给每个参数自适应步长，对学习率不太敏感、收敛快，Transformer 和 LLM 基本都用 AdamW。SGD + Momentum 有时泛化更好，而且省显存：Adam 每个参数多存 $m$、$v$ 两份状态，Momentum 只多存一份速度。
- **Nesterov 有什么不同？** 先按惯性往前看一步，在预估的位置算梯度，冲过头时能更早刹车；PyTorch 的写法把更新量从 $v$ 换成 $g+\beta v$，不用多算一次梯度。
- **AdamW 改了什么？** Adam 把 L2 加进梯度后，正则项也被 $\sqrt{\hat v}$ 除掉，梯度大的参数衰减得少；AdamW 把 weight decay 从梯度里拿出来，单独做一步 `p -= lr * wd * p`。

---

## 10.1 PyTorch Training-Validation Loop

每个 epoch 分两半。**训练**：遍历训练集，每个 batch 做五步，清梯度、前向、算 loss、反向、更新。**验证**：在 `model.eval()` 和 `torch.no_grad()` 下只做前向，算 loss 和 accuracy，用来挑模型和 early stopping，所以最终效果要在另一份测试集上报告。五步和第 11 节的 NumPy MLP 一一对应：`loss.backward()` 代替手写的逐层 backward，`optimizer.step()` 代替第 10 节的 `opt.step(grads)`。

### 实现

```python
import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                                 # (B, in_dim) -> (B, num_classes)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: str = "cpu") -> float:
    """
    Args:
        loader: 每次给出 xb (B, D) 和 yb (B,)
    Returns:
        这个 epoch 按样本数加权的平均训练 loss
    """
    model.train()                                          # 打开 Dropout，BatchNorm 用 batch 统计量
    total_loss, total_count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)              # (B, D), (B,)
        optimizer.zero_grad()                              # 1. 清梯度：.grad 默认累加
        logits = model(xb)                                 # 2. 前向 (B, C)
        loss = criterion(logits, yb)                       # 3. 标量，batch 内平均
        loss.backward()                                    # 4. 反向，梯度写进每个参数的 .grad
        # 梯度裁剪放在这里（backward 之后、step 之前）：nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                                   # 5. 用 .grad 更新参数
        total_loss += loss.item() * xb.size(0)             # 取 Python float；最后一批可能更小，按样本数加权
        total_count += xb.size(0)
    return total_loss / total_count


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: str = "cpu") -> Tuple[float, float]:
    """不更新参数。Returns: (按样本数加权的平均 loss, accuracy)"""
    model.eval()                                           # 关 Dropout，BatchNorm 用 running 统计量
    total_loss, correct, total_count = 0.0, 0, 0
    with torch.no_grad():                                  # 不建计算图，省内存
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)                             # (B, C)
            total_loss += criterion(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total_count += xb.size(0)
    return total_loss / total_count, correct / total_count


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        criterion: nn.Module, optimizer: torch.optim.Optimizer, epochs: int = 100,
        patience: int = 10, scheduler: Optional[Any] = None, device: str = "cpu",
        verbose: bool = True) -> Dict[str, List[float]]:
    """
    val loss 连续 patience 个 epoch 没变好就停，最后恢复最好那个 epoch 的权重。
    scheduler: 按 epoch 调的（StepLR 等）；ReduceLROnPlateau 要改成 scheduler.step(val_loss)
    Returns:
        history: 每个 epoch 的 train_loss、val_loss、val_acc
    """
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss, bad_epochs = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if verbose:
            print(f"Epoch {epoch + 1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.3f}")
        if scheduler is not None:
            scheduler.step()                               # 1. 在这个 epoch 所有 optimizer.step() 之后
        if val_loss < best_val_loss:                       # 2. early stopping：变好就存一份权重
            best_val_loss, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())   # state_dict() 和参数共享内存，必须深拷贝
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)                      # 3. 恢复最好的权重
    return history


def make_loaders(n_train: int = 1000, n_val: int = 300, noise: float = 0.1,
                 batch_size: int = 64, seed: int = 0) -> Tuple[DataLoader, DataLoader]:
    """合成数据：二维点，一三象限为 1；noise 比例的标签翻转；两个特征尺度差很多"""
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_val
    z = torch.randn(n, 2, generator=g)                     # (n, 2)
    y = (z[:, 0] * z[:, 1] > 0).long()                     # (n,) 类别标签必须是 long
    flip = torch.rand(n, generator=g) < noise              # 标签噪声，early stopping 才有事做
    y[flip] = 1 - y[flip]
    X = z * torch.tensor([10.0, 0.1]) + torch.tensor([50.0, -3.0])   # (n, 2) 故意拉开尺度
    mean, std = X[:n_train].mean(dim=0), X[:n_train].std(dim=0)      # 只用训练集算，防止验证集信息泄露
    X = (X - mean) / std
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], y[n_train:])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),    # 训练集打乱
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))
```

### 自测

```python
def test_training_loop() -> None:
    torch.manual_seed(42)
    train_loader, val_loader = make_loaders()              # 验证集 300 条，最后一批 44 条
    model, criterion = MLP(2, 64, 2), nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    hist = fit(model, train_loader, val_loader, criterion, optimizer, patience=8, scheduler=scheduler, verbose=False)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    best = hist["val_loss"].index(min(hist["val_loss"]))
    assert abs(val_loss - hist["val_loss"][best]) < 1e-7   # 1. 恢复的是最好那个 epoch 的权重
    assert len(hist["val_loss"]) == best + 1 + 8           # 2. 之后连续 8 个 epoch 没变好就停
    X_val, y_val = val_loader.dataset.tensors              # (300, 2), (300,)
    with torch.no_grad():                                  # 3. 按样本数加权 = 整个验证集一次算
        assert abs(val_loss - criterion(model(X_val), y_val).item()) < 1e-5
    assert val_acc > 0.8 and evaluate(model, val_loader, criterion) == (val_loss, val_acc)   # 4. eval 下没有随机性
    print(f"stopped at epoch {len(hist['val_loss'])}, best epoch {best + 1}, val acc {val_acc:.3f}")
    print("all tests passed")


if __name__ == "__main__":
    test_training_loop()
# stopped at epoch 34, best epoch 26, val acc 0.913
# all tests passed
```

### 关键追问

- **`model.eval()` 和 `torch.no_grad()` 有什么区别？** `eval()` 改变层的计算：Dropout 不丢神经元，BatchNorm 用 running 统计量（11.1 节）；`no_grad()` 只是不建计算图，省内存和时间，不会关掉 Dropout。验证两个都要写：只写 `no_grad()` 结果每次不一样，只写 `eval()` 结果对但白建了计算图。
- **忘了 `optimizer.zero_grad()` 会怎样？** `.grad` 默认累加，每一步的梯度都混进之前所有 batch 的梯度，越攒越大、方向被带偏，不报错但 loss 不降。放在上一次 `step()` 之后、这一次 `backward()` 之前都行，习惯写在每个 batch 开头。
- **为什么累加 `loss.item() * xb.size(0)`？** 直接累加 `loss` 张量会把每个 batch 的计算图都留在内存里（基础篇 0.3 节）。乘样本数是因为最后一批更小，对每个 batch 的平均再求平均会给它过大的权重，自测已对拍加权平均等于整个验证集一次算的 loss。
- **`scheduler.step()` 放在哪？** 放在 `optimizer.step()` 之后，写反了会有 UserWarning，而且第一个学习率被跳过。`StepLR`、`CosineAnnealingLR` 每个 epoch 调一次；`OneCycleLR` 和 warmup 每个 batch 调一次，要放进 batch 循环。
- **显存只够小 batch 怎么办？** 梯度累积：每个小 batch 的 loss 先除以 `accum_steps` 再 `backward()`，中间不清梯度，攒够 `accum_steps` 次才 `step()` 和 `zero_grad()`，梯度等于大 batch 的平均梯度（有 BatchNorm 时不完全等价）。

---

## 11. 手写神经网络层：Linear、ReLU、Softmax、Dropout 与两层 MLP

反向传播就是链式法则。每层只写两个函数：`forward(x)` 算输出，并缓存 backward 要用的东西；`backward(dout)` 收到上游梯度 $\partial\mathcal L/\partial\text{out}$，算出参数梯度，再返回 $\partial\mathcal L/\partial x$ 往前传。

写 backward 最有用的一条规则：**梯度和它对应的变量形状完全相同。** $W$ 是 $(D_{in},D_{out})$，$dW$ 也必须是 $(D_{in},D_{out})$。很多时候把形状凑对就能写出正确的矩阵乘法，面试时边写边把每一步的形状念出来。

### Linear

$$
Y=XW+b,\qquad dX=dY\,W^\top,\qquad dW=X^\top dY,\qquad db=\sum_{i=1}^{N}dY_{i,:}
$$

形状：$X:(N,D_{in})$，$W:(D_{in},D_{out})$，$b:(D_{out},)$，$Y$ 和 $dY$ 都是 $(N,D_{out})$。用凑形状检查：$dW$ 要 $(D_{in},D_{out})$，手上只有 $X$ 和 $dY$，唯一能凑出来的是 $X^\top dY$。$b$ 在前向被广播到 $N$ 行，反向就把 $N$ 行的梯度加起来：**前向广播，反向求和。**

```python
from typing import Callable, List, Optional, Tuple

import numpy as np


def linear_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, tuple]:
    """x: (N, D_in)，W: (D_in, D_out)，b: (D_out,)。返回 out (N, D_out) 和 cache"""
    out = x @ W + b                  # (N, D_in) @ (D_in, D_out) + (D_out,) -> (N, D_out)
    return out, (x, W)               # backward 用 x 算 dW，用 W 算 dx


def linear_backward(dout: np.ndarray, cache: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dout: (N, D_out)。返回 dx (N, D_in)，dW (D_in, D_out)，db (D_out,)"""
    x, W = cache
    dx = dout @ W.T                  # (N, D_out) @ (D_out, D_in) -> (N, D_in)
    dW = x.T @ dout                  # (D_in, N) @ (N, D_out) -> (D_in, D_out)
    db = dout.sum(axis=0)            # (N, D_out) -> (D_out,)：前向广播，反向求和
    return dx, dW, db
```

### ReLU

前向被截成 0 的位置，反向梯度也是 0，其余位置原样传回。

```python
def relu_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.maximum(0, x), x       # 缓存输入：backward 要知道哪些位置 > 0


def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    return dout * (x > 0)            # (N, D) -> (N, D)
```

### Softmax 单独的反向

softmax 后面接的不一定是交叉熵（注意力里接的是乘 $V$），这时它是一个独立的层，要对任意上游梯度 $dy$ 算 $dx$。Jacobian 和 vector-Jacobian product：

$$
\frac{\partial y_i}{\partial x_j}=y_i(\delta_{ij}-y_j),\qquad J=\operatorname{diag}(y)-yy^\top,\qquad dx=y\odot\Big(dy-\sum_i dy_i\,y_i\Big)
$$

括号里的求和是 $dy$ 和 $y$ 的逐行点积。不要把 $J$ 建出来：一个 batch 的 $J$ 是 $(N,C,C)$，上面的公式只要 $O(NC)$。backward 只用前向输出 $y$，`axis=-1` 加 `keepdims=True` 让注意力的 `(B, H, T, T)` 也能直接用。

```python
def softmax_forward(x: np.ndarray) -> np.ndarray:
    """x: (N, C) -> y: (N, C)，每行和为 1"""
    e = np.exp(x - x.max(axis=-1, keepdims=True))    # 1. (N, C) 减每行最大值，防上溢
    return e / e.sum(axis=-1, keepdims=True)         # 2. (N, C) / (N, 1)


def softmax_backward(dy: np.ndarray, y: np.ndarray) -> np.ndarray:
    """dy: (N, C) 上游梯度；y: (N, C) 前向输出。返回 dx: (N, C)"""
    dot = (dy * y).sum(axis=-1, keepdims=True)       # 1. (N, 1) 每行的 Σ dy_i y_i
    return y * (dy - dot)                            # 2. (N, C)
```

### Softmax + Cross-Entropy 的反向

两者合在一起时梯度最简单（loss 的数值稳定写法见第 9 节）：

$$
\frac{\partial\mathcal L}{\partial z}=\frac{1}{N}\big(p-\text{onehot}(y)\big),\qquad p=\operatorname{softmax}(z)
$$

推导：单个样本 $\mathcal L=-z_y+\log\sum_j e^{z_j}$，对 $z_k$ 求导，第一项只在 $k=y$ 时给 $-1$，第二项正好是 $p_k$；batch 取了平均，再除以 $N$。

```python
def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """logits: (N, C) 未经 softmax；y: (N,) 整数标签。返回平均 loss 和 dlogits (N, C)"""
    n = logits.shape[0]
    shifted = logits - logits.max(axis=1, keepdims=True)                      # (N, C) 稳定化
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))  # (N, C) log-softmax
    loss = float(-log_probs[np.arange(n), y].mean())
    dlogits = np.exp(log_probs)          # 1. (N, C) 概率 p；exp 生成新数组，可以原地改
    dlogits[np.arange(n), y] -= 1        # 2. 真实类别那一列减 1：p - onehot
    return loss, dlogits / n             # 3. loss 对 batch 取了平均，梯度也除以 N
```

### Dropout（inverted）

先问清 $p$ 的含义：PyTorch 的 `nn.Dropout(p)` 里 $p$ 是**丢弃**概率，本节也用这个约定；原论文和 TF1 的 `keep_prob` 是保留概率。PyTorch 用 inverted dropout，训练时把留下的激活放大 $1/(1-p)$，推理时什么都不做：

$$
m\sim\operatorname{Bernoulli}(1-p),\qquad \text{out}_{\text{train}}=\frac{m\odot x}{1-p},\qquad \text{out}_{\text{eval}}=x,\qquad dx=dout\odot\frac{m}{1-p}
$$

放大是为了保持期望：$\operatorname{E}[\text{out}_i]=(1-p)\cdot x_i/(1-p)=x_i$，所以推理时 dropout 就是恒等。backward 必须用前向那一次的 mask，被丢掉的位置梯度为 0。MLP 里放在激活之后：Linear → ReLU → Dropout → Linear。

```python
def dropout_forward(x: np.ndarray, p: float, training: bool,
                    rng: np.random.Generator) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """x: (N, D)；p: 丢弃概率，0 <= p < 1。返回 out (N, D) 和缩放后的 mask"""
    if not training or p == 0.0:
        return x, None                                 # 推理：恒等
    mask = (rng.random(x.shape) >= p) / (1.0 - p)      # 1. (N, D)，值只有 0 和 1/(1-p)
    return x * mask, mask                              # 2. 缓存 mask，backward 必须用同一个


def dropout_backward(dout: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    return dout if mask is None else dout * mask       # 前向乘了什么，反向就乘什么
```

### 拼起来：两层 MLP + mini-batch SGD

Linear → ReLU → Linear → Softmax CE。前向从左往右，反向从右往左，每层把梯度交给前一层。参数更新就是第 10 节的 Momentum，这里内联三行，让本节单独能跑。

```python
def init_mlp(d_in: int, hidden: int, num_classes: int, rng: np.random.Generator) -> List[np.ndarray]:
    """He 初始化（配合 ReLU，方差 2 / fan_in）。返回 [W1, b1, W2, b2]"""
    return [rng.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in), np.zeros(hidden),
            rng.standard_normal((hidden, num_classes)) * np.sqrt(2.0 / hidden), np.zeros(num_classes)]


def mlp_loss_and_grads(params: List[np.ndarray], x: np.ndarray,
                       y: np.ndarray) -> Tuple[float, List[np.ndarray]]:
    """x: (B, D)，y: (B,)。返回 loss 和与 params 一一对应的梯度"""
    W1, b1, W2, b2 = params
    # 1. 前向：(B, D) -> (B, hidden) -> (B, hidden) -> (B, C)
    h, cache1 = linear_forward(x, W1, b1)
    a, relu_cache = relu_forward(h)
    logits, cache2 = linear_forward(a, W2, b2)
    loss, dlogits = softmax_cross_entropy(logits, y)
    # 2. 反向：倒着走一遍
    da, dW2, db2 = linear_backward(dlogits, cache2)
    dh = relu_backward(da, relu_cache)
    _, dW1, db1 = linear_backward(dh, cache1)
    return loss, [dW1, db1, dW2, db2]


def train_mlp(x: np.ndarray, y: np.ndarray, hidden: int = 16, num_classes: int = 2,
              epochs: int = 100, batch_size: int = 32, lr: float = 0.1,
              momentum: float = 0.9, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    params = init_mlp(x.shape[1], hidden, num_classes, rng)
    velocities = [np.zeros_like(p) for p in params]
    for _ in range(epochs):
        perm = rng.permutation(len(x))                       # 1. 每个 epoch 打乱一次
        for start in range(0, len(x), batch_size):
            idx = perm[start:start + batch_size]             # 2. 取一个 mini-batch
            _, grads = mlp_loss_and_grads(params, x[idx], y[idx])
            for p, g, v in zip(params, grads, velocities):   # 3. Momentum，原地更新
                v[:] = momentum * v + g
                p -= lr * v
    return params


rng = np.random.default_rng(42)
X = rng.standard_normal((400, 2))
y = (X[:, 0] * X[:, 1] > 0).astype(int)      # 一三象限为 1，二四象限为 0：一条直线分不开
W1, b1, W2, b2 = train_mlp(X, y)
logits = np.maximum(0, X @ W1 + b1) @ W2 + b2
print("train accuracy:", (logits.argmax(axis=1) == y).mean())   # train accuracy: 0.9925
```

### 梯度检查

backward 写完用数值微分验证，中心差分比单边差分准得多：$\partial f/\partial x\approx(f(x+h)-f(x-h))/(2h)$。float64 下相对误差小于 $10^{-7}$ 基本没问题，大于 $10^{-3}$ 几乎一定有 bug。

```python
def numerical_grad(f: Callable[[], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """f: 无参函数，返回标量 loss；x: 参数数组，会被临时扰动再还原"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        old = x.flat[i]
        x.flat[i] = old + h
        f_plus = f()
        x.flat[i] = old - h
        grad.flat[i] = (f_plus - f()) / (2 * h)
        x.flat[i] = old                                  # 一定要还原
    return grad


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))
```

### 自测

对 MLP 的 `dW1` 做一次梯度检查，就同时验证了 Linear、ReLU、softmax CE 三个 backward。

```python
def run_layer_tests() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((6, 2)), np.array([0, 1, 2, 0, 1, 2])
    params = init_mlp(2, 5, 3, rng)                      # 1. 梯度检查 MLP 的 dW1
    num = numerical_grad(lambda: mlp_loss_and_grads(params, x, y)[0], params[0])
    err = rel_error(mlp_loss_and_grads(params, x, y)[1][0], num)
    print(f"grad check {err:.1e}")                       # grad check 1.7e-10
    assert err < 1e-7
    # 2. 单独的 softmax 反向接上 CE 的上游梯度 -onehot / (N p)，应等于 (p - onehot) / N
    logits = rng.standard_normal((6, 3))
    p, onehot = softmax_forward(logits), np.eye(3)[y]
    assert np.allclose(softmax_backward(-onehot / (6 * p), p), softmax_cross_entropy(logits, y)[1])
    # 3. dropout：推理是恒等；训练时期望不变
    assert np.array_equal(dropout_forward(x, 0.5, False, rng)[0], x)
    assert abs(dropout_forward(np.ones((1000, 100)), 0.3, True, rng)[0].mean() - 1) < 0.01
    print("all tests passed")


if __name__ == "__main__":
    run_layer_tests()
```

### 关键追问

- **为什么 forward 要缓存？** $dW=X^\top dY$ 要用输入，ReLU 反向要知道哪里大于 0，dropout 反向要用同一个 mask。这也是训练比推理费显存的原因：每层的激活都要留到反向用完。
- **最常见的 bug？** softmax CE 的梯度忘了除以 $N$，等于把学习率放大 $N$ 倍。另一个是直接在前向的 `probs` 上原地 `-= 1`，把前向结果也改坏了，要先 `.copy()`。
- **权重怎么初始化？** 不能全 0：同一层的神经元输出和梯度永远相同，等于只有一个神经元（对称性问题），偏置为 0 没关系。配 ReLU 用 He 初始化（方差 $2/D_{in}$，补回 ReLU 截掉一半造成的方差减半），tanh、sigmoid 用 Xavier（方差 $2/(D_{in}+D_{out})$）。
- **softmax 饱和时梯度怎样？** 输出接近 one-hot 时 $J=\operatorname{diag}(y)-yy^\top$ 每一项都接近 0，梯度几乎消失。注意力里除以 $\sqrt{d_k}$，就是为了不让点积太大把 softmax 推进饱和区（第 1 节）。
- **忘了 `model.eval()` 会怎样？** dropout 继续随机丢，同一个输入每次输出不同，指标偏低还会抖。`torch.no_grad()` 只关梯度、不关 dropout，两者要一起用。

---

## 11.1 BatchNorm 与 LayerNorm

两者公式相同：先标准化，再乘可学习的 γ、加 β，区别只在统计量沿哪个轴算。

$$
\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y=\gamma\odot\hat x+\beta
$$

输入 $(N,D)$ 时，BatchNorm 沿 batch 维（`axis=0`，每个特征跨样本算一组 $\mu,\sigma^2$）；LayerNorm 沿特征维（`axis=-1`，每个样本自己算一组），和 batch 大小、序列长度都无关，是 Transformer 的标配。

BatchNorm 最常考**训练和推理不同**：训练用当前 batch 的统计量，同时累积滑动平均 $\text{running}\leftarrow(1-m)\cdot\text{running}+m\cdot\text{batch stat}$（PyTorch 的 momentum $m=0.1$）；推理时 batch 可能只有 1 条，改用 running 统计量。

BN backward（训练模式，$\mu,\sigma^2$ 也是 $x$ 的函数）记住化简后的结论，其中 $d\hat x=dy\odot\gamma$：

$$
dx=\frac{1}{N\sqrt{\sigma^2+\epsilon}}\Big(N\,d\hat x-\sum_i d\hat x_i-\hat x\odot\sum_i d\hat x_i\odot\hat x_i\Big),\qquad d\gamma=\sum_i dy_i\odot\hat x_i,\qquad d\beta=\sum_i dy_i
$$

### 实现

```python
import numpy as np


class BatchNorm1d:
    """x: (N, D)。momentum 是新 batch 统计量的权重（PyTorch 约定）"""

    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma, self.beta = np.ones(dim), np.zeros(dim)            # 初始是恒等变换
        self.running_mean, self.running_var = np.zeros(dim), np.ones(dim)
        self.momentum, self.eps = momentum, eps

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """x: (N, D) -> (N, D)"""
        if training:
            n, m = x.shape[0], self.momentum
            mu, var = x.mean(axis=0), x.var(axis=0)                    # 1. (D,) 沿 batch 维，有偏方差
            # 2. 滑动平均留给推理；PyTorch 的 running_var 用无偏方差（除以 N - 1）
            self.running_mean = (1 - m) * self.running_mean + m * mu
            self.running_var = (1 - m) * self.running_var + m * var * n / (n - 1)
        else:
            mu, var = self.running_mean, self.running_var              # 推理：用累积的统计量
        self.inv_std = 1.0 / np.sqrt(var + self.eps)                   # 3. (D,)
        self.x_hat = (x - mu) * self.inv_std                           # (N, D)，backward 要用
        return self.gamma * self.x_hat + self.beta                     # 4. (N, D)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """只对 training=True 的前向成立。dout: (N, D) -> dx: (N, D)"""
        n = dout.shape[0]
        self.dgamma = (dout * self.x_hat).sum(axis=0)                  # (D,)
        self.dbeta = dout.sum(axis=0)                                  # (D,) 前向广播，反向求和
        dx_hat = dout * self.gamma                                     # (N, D)
        return self.inv_std / n * (n * dx_hat - dx_hat.sum(axis=0)
                                   - self.x_hat * (dx_hat * self.x_hat).sum(axis=0))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """x: (B, T, D)；gamma, beta: (D,)。每个 token 沿特征维单独标准化"""
    mu = x.mean(axis=-1, keepdims=True)                                # (B, T, 1)
    var = x.var(axis=-1, keepdims=True)                                # (B, T, 1)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta               # (B, T, D)


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm：不减均值、没有 beta，只按均方根缩放。x: (B, T, D) -> (B, T, D)"""
    return gamma * x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
```

### 自测

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def run_norm_tests() -> None:
    rng = np.random.default_rng(0)
    x, dout = rng.standard_normal((8, 4)) * 3 + 1, rng.standard_normal((8, 4))   # (N, D)
    bn, ref = BatchNorm1d(4), nn.BatchNorm1d(4).double()
    xt = torch.tensor(x, requires_grad=True)
    out_ref = ref(xt)                                   # 训练模式，同时更新 running 统计量
    out_ref.backward(torch.tensor(dout))
    assert np.allclose(bn.forward(x), out_ref.detach().numpy())   # 1. 训练前向、running_var、backward
    assert np.allclose(bn.running_var, ref.running_var.numpy())
    assert np.allclose(bn.backward(dout), xt.grad.numpy())
    ref.eval()                                          # 2. 推理模式用 running 统计量
    assert np.allclose(bn.forward(x[:2], training=False), ref(torch.tensor(x[:2])).detach().numpy())
    x3 = rng.standard_normal((2, 3, 4))                 # 3. (B, T, D)：LayerNorm 对拍；RMSNorm 输出均方根为 1
    assert np.allclose(layer_norm(x3, np.ones(4), np.zeros(4)), F.layer_norm(torch.tensor(x3), (4,)).numpy())
    assert np.allclose((rms_norm(x3, np.ones(4)) ** 2).mean(axis=-1), 1, atol=1e-4)
    print("all tests passed")


if __name__ == "__main__":
    run_norm_tests()
```

### 对比

|  | BatchNorm | LayerNorm | RMSNorm |
| --- | --- | --- | --- |
| 统计量的轴 | batch 维 `axis=0` | 特征维 `axis=-1` | 特征维，只算均方根 |
| 依赖 batch | 是，batch 小时噪声大 | 否 | 否 |
| 训练和推理不同 | 是：训练用 batch 统计量，推理用 running 统计量 | 否 | 否 |
| 常见位置 | batch 较大的前馈网络 | Transformer（原论文、BERT、GPT） | LLaMA 等 decoder-only LLM |

### 关键追问

- **为什么要 γ 和 β？** 强行标准化会限制表达能力，有了 γ、β，网络可以学回需要的尺度和偏移。$\gamma=\sqrt{\sigma^2+\epsilon}$、$\beta=\mu$ 时完全还原原始输入。
- **忘了 `model.eval()` 会怎样？** BN 继续用当前 batch 的统计量，同一个样本的输出会随 batch 里其他样本变化，还会继续改 running 统计量。`torch.no_grad()` 不切换这个行为。
- **batch 很小时怎么办？** 统计量噪声大，效果明显变差；batch 为 1 时 PyTorch 在训练模式下直接报错。改用 LayerNorm 或 GroupNorm，它们不依赖 batch。
- **Pre-Norm 还是 Post-Norm？** Post-Norm（原论文）是 `LN(x + sublayer(x))`，层数深时训练不稳、依赖 warmup。Pre-Norm 是 `x + sublayer(LN(x))`，残差成为直通的主干，训练更稳，是现代 LLM 的默认选择。

---

## 11.2 手写 LSTM 与参数量

两种考法：不用 `nn.LSTM` 手写前向，以及单独一道「给定结构，算参数量」。每个时间步由 $x_t:(B,D)$ 和 $h_{t-1}:(B,H)$ 一次算出四块预激活 $a_t=x_tW_{ih}^\top+b_{ih}+h_{t-1}W_{hh}^\top+b_{hh}$，其中 $W_{ih}:(4H,D)$，$W_{hh}:(4H,H)$，$a_t:(B,4H)$。按列切开，顺序 i、f、g、o（和 PyTorch 一致）：

$$
i=\sigma(a_t^{(1)}),\quad f=\sigma(a_t^{(2)}),\quad g=\tanh(a_t^{(3)}),\quad o=\sigma(a_t^{(4)}),\qquad c_t=f\odot c_{t-1}+i\odot g,\qquad h_t=o\odot\tanh(c_t)
$$

遗忘门 f 决定旧记忆留多少，输入门 i 决定写入多少候选内容 g，输出门 o 决定读出多少；三个门用 sigmoid 取 0 到 1 的比例，g 用 tanh 让写入内容可正可负。cell state $c$ 是只做加法的传送带，hidden state $h$ 是每步对外的输出。

### 实现

```python
from typing import Optional, Tuple

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """单层单向 LSTM，batch_first=True。参数布局和 nn.LSTM 一致，门的顺序 i, f, g, o"""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)     # 对应 weight_ih_l0 (4H, D)、bias_ih_l0
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)    # 对应 weight_hh_l0 (4H, H)、bias_hh_l0

    def forward(self, x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: (B, T, D)；state: (h0, c0) 各 (B, H)，默认全 0。返回 output (B, T, H) 和 (h_n, c_n)"""
        batch_size, seq_len, _ = x.size()
        zeros = x.new_zeros(batch_size, self.hidden_size)
        h, c = state if state is not None else (zeros, zeros)  # (B, H), (B, H)
        x_proj = self.x2h(x)                                   # 1. (B, T, 4H) 输入投影和 h 无关，一次算完
        outputs = []
        for t in range(seq_len):                               # 2. h_t 依赖 h_{t-1}，只能按时间串行
            gates = x_proj[:, t] + self.h2h(h)                 # (B, 4H)
            i, f, g, o = gates.chunk(4, dim=1)                 # 各 (B, H)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            c = f * c + i * torch.tanh(g)                      # 3. (B, H) 擦掉一部分旧的，加上一部分新的
            h = o * torch.tanh(c)                              # (B, H)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)             # (B, T, H)，(B, H)，(B, H)
```

反向交给 autograd 沿时间倒着传（BPTT）。`h2h.weight` 每一步都用到，梯度是所有时间步的和，和第 11 节「前向广播，反向求和」同理。

### 参数量

每层每个方向有 k 组变换（LSTM 的 k = 4），每组是从 $[x_t,h_{t-1}]$ 到 $H$ 维的线性层。PyTorch 有 `b_ih`、`b_hh` 两组 bias，前向里直接相加，数学上冗余但确实占参数；面试先问清约定，没说就两个都报。

$$
P_{\text{教科书}}=4\,\big(H(D_{in}+H)+H\big),\qquad P_{\text{PyTorch}}=4\,\big(H(D_{in}+H)+2H\big)
$$

- **堆叠**：第 1 层 $D_{in}=D$；之后各层吃上一层的输出，单向 $D_{in}=H$，双向 $D_{in}=2H$（两个方向拼起来）。
- **双向**：每层正、反两套独立参数，每层乘 2。**换模型只换 k**：LSTM 4、GRU 3、vanilla RNN 1。

| 算例（D = 100，H = 256，PyTorch 约定） | 计算 | 参数量 |
| --- | --- | --- |
| LSTM，1 层单向 | 4 × (256 × 356 + 2 × 256) | 366,592 |
| 同上，只算一组 bias | 4 × (256 × 356 + 256) | 365,568 |
| GRU，1 层单向 | 3 × (256 × 356 + 2 × 256) | 274,944 |
| LSTM，2 层双向 | 第 1 层 2 × 366,592；第 2 层 2 × 4 × (256 × 768 + 512) | 2,310,144 |

```python
def rnn_param_count(input_size: int, hidden_size: int, num_layers: int = 1,
                    bidirectional: bool = False, num_gates: int = 4, num_biases: int = 2) -> int:
    """num_gates: LSTM 4、GRU 3、RNN 1；num_biases: PyTorch 2，教科书 1，bias=False 为 0"""
    num_directions = 2 if bidirectional else 1
    total = 0
    for layer in range(num_layers):
        layer_input = input_size if layer == 0 else hidden_size * num_directions   # 1. 本层输入维度
        weights = num_gates * hidden_size * (layer_input + hidden_size)            # 2. W_ih 和 W_hh
        biases = num_gates * hidden_size * num_biases                              # 3. b_ih 和 b_hh
        total += num_directions * (weights + biases)
    return total


def solution(input_size: int, hidden_size: int, num_layers: int, bidirectional: bool) -> int:
    """CodeSignal 单函数题入口：nn.LSTM 的参数量（PyTorch 约定，两组 bias），不 import 任何库"""
    return rnn_param_count(input_size, hidden_size, num_layers, bidirectional)
```

### 自测

```python
def run_lstm_tests() -> None:
    torch.manual_seed(42)
    model, ref = LSTM(5, 8), nn.LSTM(5, 8, batch_first=True)          # D = 5，H = 8
    ref.load_state_dict({"weight_ih_l0": model.x2h.weight, "bias_ih_l0": model.x2h.bias,
                         "weight_hh_l0": model.h2h.weight, "bias_hh_l0": model.h2h.bias})
    x = torch.randn(3, 7, 5)                                          # (B, T, D)
    (out, (h_n, c_n)), (ref_out, (_, ref_c)) = model(x), ref(x)       # ref 的状态是 (1, B, H)
    # 1. 拷贝权重后，形状和数值都和 nn.LSTM 一致
    assert out.shape == (3, 7, 8) and h_n.shape == (3, 8)
    assert torch.allclose(out, ref_out, atol=1e-6) and torch.allclose(c_n, ref_c[0], atol=1e-6)
    # 2. 参数量：solution 和 nn.LSTM（多层、双向）一致
    for d, h, layers, bi in [(100, 256, 1, False), (100, 256, 2, True), (7, 5, 3, True)]:
        lstm = nn.LSTM(d, h, num_layers=layers, bidirectional=bi)
        assert solution(d, h, layers, bi) == sum(p.numel() for p in lstm.parameters())
    print("all tests passed")


if __name__ == "__main__":
    run_lstm_tests()
```

### 关键追问

- **LSTM 为什么能缓解梯度消失？** 沿 cell 这条路是加法，$\partial c_t/\partial c_{t-1}=f$，f 接近 1 时梯度几乎无损地传回很早的时间步。vanilla RNN 每步都要乘 $W_h$ 和 tanh 的导数，连乘很快衰减或爆炸。
- **为什么常把遗忘门的 bias 初始化成 1？** $b_f=0$ 时 f 约 0.5，沿 cell 往回 20 步梯度只剩 $0.5^{20}\approx9.5\times10^{-7}$；$b_f=1$ 时 $\sigma(1)\approx0.731$，20 步后约 $1.9\times10^{-3}$。PyTorch 里遗忘门是 bias 的第二块 `[H:2H]`，两组 bias 相加才是有效值，只改一组即可。
- **为什么不能沿时间并行？** $h_t$ 依赖 $h_{t-1}$，T 步必须串行，只能并行 batch 和 H 两维；实现里把 `x2h` 提到循环外，就是把能并行的输入投影一次做完。Attention 的每个位置直接看所有位置，一次矩阵乘法算完，代价是 $O(T^2)$ 的计算和显存。
- **h 和 c 分别干什么？序列分类取哪个？** h 是对外的输出（`output` 就是每一步的 h），c 是只在步与步之间传递的内部记忆。分类取最后一层的 `h_n[-1]`；双向时拼 `h_n[-2]` 和 `h_n[-1]`，因为 `output[:, -1]` 里只有正向的最终状态。

---

## 12. 手写 N-gram 语言模型

语言模型给一句话打分：链式法则把句子概率拆成每个词在前文下的条件概率之积，Markov 假设让第 $t$ 个词只看前 $n-1$ 个词（$n=2$ 是 bigram）。条件概率直接数出来（MLE），$h$ 是前 $n-1$ 个词，$c(h)=\sum_w c(h,w)$ 是 $h$ **作为上下文**出现的次数：

$$
P(w_1,\dots,w_m)\approx\prod_{t=1}^{m+1}P(w_t\mid w_{t-n+1},\dots,w_{t-1}),\qquad P_{\text{MLE}}(w\mid h)=\frac{c(h,w)}{c(h)}
$$

句首补 $n-1$ 个 `<s>`，让第一个词也有完整上下文，`<s>` 只当上下文、永远不被预测。句尾补一个 `</s>`，它和普通词一样要被预测，所以连乘到 $m+1$；没有它，所有长度为 1 的句子概率和是 1，长度为 2 的也是 1，加起来不是合法分布。

MLE 给没见过的 n-gram 概率 0，测试句里只要有一个，整句概率就是 0，所以要 add-k 平滑（$k=1$ 是 Laplace）。连乘会下溢，句子概率一律 log 相加：$\log P(\text{句子})=\sum_{t=1}^{m+1}\log P(w_t\mid h_t)$。Perplexity 是平均每个 token 负 log 概率的指数，$N$ 是被预测的 token 数（每句词数 + 1 个 `</s>`）：

$$
P_{\text{add-}k}(w\mid h)=\frac{c(h,w)+k}{c(h)+kV},\qquad \text{PPL}=\exp\Big(-\frac{1}{N}\sum_t\log P(w_t\mid h_t)\Big)
$$

$V$ 必须是**所有可能被预测的 token 数** = 训练词 + `</s>` + `<unk>`（`<s>` 不算），否则对 $w$ 求和不等于 1。测试时的生词映射成 `<unk>`；没见过的上下文 $c(h)=0$，公式自动给出均匀分布 $1/V$。

### 实现

```python
import math
from collections import Counter
from typing import List, Sequence, Set

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NGramLM:
    def __init__(self, n: int = 2, k: float = 1.0):
        """n=2 是 bigram；k=1 是 Laplace，k=0 退化成 MLE"""
        assert n >= 1 and k >= 0
        self.n, self.k = n, k
        self.vocab: Set[str] = set()                     # 可被预测的 token：训练词 + </s> + <unk>
        self.ngram_counts: Counter = Counter()           # (h, w) -> c(h, w)，h 是 n-1 元 tuple
        self.context_counts: Counter = Counter()         # h -> c(h)，没见过的 h 返回 0

    def _pad(self, words: Sequence[str]) -> List[str]:
        """左边补 n-1 个 <s>，生词映射成 <unk>"""
        return [BOS] * (self.n - 1) + [w if w in self.vocab else UNK for w in words]

    def train(self, sentences: List[List[str]]) -> "NGramLM":
        # 1. 先定词表，计数时才知道谁是生词
        self.vocab = {w for sent in sentences for w in sent} | {EOS, UNK}
        # 2. 每个要预测的位置（第一个词到 </s>）记一次 c(h, w) 和 c(h)
        for sent in sentences:
            tokens = self._pad(sent) + [EOS]
            for t in range(self.n - 1, len(tokens)):
                h = tuple(tokens[t - self.n + 1:t])      # 前 n-1 个 token
                self.ngram_counts[(h, tokens[t])] += 1
                self.context_counts[h] += 1
        return self

    def prob(self, word: str, context: Sequence[str]) -> float:
        """P(word | context 的最后 n-1 个词) = (c(h, w) + k) / (c(h) + k * V)"""
        padded = self._pad(context)
        h = tuple(padded[len(padded) - self.n + 1:])     # n=1 时是空 tuple
        w = word if word in self.vocab else UNK
        numerator = self.ngram_counts[(h, w)] + self.k
        denominator = self.context_counts[h] + self.k * len(self.vocab)
        return numerator / denominator if denominator > 0 else 0.0   # 只有 k=0 且 h 没见过才是 0

    def sentence_logprob(self, sentence: List[str]) -> float:
        """log P(sentence)：第一个词到 </s> 每个位置的 log 条件概率相加"""
        tokens = sentence + [EOS]
        probs = [self.prob(w, tokens[:t]) for t, w in enumerate(tokens)]
        return -math.inf if 0.0 in probs else sum(math.log(p) for p in probs)   # k=0 时可能有 0

    def perplexity(self, sentences: List[List[str]]) -> float:
        """exp(-总 log 概率 / N)，N = 每句词数 + 1"""
        total = sum(self.sentence_logprob(s) for s in sentences)
        return math.exp(-total / sum(len(s) + 1 for s in sentences))


corpus = [["i", "like", "nlp"], ["i", "like", "deep", "learning"], ["you", "like", "nlp"]]
test = [["i", "like", "nlp"], ["i", "like", "learning"]]         # 第二句有没见过的 bigram
lm = NGramLM(n=2, k=1.0).train(corpus)
print(len(lm.vocab), lm.prob("nlp", ["like"]), lm.perplexity(test))   # 8 0.2727272727272727 4.163963996609268
print(NGramLM(n=2, k=0.0).train(corpus).perplexity(test))       # inf：MLE 遇到没见过的 bigram
```

### 自测

用上面的 `corpus`、`lm`（bigram，k=1）和 `NGramLM`：

```python
def test_ngram() -> None:
    # 1. 手算：V = 6 个词 + </s> + <unk> = 8；P(nlp | like) = (2 + 1) / (3 + 8)；生词按 <unk> 算，(0 + 1) / 11
    assert math.isclose(lm.prob("nlp", ["like"]), 3 / 11) and math.isclose(lm.prob("xyz", ["like"]), 1 / 11)
    # 2. 任意 n、任意上下文（包括没见过的），整个词表上的概率和都是 1
    for n in [1, 2, 3]:
        model = NGramLM(n=n, k=0.5).train(corpus)
        for context in [[], ["i", "like"], ["never", "seen"]]:
            assert math.isclose(sum(model.prob(w, context) for w in model.vocab), 1.0)
    # 3. "i like nlp" 的 4 个概率是 3/11、3/10、3/11、3/10，PPL = 几何平均的倒数
    assert math.isclose(lm.perplexity([["i", "like", "nlp"]]), (3 / 11 * 3 / 10) ** (-1 / 2))
    print("all tests passed")


if __name__ == "__main__":
    test_ngram()
```

### 关键追问

- **add-k 有什么问题？还有哪些平滑？** $V$ 很大时 add-k 从见过的词那里抢走太多概率，$k$ 一般取小于 1 并在验证集上调。改进有 Backoff（高阶没见过就退到低阶，stupid backoff 乘固定的 0.4）、Interpolation（$\lambda_3P_3+\lambda_2P_2+\lambda_1P_1$，$\lambda$ 在 held-out 上调）和 Kneser-Ney（计数减固定折扣，低阶分布用「跟在多少种不同的词后面」代替词频），Modified KN 是 n-gram 的事实标准。
- **n 越大越好吗？** 不一定：可能的 n-gram 有 $V^n$ 种，n 越大越稀疏，大多数组合没出现过或只出现一两次，估计越不可靠。实践中常用 3~5 阶，配合 KN 平滑和大语料。
- **Perplexity 怎么理解？** 它是平均负 log 似然（交叉熵）的指数，可以理解成「每一步平均在多少个词里犹豫」，对 $V$ 个词均匀猜时 PPL 正好是 $V$。比较 PPL 的前提是同一个词表、同样处理 `</s>` 和 `<unk>`，把更多词映射成 `<unk>` 会让 PPL 变低，模型本身并没有变好。
- **和神经语言模型什么关系？** 目标相同：按链式法则预测下一个 token，用 PPL 评估，交叉熵 loss（第 9 节）就是 log PPL。N-gram 是一张计数表，上下文固定 $n-1$ 个词、各算各的；Transformer（第 1 节）用 embedding 共享参数、上下文长得多，所以不需要手工平滑。

---

## 13. 手写 BPE Tokenizer

BPE（Sennrich et al., 2016）从字符出发，反复把最常见的相邻符号对合并成新 token：高频词最终变成一个 token，罕见词和新词拆成学过的片段，最坏退回单个字符，只要字符见过就不会 OOV。

1. 每个词拆成字符，末尾加词尾标记 `</w>`（`low` 变成 `l o w </w>`）。词尾的 `est</w>` 和词中间的 `est` 因此是两个 token，decode 时也靠 `</w>` 还原空格。
2. 统计所有**相邻符号对**的次数，按词频加权：`newest` 出现 6 次，它里面的 `(e, s)` 就算 6 次。
3. 把次数最多的 pair 合并成新符号，追加到 merges；在新切分上重复 2、3 步，共 `num_merges` 次。**词表大小 = 初始符号数 + 合并次数**（再加 `<unk>` 等特殊 token），实际中先定词表大小再反推合并次数。

平局规则：取最先出现的 pair（词按输入顺序、词内从左到右扫描，`Counter` 按插入顺序迭代，`max` 遇到平局返回第一个），和论文参考代码在 Python 3.7+ 上一致；想和输入顺序无关就改成 `min(pairs, key=lambda p: (-pairs[p], p))`，面试时先问清楚。

编码按 rank 重放训练（GPT-2 的写法）：merges 的下标就是 rank，每次在当前切分里找 rank 最小的相邻 pair，合并它的所有出现，直到剩下的 pair 都没学过。

### 实现

```python
import math
from collections import Counter
from typing import Dict, List, Tuple

EOW = "</w>"
Pair = Tuple[str, str]
Word = Tuple[str, ...]                       # 一个词当前的切分，如 ("l", "o", "w", "</w>")


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
        word_counts: 词 -> 词频（预分词之后）
        num_merges: 合并次数
    Returns:
        merges（按学到的顺序）和每个训练词最后的切分
    """
    splits = {word: tuple(word) + (EOW,) for word in word_counts}      # 1. 拆成字符 + </w>
    merges: List[Pair] = []
    for _ in range(num_merges):
        pairs: Counter = Counter()                                     # 2. 数相邻 pair，按词频加权
        for word, symbols in splits.items():
            for pair in zip(symbols, symbols[1:]):
                pairs[pair] += word_counts[word]
        if not pairs:                                                  # 每个词都只剩一个符号了
            break
        best = max(pairs, key=pairs.get)                               # 3. 最高频，平局取最先出现的
        splits = {w: merge_pair(s, best) for w, s in splits.items()}  # 4. 所有词里合并它
        merges.append(best)
    return merges, splits


def encode_word(word: str, ranks: Dict[Pair, int]) -> List[str]:
    """反复合并 rank 最小（最早学到）的相邻 pair，直到剩下的 pair 都没学过"""
    symbols = tuple(word) + (EOW,)
    while len(symbols) > 1:
        best = min(zip(symbols, symbols[1:]), key=lambda p: ranks.get(p, math.inf))
        if best not in ranks:
            break
        symbols = merge_pair(symbols, best)
    return list(symbols)


word_counts = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges, splits = train_bpe(word_counts, num_merges=10)
print(merges)
# [('e', 's'), ('es', 't'), ('est', '</w>'), ('l', 'o'), ('lo', 'w'),
#  ('n', 'e'), ('ne', 'w'), ('new', 'est</w>'), ('low', '</w>'), ('w', 'i')]
print(splits)
# {'low': ('low</w>',), 'lower': ('low', 'e', 'r', '</w>'),
#  'newest': ('newest</w>',), 'widest': ('wi', 'd', 'est</w>')}
ranks = {pair: rank for rank, pair in enumerate(merges)}
print(encode_word("lowest", ranks), encode_word("newer", ranks))
# ['low', 'est</w>'] ['new', 'e', 'r', '</w>']    没见过的词拆成学过的片段
```

### 自测

用上面的 `word_counts`、`merges`、`splits` 和 `ranks`：

```python
def test_bpe() -> None:
    # 1. 前三次合并都在拼 e-s-t-</w>：newest(6) + widest(3) = 9，最高
    assert train_bpe(word_counts, 3)[0] == [("e", "s"), ("es", "t"), ("est", EOW)]
    # 2. 训练词按 rank 编码 == 训练结束时的切分
    assert all(encode_word(w, ranks) == list(s) for w, s in splits.items())
    # 3. 词表大小 = 11 个初始符号（10 个字母 + </w>）+ 10 次合并
    assert len({ch for w in word_counts for ch in w} | {EOW} | {a + b for a, b in merges}) == 11 + 10
    # 4. 往返：去掉 </w> 拼回去就是原词；num_merges 给很大时每个词合并成一个 token 就停
    assert "".join(encode_word("slowest", ranks)).replace(EOW, "") == "slowest"
    assert all(len(s) == 1 for s in train_bpe(word_counts, 1000)[1].values())
    print("all tests passed")


if __name__ == "__main__":
    test_bpe()
```

### 变体

- **Byte-level BPE（GPT-2）**：初始符号是 256 个字节，任何字符串都能编码、不需要 `<unk>`；词表 50257 = 256 + 50000 次合并 + 1 个 `<|endoftext|>`，用并进 token 的词前空格（`Ġ`）代替 `</w>`。
- **WordPiece（BERT）** 选 pair 的分数是 $c(ab)/(c(a)\,c(b))$，编码按词表从左到右最长匹配，词中片段带 `##`；**Unigram（SentencePiece）** 从大词表出发反复删掉对语料似然影响最小的 token，编码用 Viterbi 找概率最大的切分。

### 关键追问

- **训练复杂度？怎么加速？** 朴素版每次合并都重扫所有词，复杂度 $O(MN)$（$M$ 是合并次数，$N$ 是去重后所有词的符号总数）。HF tokenizers、SentencePiece 维护 pair 计数和「pair 到包含它的词」的倒排索引，合并时只更新受影响的词，最高频 pair 用堆找。
- **encode 为什么必须按 rank 合并？** 后面的 merge 建立在前面的之上（`(new, est</w>)` 要求 `est</w>` 先合出来），按 rank 合并就是在这个词上重放训练，训练词能复现训练时的切分（自测第 2 条）。顺序一乱会得到训练时没见过的 token 组合，同一个词训练和推理时的 id 序列对不上。
- **词表大小怎么权衡？** 词表大则序列短、上下文窗口装得多，代价是 embedding 和输出层的 $V\times d$ 矩阵变大、罕见 token 训练不足；词表小则序列变长，注意力计算量按长度平方涨。LLaMA 2 用 32000，LLaMA 3 扩到约 12.8 万。
- **为什么先预分词？中文怎么办？** 预分词让合并不跨词边界，否则会学出 `in the`、`dog.` 这类 token，同一个词跟着不同标点就成了不同 token。中文没有空格：byte-level BPE 从 UTF-8 字节起步（一个汉字 3 个字节），SentencePiece 直接在原始文本上训练、空格当普通字符，罕见字用 byte fallback 退回字节。

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
| [11. ML Coding 基础](11-ml-coding-basics.md)  |
| [12. ML Coding 专题](12-ml-coding.md)         |
| [参考资料](references.md)                     |