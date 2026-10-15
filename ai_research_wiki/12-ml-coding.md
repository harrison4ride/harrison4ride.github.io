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

`d_model` 是每个 token 向量的长度。输入只是 token id，embedding 把每个 token 变成 `d_model` 维的向量，之后每一层的输入输出都保持这个宽度，所以叫「模型的维度」。它是自己选的超参数，不由数据决定，原论文取 512。同一个数在 `nn.MultiheadAttention` 里叫 `embed_dim`，在 HuggingFace 配置里叫 `hidden_size`。

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
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_k) -> (B, T, C)；transpose 后内存不连续，reshape 会自动复制（view 必须先 .contiguous()）"""
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

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

**第 2 步：FFN、位置编码、Encoder 层和 Decoder 层。** FFN 对每个位置单独做 `C → d_ff → C`（原论文 512 → 2048 → 512，ReLU）。位置编码用原论文的正弦编码，原理见 1.1 节。层结构按原论文的 Post-LN：每个子层都是 `x = LayerNorm(x + Dropout(Sublayer(x)))`。

正弦编码

$$
PE_{(pos,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad
PE_{(pos,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

记 $\omega_i=10000^{-2i/d}$。**时钟指针**：每对 $(\sin(pos\,\omega_i), \cos(pos\,\omega_i))$ 是单位圆上的一根指针，位置加 1 就转 $\omega_i$ 弧度。$i=0$ 每步转 1 弧度，像秒针；$i$ 越大转得越慢，像时针。一个位置就是所有指针读数的组合，和词向量相加后作为输入。原论文试过可学习的位置表，效果几乎一样，选正弦是希望能外推到更长的序列。

**平移 = 旋转**：由和角公式，$PE_{pos+k}=M_k\,PE_{pos}$，$M_k$ 是把第 $i$ 对指针转 $k\omega_i$ 的块对角旋转矩阵，只和 $k$ 有关。所以模型用一个固定的线性变换就能表达「往后数 $k$ 个」，而且 $PE_{pos}\cdot PE_{pos+k}=\sum_i\cos(k\omega_i)$ 只依赖距离。PyTorch 模块见第 1 节。

```python
class PositionwiseFeedForward(nn.Module):
    """逐 token 的前馈网络：变换特征，不混合不同位置的信息，也不添加位置信息。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # C -> d_ff：投影到中间特征空间，通常 d_ff > C
        self.fc2 = nn.Linear(d_ff, d_model)  # d_ff -> C：恢复宽度，才能和输入做残差相加
        self.relu = nn.ReLU()               # 引入非线性，否则两次线性变换仍等价于一次

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)"""
        # nn.Linear 只作用于最后一维：B、T 不变，所有位置共用同一套权重
        # 1. fc1：(B, T, C) -> (B, T, d_ff)
        # 2. ReLU：负值置零，形状不变
        # 3. fc2：(B, T, d_ff) -> (B, T, C)
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """
    固定正弦位置编码：给每个位置一个 C 维向量，加到 token embedding 上。
    PE[pos, 2i] = sin(pos / 10000^(2i/C))，PE[pos, 2i+1] = cos(同上)。
    本示例要求 d_model 为偶数，输入长度 T <= max_len。
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        # dropout 是相加后的正则化，不负责生成位置编码；model.eval() 时不丢弃元素
        self.dropout = nn.Dropout(dropout)
        # 1. 列出位置 pos = 0, 1, ..., max_len-1；加一维，便于和各频率广播相乘
        position = torch.arange(max_len).unsqueeze(1).float()                    # (max_len, 1)
        # 2. 每一对通道共用一个频率：exp(-log(10000) * 2i/C) = 1 / 10000^(2i/C)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (C/2,)
        # 3. 广播得到 (max_len, C/2) 的相位；偶数通道填 sin，奇数通道填 cos
        pe = torch.zeros(max_len, d_model)                                       # (max_len, C)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 4. 加 batch 维便于复用；buffer 不是可训练参数，随 .to(device) 移动并存入 state_dict
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, C)：不训练，但跟着 .to(device) 走、存进 state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)"""
        # 1. 按实际长度 T 取前 T 个位置：(1, T, C)，同一套位置向量广播到 B 个样本
        # 2. 与 embedding 逐元素相加，形状仍为 (B, T, C)
        # 3. 对相加结果做 dropout；若只想加位置编码，可以省略这一步
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderLayer(nn.Module):
    """self-attention -> FFN，每个子层 Post-LN：LN(x + Dropout(Sublayer(x)))"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        # 每个子层各有一套可学习的 LayerNorm 参数；沿最后的 C 维归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 对子层输出做 dropout，再和子层输入相加；不是对残差相加后的整体做 dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T_src, C)；src_mask: (B, 1, 1, T_src)。返回 (B, T_src, C)"""
        # 1. self-attention：Q = K = V = x，读取源句各位置；src_mask 只屏蔽 padding
        #    执行顺序：Attention -> Dropout -> 加原输入 x（残差）-> LayerNorm
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        # 2. FFN：每个 token 独立变换特征，再做 Dropout -> 残差相加 -> LayerNorm
        #    两个子层的输入输出均为 (B, T_src, C)，可以直接堆叠下一层
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderLayer(nn.Module):
    """masked self-attention -> cross-attention -> FFN，每个子层 Post-LN"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        # 三个子层分别归一化，参数互不共享；都只沿最后的 C 维计算
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_tgt, C)，目标端当前表示
            memory: (B, T_src, C)，Encoder 输出，在各 Decoder 层中复用
            src_mask: (B, 1, 1, T_src)，屏蔽源端 padding
            tgt_mask: (B, 1, T_tgt, T_tgt)，屏蔽目标端未来位置和 padding
        Returns:
            (B, T_tgt, C)，每个目标位置融合前缀和源句信息后的表示
        """
        # 每步都按 Sublayer -> Dropout -> 残差相加 -> LayerNorm 执行（Post-LN）
        # 1. masked self-attention：Q = K = V = decoder，tgt_mask 挡住未来和 padding
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        # 2. cross-attention：Q 来自 decoder，K、V 来自 memory，注意力矩阵 (B, H, T_tgt, T_src)
        #    每个目标位置查询完整源句；输出长度由 Q 决定，仍为 T_tgt，memory 不被修改
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        # 3. FFN：逐 token 做 C -> d_ff -> C，返回 (B, T_tgt, C)
        return self.norm3(x + self.dropout(self.ffn(x)))
```

**第 3 步：拼成 Transformer。** N 个层要放进 `nn.ModuleList`，放在普通 list 里不会被注册，`parameters()` 找不到，优化器也不会更新它们。`encode` 和 `decode` 分开写，是因为生成时源句只编码一次，`decode` 每生成一个 token 调用一次（见 1.3 节）。原论文把 embedding 乘 $\sqrt{d_{\text{model}}}$，让词向量和取值在 -1 到 1 之间的位置编码量级相当。

```python
def make_src_mask(src: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """(B, T_src) -> (B, 1, 1, T_src)，True = 真实 token；广播到每个头、每个 query"""
    # 只限制哪些 key 可被读取：两个单例维分别对应 H 和 T_q
    # Encoder self-attention 和 Decoder cross-attention 都可以复用这个源端 mask
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """(B, T_tgt) -> (B, 1, T_tgt, T_tgt)：padding mask 与因果 mask 同时为 True 才可见"""
    seq_len = tgt.size(1)
    # 1. padding mask：每个 query 都不能读取 PAD key
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)                                # (B, 1, 1, T)
    # 2. 因果 mask：行是 query 位置 i，列是 key 位置 j，只保留 j <= i（包括自己）
    causal = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).bool()      # (T, T)，下三角
    # 3. 广播后取交集，得到每个样本的可见位置；H 维仍为 1，供所有头共用
    return pad_mask & causal                                                          # (B, 1, T, T)


class Transformer(nn.Module):
    """Encoder-Decoder Transformer：forward(src, tgt) -> logits (B, T_tgt, V)"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048,
                 max_len: int = 512, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        # 源语言和目标语言分别查自己的词表：token id (B, T) -> 向量 (B, T, C)
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        # 正弦编码没有可训练参数，两端共用同一张位置表；各自在入栈前调用一次
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        # 每次构造一个新层，各层参数独立；ModuleList 让 PyTorch 注册并管理这些子模块
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # 输出投影与 FFN 是不同用途：C -> V，为目标词表中每个 token 产生一个分数
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """src: (B, T_src) -> memory: (B, T_src, C)"""
        # 1. 查 embedding -> 乘 sqrt(C) 调整量级 -> 加位置编码并 dropout
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        # 2. 依次经过 N 个 Encoder 层，形状始终为 (B, T_src, C)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        # 3. 最后一层表示作为 memory，供每个 Decoder 层的 cross-attention 查询
        return x

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """tgt: (B, T_tgt)，memory: (B, T_src, C) -> logits: (B, T_tgt, V)"""
        # 1. 目标前缀也要查自己的 embedding，并在入栈前加位置编码
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        # 2. 每层先读目标前缀，再查相同的源端 memory，形状始终为 (B, T_tgt, C)
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # 3. 投影到词表：(B, T_tgt, C) -> (B, T_tgt, V)
        #    返回未归一化的 logits；训练时 CrossEntropyLoss 内部处理，无须先 softmax
        return self.fc_out(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src: (B, T_src)；tgt: (B, T_tgt)，以 BOS 开头的目标前缀（teacher forcing）"""
        # 1. 根据 token id 创建 mask：源端屏蔽 PAD，目标端同时屏蔽 PAD 和未来位置
        src_mask = make_src_mask(src, self.pad_id)
        tgt_mask = make_tgt_mask(tgt, self.pad_id)
        # 2. 源句编码成 memory，再结合目标前缀解码；训练时一次计算全部目标位置的 logits
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

| 方法        | 怎么挑下一个 token                     | 输出 | 常见用途                              |
| ----------- | -------------------------------------- | ---- | ------------------------------------- |
| Greedy      | argmax                                 | 确定 | 抽取、代码、评测                      |
| Beam Search | 保留累计 log-prob 最高的 k 条路径      | 确定 | 翻译、摘要、语音识别（k 常取 4 到 5） |
| 采样        | 温度缩放、top-k / top-p 截断后按概率抽 | 随机 | 对话、创意写作                        |

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

两步循环：**分配**（每个点归到最近的中心）和**更新**（每个簇的均值当新中心），直到中心基本不再移动。它在最小化每个点到所属中心的平方距离之和，也叫 inertia：$J=\sum_i\min_j\|x_i-\mu_j\|^2$。

### 实现（面试基础版）

假设输入是有效的 NumPy 数组 `X (n, d)`，`1 <= n_clusters <= n`。随机选 k 个样本作初始中心，随后反复分配、更新。`fit` 返回自己，`centroids` 保存中心，`labels_` 保存训练样本的簇下标。

```python
import numpy as np


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray) -> "KMeans":
        """X: (n, d)。分配到最近中心，再用簇内均值更新中心。"""
        X = np.asarray(X, dtype=float)  # 中心要存均值，使用浮点数
        # 1. 随机选 k 个不同样本作为初始中心：(k, d)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iter):
            # 2. 算距离，找归属
            distances = np.sum((X[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)

            # 3. 更新：对每个簇沿样本维求均值，得到新的中心 (k, d)
            # 3. 更新中心
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # 4. 如果中心不动了，就停止
            if np.all(centroids == new_centroids):
                break
            self.centroids = new_centroids
```

**距离计算怎么读？** `X[:, None, :]` 给样本增加一个中心维，`centroids[None, :, :]` 给中心增加一个样本维。广播相减就得到「每个点减每个中心」，平方后沿特征维 `axis=2` 求和，得到 `(m, k)` 的平方距离矩阵。`argmin(axis=1)` 为每个点选一个中心。

### CodeSignal ML Core 版（纯 Python）

题面不许 import 时，用 list 和循环写同样的两步。初始中心按题面指定；这里取前 k 个点，距离相同取下标较小的中心。

```python
def squared_distance(p: list, q: list) -> float:
    return sum((a - b) ** 2 for a, b in zip(p, q))


def solution(data: list, k: int, max_iter: int = 100) -> list:
    """data: n 个 d 维点。返回每个点的簇下标。"""
    centroids = [list(p) for p in data[:k]]
    for _ in range(max_iter):
        # 1. 分配：min 返回距离最小的中心下标
        labels = [min(range(k), key=lambda j: squared_distance(p, centroids[j])) for p in data]

        # 2. 更新：逐簇计算每一维的均值；空簇保留旧中心
        new_centroids = []
        for j in range(k):
            points = [p for p, label in zip(data, labels) if label == j]
            if points:
                mean = [sum(col) / len(points) for col in zip(*points)]
                new_centroids.append(mean)
            else:
                new_centroids.append(centroids[j])

        # 相同的分组会算出相同的均值；这里比较列表，不引入额外的容差参数
        if new_centroids == centroids:
            break
        centroids = new_centroids

    return [min(range(k), key=lambda j: squared_distance(p, centroids[j])) for p in data]
```

### Example 与自测

```python
np.random.seed(0)  # 固定随机种子，让例子可复现
X = np.array([[0, 0], [0, 1], [1, 0], [9, 9], [9, 10], [10, 9]])
model = KMeans(n_clusters=2).fit(X)
print(model.centroids)  # 两个中心约为 (0.33, 0.33) 和 (9.33, 9.33)，编号可能相反
print(model.labels_)   # 前三个点一簇，后三个点一簇

# 簇编号本身没有意义：检查两组点各自同簇、两组之间不同簇
labels = model.labels_
assert np.all(labels[:3] == labels[0])
assert np.all(labels[3:] == labels[3])
assert labels[0] != labels[3]
assert np.allclose(model.centroids[labels[0]], [1 / 3, 1 / 3])
assert np.allclose(model.centroids[labels[3]], [28 / 3, 28 / 3])
assert np.array_equal(model.predict(np.array([[0, 0], [10, 10]])), labels[[0, 3]])
assert solution([[0, 0], [10, 10], [0, 1], [10, 9]], k=2) == [0, 1, 0, 1]
print("all tests passed")
```

### 关键追问

- **收敛到全局最优吗？** 不保证，初始化会影响结果。追问时再补充 k-means++（按到最近中心的平方距离加权选新中心）和多次重启（选 inertia 最小的一次）。
- **空簇怎么办？** 上面保留旧中心；也可以重置到离当前中心最远的点。
- **复杂度？** 每轮 $O(nkd)$，n 是样本数、k 是簇数、d 是维度。上面广播距离的临时数组占 $O(nkd)$ 空间。
- **k 怎么选？** inertia 随 k 增大单调下降，不能直接取最小；用肘部法或 silhouette score。
- **为什么要标准化？** 欧氏距离会被尺度较大的特征主导。
- **题面指定初始中心怎么办？** 将随机选中心那两行替换成题面要求；最大轮数、距离平局规则也按题面执行。

---

## 3. KNN

KNN 的 `fit` 只保存训练数据。预测时，对每个新点**算距离 → 排序取最近的 k 个 → 多数投票**。分类投票，回归则取邻居标签的均值。

### 实现（面试基础版）

先写最常见的分类版本。假设 `X`、`y` 是有效的 NumPy 数组，`1 <= k <= 训练样本数`，先调用 `fit` 再调用 `predict`。

```python
from collections import Counter

import numpy as np


class KNN:
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        """X: (n, d)，y: (n,)。保存数据，没有参数训练。"""
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (m, d) -> 预测类别 (m,)"""
        predictions = []
        for x in X:
            # 1. 当前点 (d,) 与全部训练点 (n, d) 相减，得到 n 个平方距离
            distances = ((self.X_train - x) ** 2).sum(axis=1)  # (n,)，不用开根号
            # 2. 按距离从小到大排序，取前 k 个训练样本的下标
            indices = np.argsort(distances)[:self.k]         # (k,)
            # 3. 取邻居标签，多数投票得到当前点的预测类别
            neighbor_labels = self.y_train[indices]          # (k,)
            prediction = Counter(neighbor_labels).most_common(1)[0][0]
            predictions.append(prediction)
        return np.array(predictions)
```

`Counter` 统计每个类别出现的次数，`most_common(1)` 返回票数最多的 `(类别, 次数)`，所以 `[0][0]` 取出类别。票数相同时，这个写法选在邻居列表中先出现的类别。回归版只需把投票那行换成 `prediction = neighbor_labels.mean()`。

### Example 与自测

```python
X_train = np.array([[0, 0], [0, 1], [1, 0], [9, 9], [9, 10], [10, 9]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_test = np.array([[0.2, 0.2], [9.2, 9.2]])

model = KNN(k=3).fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)  # [0 1]
assert np.array_equal(predictions, [0, 1])

# 用 k=1 检查预测确实跟随最近样本的标签；标签也可以是字符串
model = KNN(k=1).fit(X_train, np.array(["a", "b", "c", "d", "e", "f"]))
assert np.array_equal(model.predict(np.array([[0, 0.9], [9.9, 9]])), ["b", "f"])
print("all tests passed")
```

### 关键追问

- **复杂度？** 保存数组引用的 `fit` 是 $O(1)$，保留的训练数据占 $O(nd)$ 空间；每个测试点算距离 $O(nd)$、完整排序 $O(n\log n)$，m 个测试点共 $O(m(nd+n\log n))$。逐个预测时，距离计算的临时数组占 $O(nd)$ 空间。
- **怎么优化？** 追问时再用 `argpartition` 以 $O(n)$ 选出最近 k 个；批量算距离可展开 $\|a-b\|^2=\|a\|^2-2a\cdot b+\|b\|^2$，避免构造 `(m, n, d)` 数组。
- **k 怎么选？** k 小对噪声敏感，k 大容易欠拟合；用交叉验证选择。二分类选奇数 k 可以避免投票平局，多分类仍可能平局。
- **为什么要标准化？** 距离会被尺度最大的特征主导，比如年收入会盖住年龄。
- **数据量大或者高维怎么办？** 低维可用 KD-Tree、Ball Tree；高维可先降维或用 embedding，再使用近似最近邻索引。
- **题面不许 import 怎么写？** 用循环算平方距离，`sorted` 取前 k 个，再用 dict 计票；核心流程不变。

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
- **阈值一定是 0.5 吗？** 不一定，由误报和漏报的业务成本决定，按 precision、recall 选（6.1 节）。
- **多分类怎么办？** 换成 softmax 加交叉熵（第 7 节），梯度形式同样是 $P-Y$；或者训练 K 个一对多（one-vs-rest）的二分类器。
- **输出的概率能直接用吗？** 用 BCE 训练的 LR 本身校准得比较好；加了强正则或者类别重采样后，概率会偏，需要重新校准。

---

## 5. 决策树：信息熵与信息增益

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

## 6. Cosine Similarity

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

## 6.1 分类指标：Precision、Recall、F1 与 AUC

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

## 7. Softmax、交叉熵与常见 Loss

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

带 batch 时必须写 `axis=-1, keepdims=True`：省掉 axis 的 `np.max(x)`、`e.sum()` 会对整个矩阵取最大、求和，每行不再各自归一化。合并后对 logits 的梯度是 $(p-\text{onehot})/N$。手上只有概率时，先 `np.clip(p, 1e-15, 1)` 再取 log。

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
- **有 ignore_index 时 mean 除以什么？** 除以有效位置数。梯度累积（8 节）时各 micro-batch 的有效 token 数不同，各自取 mean 再平均有偏，严格写法是用 `reduction="sum"`，最后除以总有效 token 数。
- **类别不均衡怎么处理？** 多分类给 `F.cross_entropy` 传 `weight`（形状 `(C,)`），二分类用 `pos_weight`，简单负样本特别多（欺诈、CTR）时用 focal loss。focal loss 默认 γ = 2、α = 0.25，α 小于 0.5 是因为大量简单负样本已经被 γ 压掉了。
- **Label smoothing 的直觉？** one-hot 目标要求真实类别概率等于 1，logits 只能无限拉大，模型越来越过度自信；软目标的最优 logits 有限，校准和泛化通常更好。副作用是 loss 降不到 0；也有定义把 ε 分给其余 $C-1$ 类，面试时先问清楚。
- **蒸馏为什么乘 T²？** 不乘时梯度约按 $1/T^2$ 缩小：链式法则从 $z/T$ 带出一个 $1/T$，两个分布变平后 $p^{s}-p^{t}$ 又约正比于 $1/T$。乘 $T^2$ 后梯度量级基本不随 $T$ 变，改 $T$ 时不用重调和 CE 的权重 α。

---

## 8. PyTorch Training-Validation Loop

每个 epoch 分两半。**训练**：遍历训练集，每个 batch 做五步，清梯度、前向、算 loss、反向、更新。**验证**：在 `model.eval()` 和 `torch.no_grad()` 下只做前向，算 loss 和 accuracy，用来挑模型和 early stopping，所以最终效果要在另一份测试集上报告。`loss.backward()` 自动计算梯度，`optimizer.step()` 根据梯度更新模型参数。

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

- **`model.eval()` 和 `torch.no_grad()` 有什么区别？** `eval()` 改变层的计算：Dropout 不丢神经元，BatchNorm 用 running 统计量（9 节）；`no_grad()` 只是不建计算图，省内存和时间，不会关掉 Dropout。验证两个都要写：只写 `no_grad()` 结果每次不一样，只写 `eval()` 结果对但白建了计算图。
- **忘了 `optimizer.zero_grad()` 会怎样？** `.grad` 默认累加，每一步的梯度都混进之前所有 batch 的梯度，越攒越大、方向被带偏，不报错但 loss 不降。放在上一次 `step()` 之后、这一次 `backward()` 之前都行，习惯写在每个 batch 开头。
- **为什么累加 `loss.item() * xb.size(0)`？** 直接累加 `loss` 张量会把每个 batch 的计算图都留在内存里（基础篇 0.3 节）。乘样本数是因为最后一批更小，对每个 batch 的平均再求平均会给它过大的权重，自测已对拍加权平均等于整个验证集一次算的 loss。
- **`scheduler.step()` 放在哪？** 放在 `optimizer.step()` 之后，写反了会有 UserWarning，而且第一个学习率被跳过。`StepLR`、`CosineAnnealingLR` 每个 epoch 调一次；`OneCycleLR` 和 warmup 每个 batch 调一次，要放进 batch 循环。
- **显存只够小 batch 怎么办？** 梯度累积：每个小 batch 的 loss 先除以 `accum_steps` 再 `backward()`，中间不清梯度，攒够 `accum_steps` 次才 `step()` 和 `zero_grad()`，梯度等于大 batch 的平均梯度（有 BatchNorm 时不完全等价）。

---

## 9. BatchNorm 与 LayerNorm

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

|                | BatchNorm                                      | LayerNorm                        | RMSNorm                   |
| -------------- | ---------------------------------------------- | -------------------------------- | ------------------------- |
| 统计量的轴     | batch 维 `axis=0`                              | 特征维 `axis=-1`                 | 特征维，只算均方根        |
| 依赖 batch     | 是，batch 小时噪声大                           | 否                               | 否                        |
| 训练和推理不同 | 是：训练用 batch 统计量，推理用 running 统计量 | 否                               | 否                        |
| 常见位置       | batch 较大的前馈网络                           | Transformer（原论文、BERT、GPT） | LLaMA 等 decoder-only LLM |

### 关键追问

- **为什么要 γ 和 β？** 强行标准化会限制表达能力，有了 γ、β，网络可以学回需要的尺度和偏移。$\gamma=\sqrt{\sigma^2+\epsilon}$、$\beta=\mu$ 时完全还原原始输入。
- **忘了 `model.eval()` 会怎样？** BN 继续用当前 batch 的统计量，同一个样本的输出会随 batch 里其他样本变化，还会继续改 running 统计量。`torch.no_grad()` 不切换这个行为。
- **batch 很小时怎么办？** 统计量噪声大，效果明显变差；batch 为 1 时 PyTorch 在训练模式下直接报错。改用 LayerNorm 或 GroupNorm，它们不依赖 batch。
- **Pre-Norm 还是 Post-Norm？** Post-Norm（原论文）是 `LN(x + sublayer(x))`，层数深时训练不稳、依赖 warmup。Pre-Norm 是 `x + sublayer(LN(x))`，残差成为直通的主干，训练更稳，是现代 LLM 的默认选择。

---

## 10. 手写 LSTM 与参数量

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

反向交给 autograd 沿时间倒着传（BPTT）。`h2h.weight` 每一步都用到，梯度是所有时间步的和。

### 参数量

每层每个方向有 k 组变换（LSTM 的 k = 4），每组是从 $[x_t,h_{t-1}]$ 到 $H$ 维的线性层。PyTorch 有 `b_ih`、`b_hh` 两组 bias，前向里直接相加，数学上冗余但确实占参数；面试先问清约定，没说就两个都报。

$$
P_{\text{教科书}}=4\,\big(H(D_{in}+H)+H\big),\qquad P_{\text{PyTorch}}=4\,\big(H(D_{in}+H)+2H\big)
$$

- **堆叠**：第 1 层 $D_{in}=D$；之后各层吃上一层的输出，单向 $D_{in}=H$，双向 $D_{in}=2H$（两个方向拼起来）。
- **双向**：每层正、反两套独立参数，每层乘 2。**换模型只换 k**：LSTM 4、GRU 3、vanilla RNN 1。

| 算例（D = 100，H = 256，PyTorch 约定） | 计算                                                   | 参数量    |
| -------------------------------------- | ------------------------------------------------------ | --------- |
| LSTM，1 层单向                         | 4 × (256 × 356 + 2 × 256)                              | 366,592   |
| 同上，只算一组 bias                    | 4 × (256 × 356 + 256)                                  | 365,568   |
| GRU，1 层单向                          | 3 × (256 × 356 + 2 × 256)                              | 274,944   |
| LSTM，2 层双向                         | 第 1 层 2 × 366,592；第 2 层 2 × 4 × (256 × 768 + 512) | 2,310,144 |

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

## 11. 手写 N-gram 语言模型

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
- **和神经语言模型什么关系？** 目标相同：按链式法则预测下一个 token，用 PPL 评估，交叉熵 loss（第 7 节）就是 log PPL。N-gram 是一张计数表，上下文固定 $n-1$ 个词、各算各的；Transformer（第 1 节）用 embedding 共享参数、上下文长得多，所以不需要手工平滑。

---

## 12. 手写 BPE Tokenizer

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
