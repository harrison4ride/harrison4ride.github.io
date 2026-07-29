# 04. LLM 基础

[返回目录](README.md)

## 1. Transformer 自注意力如何工作？

### 30 秒回答

自注意力让序列中每个 token 根据内容动态聚合其他 token 的信息。输入表示 $X$ 经三个线性映射得到：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

注意力输出为：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

其中 $QK^\top$ 衡量 token 间相关性，$\sqrt{d_k}$ 防止点积随维度变大导致 softmax 饱和，$M$ 可用于 padding mask 或 decoder 的 causal mask。多头注意力让不同头在不同子空间学习语法、指代、位置或语义关系。

在 self-attention 中，Q/K/V 都来自同一输入 $X$，但 $W_Q,W_K,W_V$ 是三组独立的可学习投影参数，并不共享。

**直觉：Q/K/V 的“身世”。** 把 $W_Q,W_K,W_V$ 想成模型**学出来的三副“滤镜 / 眼镜”**，把同一个 token 投影成三种视角：

- $Q$（Query）= 当前 token 的“**搜索请求**”：我在找什么？
- $K$（Key）= 各 token 的“**标签**”：我能提供什么？
- $V$（Value）= 各 token 的“**内容本体**”：真正要被传递的语义。

**为什么只有 $QK^\top$ 有意义？** 它衡量“搜索请求”与“标签”的匹配分数（token 间相关性），$V$ 不参与匹配。$V$ 的作用是：$\operatorname{softmax}(QK^\top/\sqrt{d_k})$ 只给出**注意力权重（分配比例）**，$V$ 才是被这些权重**加权混合**出来的实际内容——没有 $V$，就只有一份“比例”而没有内容可传。

**类比：为什么必须有 $V$。** $Q$ 与 $K$ 像 HR 和面试官：他们反复比对后只产出一份“**评估报告**”（注意力权重，比如给“苹果” 90%、“今天” 10%），但报告本身不产出内容；真正被录用、干活的是 $V$。所以最后一步 $\text{权重}\times V$ 才是把内容按比例混合出来：$0.9\,V_{\text{苹果}}+0.1\,V_{\text{今天}}$。

一句话点题：真正被训练学到的“真东西”是**静态的透镜** $W_Q,W_K,W_V$，$X$ 是动态输入；同一套 $W$ 以不变应万变，把任意 $X$ 规范成 $Q/K/V$。

Transformer 比 RNN 更适合大规模长序列训练，主要因为 token 间计算可并行，任意两个位置的信息路径长度为 $O(1)$，而 RNN 必须顺序计算，路径长度为 $O(n)$，更容易出现梯度衰减和吞吐瓶颈。

### 深入追问

**为什么 Attention 也不是完美的长序列方案？**

- 标准 Attention 的计算和注意力矩阵显存复杂度为 $O(n^2)$。直觉是**全员两两握手**：每个 token 都要和所有 token 打一次分，$n=1000$ 就是 $1000\times1000=10^6$ 次打分；$n=10^5$ 时 $n^2=10^{10}$，注意力矩阵直接撑爆显存。
- 长上下文中模型可能“看得到但用不好”，出现位置偏置和 Lost in the Middle。
- KV Cache 在自回归推理时随上下文长度线性增长。
- 因此出现 FlashAttention、滑动窗口、稀疏注意力、线性注意力、状态空间模型和 KV 压缩等方案。

**为什么除以 $\sqrt{d_k}$？**

点积 $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ 是 $d_k$ 个乘积之和。若各分量独立、均值 0、方差 1，则点积**方差 $\approx d_k$**（维度越大，数值越容易变得很大）。数值过大时 softmax 进入**饱和（赢者通吃，某个位置 $\approx100\%$）**，该处梯度 $\approx0$，模型学不动。除以 $\sqrt{d_k}$ 把方差**缩回 $\approx1$**（要让方差缩小 $N$ 倍就除以 $\sqrt N$），softmax 平滑、梯度健康。

**具体数字。** 以 $d_k=128$ 为例：分量方差为 1 时，128 项累加后点积方差约 128、标准差 $\approx11.3$，分数峰值可达 $\pm30$ 量级（约 $2.65$ 个标准差）。而 softmax 走指数：$e^{30}$ 比 $e^{10}$ 大约 5 亿倍（$e^{20}\approx4.9\times10^8$），两者的权重会塌成几乎 $100\%$ 对几乎 $0$（约 $2\times10^{-9}$）。除以 $\sqrt{128}\approx11.3$ 后，$30$ 被拉回 $\approx2.65$，方差回到 $\approx1$，softmax 重新平滑。

**为什么把 $K,V$ 提前投影好，而不是“传 $X$、最后再乘 $W_V$”？**

数学上二者等价（$\operatorname{softmax}(\cdot)(XW_V)=(\operatorname{softmax}(\cdot)X)W_V$），先算 $V=XW_V$ 是**工程**考量：

- **多头先降维**：每个头先把维度从 $d_{\text{model}}$ 降到 $d_{\text{model}}/h$（如 $4096\to128$）再与 $n\times n$ 注意力矩阵相乘，FLOPs 小得多；若拿全维度 $X$ 去乘 $n\times n$，计算量暴涨数倍。
- **子空间解耦**：$W_V$ 把 $X$ 投影到“内容 / 价值”子空间，过滤并解耦特征。
- **KV Cache**：推理时预先算好 $K,V$ 存起来即可复用；若只缓存 $X$，每步还要按各头的 $W_K,W_V$ 重新提特征。

**训练和推理有什么差异？**

- 训练时可用 teacher forcing，一次并行计算所有位置。
- 自回归推理必须逐 token 生成，但会缓存历史 K/V，避免重复计算历史 token。
- Prefill 处理完整 prompt，通常计算密集；Decode 每次生成一个 token，通常受显存带宽和 KV Cache 访问限制。

### 常见误区

- “Transformer 处理长序列的复杂度比 RNN 更低”不准确。标准 Attention 是 $O(n^2)$，RNN 每层是 $O(n)$；Transformer 的优势主要是并行性和更短的信息路径。
- 自注意力本身不包含顺序信息，必须额外注入位置。

---

## 1.1 原始 Transformer 的整体结构

### 它抛弃了什么，用什么替代？

《Attention Is All You Need》中的 Transformer 不再依赖 RNN 的序列递归和 CNN 的局部卷积，而是使用 Attention 在任意位置之间直接传递信息，并用逐位置 FFN 做非线性变换。

这带来两个主要变化：

- 训练时所有位置可以并行计算，不再受 RNN 的时间步依赖限制。
- 任意两个 token 的信息路径长度从 RNN 的 $O(n)$ 缩短为单层 self-attention 中的 $O(1)$。

这里的“完全抛弃”特指原论文架构。后续模型仍可能把卷积、循环、状态空间层或局部注意力与 Transformer 组合。

### Encoder 和 Decoder

原论文的 base Transformer 是 Encoder-Decoder 架构：

- Encoder 堆叠 6 层。
- Decoder 堆叠 6 层。
- $d_{\text{model}}=512$，8 个注意力头，FFN 中间维度为 2048。

“6+6 层”是原论文配置，不是 Transformer 的固定定义。现代模型的层数和架构可以完全不同。

### Encoder 每层

1. Multi-Head Self-Attention。
2. Residual Connection + LayerNorm。
3. Position-wise Feed-Forward Network。
4. Residual Connection + LayerNorm。

原论文采用 Post-LN：

$$
\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))
$$

### Decoder 每层

1. Masked Multi-Head Self-Attention。
2. Residual Connection + LayerNorm。
3. Encoder-Decoder Cross-Attention。
4. Residual Connection + LayerNorm。
5. Position-wise Feed-Forward Network。
6. Residual Connection + LayerNorm。

Decoder self-attention 使用 causal mask，禁止位置 $t$ 读取未来位置 $>t$，避免 teacher forcing 训练时发生答案泄漏。

**面试核心：Decoder 比 Encoder 多两点。** (1) **Masked / causal self-attention**——单向，训练时戴因果 mask 禁止第 $t$ 步看到 $>t$ 的位置（防止“抄答案”）；而 Encoder 是**双向**、可看全局。(2) 经典 enc-dec 里 Decoder 特有一层 **Cross-Attention**（见下）。

### Cross-Attention 的 Q/K/V 来自哪里？

- $Q$：来自 Decoder 上一个子层的输出。
- $K,V$：来自 Encoder 最后一层的输出。

它让 Decoder 在生成每个 token 时，动态读取输入序列中相关的位置。例如在翻译中，decoder 的当前输出位置可以关注对应的源语言词语。一句话：用 **Decoder 的 $Q$（当前生成需求）去检索 Encoder 的 $K,V$（已编码的源序列）**。

Decoder-only LLM 没有独立 Encoder，因此通常也没有这种 Encoder-Decoder Cross-Attention；多模态模型则可能使用 Cross-Attention 接收视觉或音频表示。

### Self-Attention 与传统 Seq2Seq Attention

传统 Bahdanau/Luong Attention 通常发生在 Encoder 和 Decoder 之间：Decoder 当前状态作为 query，读取 Encoder 的一组 hidden states，解决固定长度上下文瓶颈并学习源端到目标端的对齐。

**直觉：闭卷 vs 开卷。** 早期 Encoder-Decoder 把整句压成一个固定向量再生成，像**闭卷考试**靠背诵；Attention 则让解码的每一步都能**回头查看全部输入**，像**开卷考试**——翻译到某个词时就重点看原文对应的位置。

Self-Attention 则发生在同一序列内部，Q/K/V 来自同一组输入表示，用来建模序列内部任意位置之间的关系。Transformer 同时使用了两类机制：

- Encoder/Decoder 内部使用 self-attention。
- Encoder-Decoder 之间使用 cross-attention。

因此“传统 Attention”和“Self-Attention”不是简单的新旧替代关系，关键区别是 Q 与 K/V 是否来自同一序列。

### 原论文 FFN

FFN 对每个位置独立应用同一组参数：

$$
\operatorname{FFN}(x)
=\max(0,xW_1+b_1)W_2+b_2
$$

原论文结构为：

$$
d_{\text{model}}\rightarrow 4d_{\text{model}}
\rightarrow d_{\text{model}}
$$

并使用 ReLU。后续 BERT/GPT 常改用 GELU，Llama 等现代 LLM 常使用 SwiGLU。

---

## 2. 什么是位置编码？为什么必需？

### 30 秒回答

如果不加入位置信息，自注意力对输入顺序是置换等变的：打乱 token，输出也只会对应地打乱，无法区分“狗咬人”和“人咬狗”。位置编码用于告诉模型 token 的绝对位置或相对距离。

常见方法包括：

1. **Sinusoidal absolute position encoding**：用不同频率的正弦和余弦函数生成固定位置向量，与 token embedding 相加。
2. **Learned absolute position embedding**：为每个位置学习一个向量，如早期 GPT、BERT。
3. **Relative position bias**：直接给注意力分数加入与相对距离有关的偏置，如 T5、ALiBi。
4. **RoPE**：根据位置旋转 Q/K，使点积自然包含相对位置信息。

### 原论文的输入表示与正弦位置编码

原论文将 token embedding 缩放后与位置编码相加：

$$
x_{\text{input}}
=\sqrt{d_{\text{model}}}\,E(\text{token})+PE(\text{pos})
$$

乘以 $\sqrt{d_{\text{model}}}$ 用于调整 embedding 的数值尺度，使其与位置编码及网络内部表示处于更合适的量级。只把原因表述为“防止 PE 主导 embedding”过于绝对，因为实际尺度还取决于 embedding 初始化和训练。

原论文使用固定正弦/余弦编码：

$$
PE_{(pos,2i)}
=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos,2i+1)}
=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

不同维度对应不同频率。**为什么用 sin/cos 而不是 $1,2,3,\dots$？** 直接用递增的大整数会**淹没**词向量（词向量数值多在 $[-1,1]$）；sin/cos 把位置编成**不同频率的指针**——低维分量转得快（像秒针）、高维分量转得慢（像时针），取值恒在 $[-1,1]$，不同位置的“指针组合”唯一，且可编码任意长度的位置。

选择固定 sin/cos 的动机包括：

- 不增加可训练位置参数。
- 任意位置都可直接计算，因此形式上可用于训练长度外的位置。
- $PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性变换，有利于模型利用相对位移。

需要注意：**能计算更长位置不等于模型能可靠外推到更长序列**。原论文也测试了 learned position embedding，并报告二者结果相近。

### 绝对与相对位置的权衡

| 方法 | 优点 | 局限 |
| --- | --- | --- |
| 固定正弦 | 无额外可训练参数；可计算训练长度外的位置 | 外推效果不一定稳定；位置和语义直接相加 |
| 可学习绝对位置 | 简单；训练范围内表达灵活 | 通常受最大位置表限制；长度外泛化弱 |
| Relative Bias / ALiBi | 直接建模距离；常有较好长度外推 | 表达形式受偏置函数限制 |
| RoPE | 相对位置自然进入 QK 点积；实现高效 | 原始版本超出训练长度后退化；需要缩放策略 |

---

## 3. 详细介绍 RoPE

### 核心思想

RoPE（Rotary Position Embedding）不把位置向量加到 hidden state 上，而是根据位置 $m$ 对 Query 和 Key 的每对二维通道做**旋转**（把词向量看作平面里的箭头，位置越靠后、转的角度越大）：

$$
\begin{bmatrix}
x'_{2i}\\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
x_{2i}\\
x_{2i+1}
\end{bmatrix}
$$

其中不同维度使用不同频率：

$$
\theta_i=\text{base}^{-2i/d}
$$

关键性质是：

$$
\langle R_m q, R_n k\rangle
=q^\top R_{n-m}k
$$

因此 Q/K 的点积只依赖相对位置 $n-m$，同时每个表示仍保留自身绝对位置对应的旋转相位。直觉：给每个词的是**绝对**旋转角，但做 $Q\cdot K$ 点积时绝对位置**自动抵消、只剩相对距离 $n-m$**（语言里相对距离往往比绝对位置更重要）。

### 相比绝对位置编码

**优势**

- 相对位置信息直接作用于注意力分数。
- 不需要最大长度的位置 embedding 表。
- 邻近 token 的旋转差较小，通常具有合理的位置衰减归纳偏置。
- 实现简单，已广泛用于 Llama、Qwen、DeepSeek 等 decoder-only 模型。

**劣势**

- 长度外推不是自动成立的。推理位置远超训练范围时，相位分布发生偏移。
- 高频维度旋转很快，可能导致远距离信息不稳定。
- 扩展上下文通常需要 Position Interpolation、NTK-aware scaling、YaRN 等缩放方案，并可能需要长上下文继续训练。

### 高频追问：RoPE 如何扩长？

- **Position Interpolation**：把更长的位置压缩到训练过的位置范围，相当于降低旋转角度。
- **NTK-aware scaling**：按维度调整频率基数，尽量兼顾高频局部信息和低频长期信息。
- **YaRN**：对不同频率分段处理，并配合 attention scaling。
- 仅修改 RoPE 参数不等于模型真正学会长上下文利用，仍要检查检索、推理、困惑度以及真实长文任务。

---

## 3.1 KV Cache 是什么？为什么不能缓存 QK 分数？

### 30 秒回答

自回归**逐 token** 生成。历史 token 的 $K,V$ 一旦算出就**固定不变**，于是算一次就**存进显存缓存**；生成新 token 时只算它自己的 $Q_t,K_t,V_t$，并**复用缓存的** $K_{1..t-1},V_{1..t-1}$，避免把历史 token 重新乘 $W_K,W_V$ 提特征。

### 为什么不能缓存 QK（注意力分数）？

$Q_t$ 是当前新 token 专属的“搜索请求”，**每步都变**；$Q$ 一变，上一步的 $QK$ 分数就作废，必须用新的 $Q_t$ 与所有 $K$ 重算。所以 KV Cache 省下的是“**把历史 token 重新投影成 $K,V$**”的计算，而不是省 $QK$ 乘法。一句话：缓存的是“**被搜索的库（K/V）**”，不是“搜索结果”。

### 代价

KV Cache **极吃显存**，且随上下文长度、并发数线性增长，长上下文 / 高并发容易 OOM。这正是 MQA/GQA、PagedAttention、KV 压缩等方案的动机。

---

## 4. MHA、MQA、GQA 的区别

### 结构

设 Query 头数为 $H$：

- **MHA（Multi-Head Attention）**：每个 Query 头都有独立 K/V 头，$H_Q=H_K=H_V=H$。
- **MQA（Multi-Query Attention）**：所有 Query 头共享同一组 K/V，$H_K=H_V=1$。
- **GQA（Grouped-Query Attention）**：若干 Query 头共享一组 K/V，$1 < H_{KV} < H_Q$。

**具体到 32 个头**：MHA = **32 份**独立 K/V（缓存最大）；MQA = 32 个 $Q$ 共用 **1 份** K/V（缓存约缩到 $1/32$）；GQA = 分成 **8 组、每组 4 个头共享 1 份** K/V（缓存约 $1/4$），是当前主流折中。

标准 MHA 的完整形式为：

$$
\operatorname{head}_i
=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\operatorname{MultiHead}(Q,K,V)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,
\operatorname{head}_h)W^O
$$

通常每头维度 $d_k=d_v=d_{\text{model}}/h$。多个头可以在不同投影子空间中关注不同关系；其优势是表达多样性，而不是“天然更不容易过拟合”。

### 为什么要共享 K/V？

动机是 decode 阶段受**显存带宽墙（memory-bound）**：瓶颈不在算力，而在把庞大的 **KV Cache** 从显存搬到计算核心。自回归推理要为每层缓存所有历史 token 的 K/V。粗略地，KV Cache 大小为：

$$
2 \times L \times n_{\text{layers}} \times H_{KV}
\times d_{\text{head}} \times \text{bytes}
$$

因此把 KV 头从 32 降至 8，KV Cache 理论上约缩小到四分之一，也减少 decode 阶段的显存读取。

### 权衡

| 方法 | 质量 | KV Cache / 带宽 | 典型适用 |
| --- | --- | --- | --- |
| MHA | 表达能力强 | 最大 | 质量优先、较短上下文 |
| MQA | 可能损失质量 | 最小 | 极致推理吞吐 |
| GQA | 接近 MHA | 明显降低 | 当前主流折中 |

### 面试加分点

- 共享的是 K/V 投影头，不是 Query 头：$Q$ 代表“当下的思考需求”，需保持多样；$K,V$ 是历史静态信息，可压缩共享。MHA → MQA 缓存缩到约 $1/h$、吞吐暴涨但质量掉点；GQA（如 32 头 → 8 组）缓存约 $1/4$、质量 ≈ MHA。
- GQA 的优势主要体现在自回归 decode 和长上下文服务，不应只说“参数更少”。
- DeepSeek 的 MLA 进一步把 K/V 表示压缩到低维潜在向量，目标同样是降低 KV Cache，但机制不同于简单共享 KV 头。

---

## 5. Encoder-Only、Decoder-Only、Encoder-Decoder

| 架构 | Attention | 训练目标 | 擅长任务 | 代表 |
| --- | --- | --- | --- | --- |
| Encoder-Only | 双向 self-attention | Masked LM 等 | 分类、检索、序列标注、Embedding | BERT |
| Decoder-Only | Causal self-attention | Next-token prediction | 开放生成、对话、代码、in-context learning | GPT、Llama、Qwen、DeepSeek |
| Encoder-Decoder | Encoder 双向；Decoder 因果并 cross-attend encoder | 条件生成、span corruption | 翻译、摘要、结构化 seq2seq | T5、BART |

### 为什么通用 LLM 多为 Decoder-Only？

- 训练目标统一，任何文本都能转为 next-token prediction。
- 自回归生成接口自然，容易通过 prompt 统一大量任务。
- Scaling 路径和推理基础设施成熟。
- 这不代表 decoder-only 在所有任务都更优。高吞吐 embedding、分类或固定输入到输出任务，encoder 或 encoder-decoder 仍可能更高效。

**架构演进脉络（面试可一句话概括）**：Encoder-only（BERT）**没有被淘汰**，擅长**理解类**任务——文本转 embedding 做**检索 / 相似匹配**、**分类 / 情感**、**NER**，体积小、快、双向理解深；翻译等结构化 seq2seq 走 **Encoder-Decoder**（原始 Transformer、T5、BART）；现代通用 LLM 收敛到 **Decoder-only**（GPT、Llama、Qwen），把万物统一成 next-token 续写，利于 Scaling 与工程。

### Transformer 相比 LSTM 的优势与代价

- **并行性**：训练时 self-attention 可并行处理所有 token；LSTM 必须沿时间步递归。
- **长程路径**：单层全局 attention 中任意位置路径为 $O(1)$，LSTM 为 $O(n)$。
- **硬件利用率**：Transformer 主要由大规模矩阵乘法组成，适合 GPU/TPU。
- **代价**：标准 attention 对序列长度是二次计算和显存；自回归生成仍必须逐 token decode，并非所有阶段都完全并行。

LSTM 的门控缓解了普通 RNN 的梯度问题，但不能消除串行依赖；在小数据、流式处理、低延迟或有限设备场景中，RNN 仍可能有价值。

---

## 5.1 BERT

### 30 秒回答

BERT（Bidirectional Encoder Representations from Transformers）是 encoder-only 的双向预训练模型。它用 **Masked Language Modeling（MLM）** 随机遮盖部分 token 并预测它们，从而让每个位置都能同时看到左右上下文，得到深度双向表示。

### 两个预训练任务

- **MLM**：随机 mask 约 15% 的 token 让模型还原。这是 BERT 双向能力的来源——不同于 GPT 只能看左侧。
- **NSP（Next Sentence Prediction）**：判断句子 B 是否紧接句子 A。后续工作（如 RoBERTa）发现 NSP 收益有限，常被去掉。

### 为什么是双向？和 GPT 的区别

- BERT 用双向 self-attention，适合**理解类**任务：分类、NER、抽取式问答、句向量/检索。
- GPT 用 causal mask 单向自回归，适合**生成类**任务。
- BERT 不能直接做开放生成；用它通常是“预训练 + 下游微调”，给每个任务接一个 head。

### 追问：BERT 为什么不能像 GPT 那样直接生成文本？

因为 MLM 的训练目标是“填空”而非“续写”，且 attention 是双向的，没有自回归的因果结构。要生成需要 encoder-decoder（如 BART/T5）或 decoder-only 架构。

### 面试怎么说？

> BERT 是 encoder-only 双向模型，用 Masked LM 让每个 token 看到双向上下文，擅长分类、NER、检索这类理解任务；GPT 是单向自回归，擅长生成。BERT 一般是预训练加下游微调，本身不做开放生成。

---

## 6. 什么是 Scaling Laws？

### 30 秒回答

Scaling Laws 指模型损失或性能随模型参数量 $N$、数据量 $D$ 和计算量 $C$ 呈现近似幂律关系，例如：

$$
L(N,D)\approx L_\infty + aN^{-\alpha}+bD^{-\beta}
$$

核心启示不是“模型越大越好”，而是在固定计算预算下，需要合理分配模型大小和训练 token。早期工作发现扩大模型、数据和算力可预测地降低损失；Chinchilla 进一步指出很多模型训练不足，在固定 FLOPs 下，使用更小模型配更多数据可能更优。

### 对研发的指导意义

- 用小规模实验拟合曲线，预测大规模训练的收益，减少盲目试错。
- 进行 compute-optimal 的参数量与 token 配比。
- 预算还要覆盖数据质量、后训练、推理成本和部署约束。
- 测试损失的平滑改善不保证所有下游能力同步改善。

### 关于“涌现能力”

不能给出一个通用的参数阈值。能力出现取决于模型、数据、训练目标、任务和评价指标。某些“突然涌现”可能来自：

- 指标是离散或非线性的，例如 exact match。
- 小模型能力低于测量噪声或随机基线。
- In-context learning 在规模增大后跨过可观察阈值。

更严谨的说法是：部分能力随规模平滑增强，但被某些指标观察成突变；也有复杂能力可能在足够规模和数据多样性下出现明显相变，目前没有统一理论和固定规模。

---

## 7. 常见解码策略

### Greedy Search

每一步选择概率最大的 token：

$$
y_t=\arg\max_y p(y\mid y_{<t},x)
$$

优点是快、确定、易复现；缺点是局部最优，容易重复或得到平庸答案。适合分类式输出、确定性抽取和部分代码任务。

### Beam Search

每一步保留累计 log probability 最大的 $B$ 个候选序列。相比 greedy 更接近全局高概率序列。

- 优点：翻译、语音识别等目标相对确定的任务中常有效。
- 缺点：成本约随 beam size 增长；开放生成中可能偏好安全、短或重复文本；高概率不等于高质量。
- 常配合长度归一化、重复惩罚和 early stopping。

### Top-K Sampling

只保留概率最高的 K 个 token，重新归一化后采样。

- 优点：过滤长尾低质量 token；多样性可控。
- 缺点：固定 K 不适应概率分布的尖锐程度。某一步前 5 个已覆盖 99%，另一步前 5 个可能只覆盖 40%。

### Nucleus / Top-P Sampling

选择累计概率达到 $p$ 的最小 token 集合，再归一化采样。

- 优点：候选集合大小动态变化，通常比固定 Top-K 灵活。
- 缺点：仍可能产生事实错误；结果有随机性。

### 其他常见参数

- `temperature < 1` 使分布更尖锐，`temperature > 1` 增加随机性。
- repetition / frequency penalty 抑制重复，但过强会破坏术语一致性。
- constrained decoding 可用 grammar、JSON Schema、正则或状态机保证结构合法。
- 推理模型常使用多样采样加 verifier、majority vote 或 best-of-N。

---

## 8. Tokenization：BPE 与 WordPiece

### 什么是 Tokenization？

Tokenizer 把文本映射为有限词表中的 token ID。子词算法在“整词词表太大”和“字符序列太长”之间折中，还能处理未登录词。

### BPE

从字符或字节开始，反复合并语料中最高频的相邻 token 对，直到达到目标词表大小。GPT 系列常使用 byte-level BPE。

### WordPiece

同样迭代合并子词，但通常按语言模型似然或类似于
$\frac{p(ab)}{p(a)p(b)}$ 的分数选择合并，而不只是原始频次。BERT 使用 WordPiece。

| 维度 | BPE | WordPiece |
| --- | --- | --- |
| 合并依据 | 相邻对频率 | 使语料似然改善的分数 |
| 实现 | 简单、常见 | 训练准则更贴近概率模型 |
| OOV | byte-level BPE 可基本消除 | 常依赖字符覆盖和 `[UNK]` |

### 面试加分点

- Tokenizer 会影响上下文有效长度、不同语言的成本、代码表示和数字能力。
- 中文不等于“一字一个 token”；结果取决于词表和训练语料。
- 词表越大，序列通常越短，但 embedding/output head 更大，稀有 token 学习也可能不足。
- SentencePiece 是一种不依赖预分词的实现框架，可训练 BPE 或 Unigram 模型，不应直接与 BPE 当作同一层级概念比较。

---

## 9. NLP 和 LLM 的共同点与区别

### 推荐回答

NLP 是研究和处理人类语言的整个领域，LLM 是当前 NLP 中以大规模预训练 Transformer 为核心的一类技术路线。两者不是替代关系。

传统 NLP 往往针对分词、NER、分类、翻译等单任务设计模型、特征和监督数据；LLM 通过大规模自监督预训练获得通用语言表示，再通过 prompt、in-context learning、SFT 或 RL 适配大量任务。

共同点是都关注语言表示、理解、生成、评估、数据偏差和泛化。不同点主要在：

- **范式**：任务专用训练 vs 通用预训练加适配。
- **接口**：固定标签/结构 vs 自然语言统一接口。
- **能力来源**：任务标注数据 vs 大规模预训练、规模化和后训练。
- **工程问题**：LLM 更强调分布式训练、推理服务、对齐、工具使用和安全。

---

## 10. Transformer 里的激活函数

激活函数的通用对比（Sigmoid/Tanh/ReLU/Dying ReLU 等）见 [01. 基础与神经网络机制](01-foundations.md)。这里只讨论现代 Transformer FFN 常用的几种及其取舍。

### GELU

$$
\operatorname{GELU}(x)=x\Phi(x)
$$

平滑地按输入大小进行门控，BERT、GPT-2/3 等常用。

### SiLU / Swish

$$
\operatorname{SiLU}(x)=x\sigma(x)
$$

平滑、非单调，现代模型中常见。

### GLU、SwiGLU

$$
\operatorname{SwiGLU}(x)
=\operatorname{SiLU}(xW_1)\odot(xW_2)
$$

随后通常再经过输出投影。门控分支可以动态控制信息通过，经验上在相近计算预算下优于普通 ReLU/GELU FFN。Llama 等模型采用 SwiGLU。

**为什么 FFN 中间维度有时不是 $4d$？**

SwiGLU 有两个输入投影。为保持参数量或 FLOPs 接近普通 FFN，常相应调小中间维度，并对齐硬件友好的倍数。

---

## 11. MoE 如何扩大参数而不同比例增加推理成本？

### 30 秒回答

Mixture-of-Experts 通常把 Transformer 的部分 FFN 替换为多个 expert。Router 根据每个 token 的 hidden state 计算 expert 分数，只把 token 分发给 Top-K 个 expert：

$$
p(e\mid x)=\operatorname{softmax}(W_rx),\qquad
y=\sum_{e\in\operatorname{TopK}}p(e\mid x)E_e(x)
$$

模型可以有很大的总参数量，但每个 token 只激活少量 expert，因此单 token FLOPs 不随总 expert 数同比增长。

### 关键挑战

- **负载均衡**：热门 expert 过载，其他 expert 学不到。
- **通信**：Expert Parallelism 需要 all-to-all 分发 token，可能受网络带宽限制。
- **Capacity**：单 expert 容量有限，溢出 token 可能被丢弃或重新路由。
- **训练稳定性**：Router 可能塌缩，需要辅助负载均衡损失或无辅助损失的动态 bias。
- **部署**：虽然激活参数少，总权重仍需存储；小 batch 下难以充分利用各 expert。

### DeepSeekMoE 的面试点

- 细粒度 expert 划分，提高组合灵活性。
- Shared Experts 捕获通用知识，Routed Experts 学习差异化知识。
- DeepSeek-V3 使用 auxiliary-loss-free load balancing，减少辅助损失对主目标的干扰。

---

## 12. 高频综合追问

### 为什么用 LayerNorm 而不是 BatchNorm？

LayerNorm 对**单个样本内部的所有特征维**归一化，与 batch 大小、序列长度**无关**；BatchNorm 则对一个 batch 内所有样本的同一维求统计。

Transformer 更适合 LayerNorm，主要因为：

- NLP 序列长度不同，padding 和 token 分布会影响 batch 统计。
- 小 batch 或自回归推理时，BatchNorm 统计不稳定或训练/推理不一致。
- LayerNorm 对每个样本独立，适合变长序列和逐 token 推理。

### 为什么当前 LLM 多使用 Pre-Norm？

残差块记为 $x_{\text{out}}=x+\operatorname{SubLayer}(\cdot)$，区别在于 LayerNorm 放在残差之前还是之后：

- **Post-Norm**（原论文）：$x_{\text{out}}=\operatorname{LayerNorm}(x+\operatorname{SubLayer}(x))$，归一化在残差**之后**；深层堆叠时底层**梯度易消失**、训练不稳，往往需要小心的 **LR warmup**。有时最终表示能力更强，但深层训练更敏感。
- **Pre-Norm**（现代 LLM 标配）：$x_{\text{out}}=x+\operatorname{SubLayer}(\operatorname{LayerNorm}(x))$，归一化在子层**之前**；残差主干是一条**无阻碍的“高速公路”**，梯度可近乎**无损回传**，深层训练更稳定。

### RMSNorm 和 LayerNorm 有何区别？

LayerNorm 同时减均值、除标准差；RMSNorm 只按均方根缩放，不做中心化：

$$
\operatorname{RMSNorm}(x)
=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot g
$$

RMSNorm 更简单、计算略省，现代 decoder-only LLM 中很常见。

### 为什么使用 Causal Mask？

保证位置 $t$ 只能看到 $\le t$ 的 token，使训练目标和自回归推理一致，避免标签泄漏。

### 原始 Transformer 如何正则化和稳定训练？

原论文用于正则化的主要方法是：

- Dropout：作用于各子层输出，以及 embedding 与位置编码之和。
- Label Smoothing：把 one-hot 目标软化，降低过度自信并改善泛化。

此外，原论文使用 Adam、学习率 warmup 和随后按步数衰减来稳定优化。它使用的不是 AdamW；weight decay 可以用于后续 Transformer 训练，但不应当作原论文配置。LayerNorm 的主要作用也是稳定优化；causal/padding mask 用于表达结构和屏蔽非法位置，它们不属于典型正则化方法。

### Self-Attention 与 RNN 的复杂度如何比较？

对序列长度 $n$、表示维度 $d$：

- Self-Attention 每层计算复杂度约为 $O(n^2d)$，注意力矩阵显存为 $O(n^2)$。
- RNN 每层计算复杂度约为 $O(nd^2)$，但沿 $n$ 个时间步串行。

**别把 $O(n^2)$ 简单理解成“更慢”。** 每个 Q 都要和全部 $n$ 个 K 打一次分，得到 $n\times n$ 注意力矩阵，计算与显存都是 $O(n^2)$，长序列显存爆炸（$n=10^5\Rightarrow10^{10}$）。但这 $n^2$ 次打分**彼此独立、可并行**（GPU 擅长），任意两 token 间**路径长度 $O(1)$**（没有 RNN 那种长程信息瓶颈）；RNN 虽是 $O(n)$ 却**必须串行**（第 $t$ 步等第 $t-1$ 步）。所以只要显存够，Transformer 实际更快、长依赖也更好。

严格的训练显存还必须计入每层激活，不能简单把 RNN 总空间写成 $O(d)$。当 $n<d$ 时，self-attention 的单层计算甚至可能更便宜；当 $n$ 很长时，$n^2$ 项成为瓶颈。

例如序列从 2048 增长到 4096，标准注意力矩阵元素数约增加到 4 倍。这正是长文本训练和 prefill 容易变慢、占显存的原因。

### Cross-Entropy 在做什么？

直觉：逼模型当“**猜下一个词的神算子**”——每个位置在词表上输出一个概率分布，真实的下一个词概率越高、loss 越小。形式上对真实下一个 token $y_t$ 最小化负对数似然：

$$
\mathcal L=-\sum_t \log p_\theta(y_t\mid y_{<t})
$$

它等价于最小化经验数据分布与模型分布的交叉熵；预训练本质就是用 CE 逼模型的预测分布**逼近真实语言分布**。Perplexity 通常为平均 token loss 的指数，但不同 tokenizer 下不可直接公平比较。

### 修改开源模型时如何读取 Attention Weights？

不要假设模型调用一定返回完整 attention matrix。现代实现可能使用 FlashAttention、PyTorch SDPA 或 fused kernel，它们为了节省 $O(n^2)$ 显存，通常不会物化或返回注意力权重；即使设置 `output_attentions=True`，也可能触发慢速 eager fallback，或因具体模型实现而不受支持。

**顺带点明 FlashAttention 是什么。** 它是**实现层优化，数学结果与普通 Attention 完全一致**：靠**分块（tiling）+ 在线 softmax（online softmax）+ 边算边扔**，在极快的片上 SRAM 里把 $QK^\top$、softmax、乘 $V$ 一气呵成，**不落地那张 $n\times n$ 矩阵**，因此显存从 $O(n^2)$ 降到 $O(n)$、速度提升约 2–4×（属“访存受限”问题的优化）。正因为它不显式生成 $n\times n$ 注意力矩阵，想 `output_attentions=True` 拿权重才会 OOM 或退化成慢速 eager attention。

进行可视化或干预前应：

1. 检查具体模型版本和 attention backend。
2. 阅读 forward 返回值与 `output_attentions` 文档。
3. 必要时切换 eager attention 或注册 hook。
4. 用短序列验证 shape 和数值，评估切换实现是否改变精度、速度和显存。

这是实现层面的行为，不应概括成某个模型“Attention Weights 永远为 None”。
