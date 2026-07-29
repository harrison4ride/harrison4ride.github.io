# 02. NLP、RNN 与词向量

[返回目录](README.md)

## 1. 为什么使用 RNN？

RNN 的核心价值是**记忆**。它用共享参数递归处理序列：

$$
h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$

$$
o_t=W_{hy}h_t+b_y
$$

直观理解这条递推：新记忆 $h_t$ 由**当前输入 $x_t$** 和**上一步的旧记忆 $h_{t-1}$** 共同决定；$W_{xh},W_{hh}$ 决定"新输入 vs 旧记忆"各占多重；$\phi$（通常是 $\tanh$）把数值压到 $[-1,1]$，防止逐步累加发散。示意图里那一长排 node 其实是**同一套单元在不同时间步的"展开(unroll)"**（同一台机器在重播），并不是多个不同单元；每步的输入 $x_t$ 通常是**词向量(embedding)**。

**逐步走一遍**：以 "我 / 今天 / 吃了 / 苹果" 为例。第 1 步只有输入 "我"，产生初始记忆 $h_1$；第 2 步输入 "今天"，与旧记忆 $h_1$ 按权重相加再经 $\phi$ 揉合，得到同时含 "我" 和 "今天" 的 $h_2$；第 3 步输入 "吃了" 与 $h_2$ 合成 $h_3$……像**接力赛**，每一棒手里的接力棒都带着前面所有棒的痕迹，因此只要链条不断，理论上第 1 步的信息能一路传到最后一步。

因此 RNN 有三点优势：

- **带记忆理解上下文**：把历史信息压进 hidden state $h_t$ 带到当前步，当前预测能看到之前发生了什么。
- **天然处理变长序列**：逐步处理，来一个 token 算一步，序列多长都能吃。
- **参数共享**：无论序列多长都复用同一套权重，参数量不随序列长度增长，模型不会因序列变长而变大。

适合变长序列、因果预测和流式处理。

### 典型用途

- Language modeling。
- 序列分类和标注。
- 时间序列。
- 语音、传感器和在线流式推理。

Transformer 成为主流后，RNN 仍可用于低延迟流式、小模型和严格内存场景。

---

## 2. Vanilla RNN 的局限

RNN 训练用 **BPTT（Backpropagation Through Time）**：把展开的 $n$ 步当成一个 $n$ 层的深网络，前向传到最后算 Loss，再让误差沿时间**反向**逐步传回、更新那套共享的 $W$。由此带来两大痛点：

- **梯度消失/爆炸**：因为共享 $W$，反向传播每经过一个时间步就**连乘一次 $W$**。$W$ 的量级 $<1$，连乘后梯度指数衰减到 $0$，远端节点几乎收不到梯度、"摆烂"学不到长程关系；$W$ 的量级 $>1$，连乘后梯度爆炸。
  - **具体数字**感受连乘的威力：若每步回传的量级约 $0.9$，回传 100 步就是 $0.9^{100}\approx2.6\times10^{-5}$，梯度传到最前面几乎归零，早期时间步"学不动"；若量级约 $1.1$，$1.1^{100}\approx1.4\times10^{4}$，梯度爆炸。
- **信息瓶颈**：固定大小的 hidden state 装不下很长的历史，新信息会不断冲刷、覆盖旧信息。
  - 直觉：用固定大小的 hidden state 装长历史，像**用一句话总结一整本书**，后面的内容会不断把前面的冲刷掉。

其他局限：

- 时间步必须串行，训练难以并行。
- 长程依赖的信息路径长度为 $O(n)$。
- 隐状态容量和记忆时间缺少显式控制。

从数学上说，若 recurrent Jacobian 的谱范数持续小于 1，梯度随时间指数衰减；持续大于 1 则可能爆炸。

---

## 3. LSTM 公式

LSTM 的核心是 **门控 + 一条独立的细胞状态 $c_t$「传送带」**：$c_t$ 默认只做加法、信息近乎无损地一路向前流，门则决定往这条传送带上删什么、加什么、读什么。

先记住两个通用符号：

- $\sigma$（Sigmoid）= **门控开关**，输出 $0\sim1$（$0$ 全关/全忘，$1$ 全开/全留）。
- $\tanh$ = **内容调节器**，输出 $-1\sim1$（可增可减，正=强化、负=抵消，避免只加不减导致数值爆炸）。

给定输入 $x_t$、上一时刻 hidden state $h_{t-1}$ 和 cell state $c_{t-1}$：

$$
f_t
=\sigma(W_f[x_t;h_{t-1}]+b_f)
$$

$$
i_t
=\sigma(W_i[x_t;h_{t-1}]+b_i)
$$

$$
\tilde c_t
=\tanh(W_c[x_t;h_{t-1}]+b_c)
$$

$$
c_t
=f_t\odot c_{t-1}
+i_t\odot\tilde c_t
$$

$$
o_t
=\sigma(W_o[x_t;h_{t-1}]+b_o)
$$

$$
h_t=o_t\odot\tanh(c_t)
$$

**读公式的套路**：凡是看到 $[x_t;h_{t-1}]$，都是"把眼前输入和上一步的短期记忆拼起来综合考虑"；再配上上面 $\sigma$=开关、$\tanh$=内容两条，六个公式其实只有**四步**：

- ① **决定忘掉多少**：$f_t$ 给旧记忆打一个保留比例。
- ② **准备新内容**：$\tilde c_t$ 是模型打的**新草稿**，$i_t$ 决定这份草稿写进去多少。
- ③ **更新长期记事本**：$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$，旧的擦一部分 + 新的写一部分，是**加法**。
- ④ **决定这一步往外说多少**：$o_t$ 过滤 $\tanh(c_t)$ 得到 $h_t$。

三个门各管一件事：

- **遗忘门 $f_t$**：旧记忆 $c_{t-1}$ 留多少。
- **输入门 $i_t$**：新候选内容 $\tilde c_t$ 记多少。
- **输出门 $o_t$**：当前这一步从记忆里吐出多少给 $h_t$。

关键在细胞更新 $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$ 是**加法**——旧记忆按 $f_t$ 保留、新内容按 $i_t$ 写入，两路相加放回传送带，而不是像 Vanilla RNN 那样每步把状态整体非线性重写。

实际实现通常把四个 affine transformation 合并成一次大矩阵乘法，提高效率。

---

## 4. LSTM 为什么比 Vanilla RNN 更适合长依赖？

Vanilla RNN 每一步都对 hidden state 做完整非线性变换。LSTM 的 cell state 使用加法更新，并由 gate 控制：

$$
\frac{\partial c_t}{\partial c_{t-1}}
\approx f_t
$$

关键区别在于：cell state 的更新是**加法**而非像 Vanilla RNN 那样连乘 $W$。因此只要遗忘门 $f_t\approx1$，$\partial c_t/\partial c_{t-1}\approx1$，梯度就能沿这条"传送带高速公路"**几乎无损地**一路回传到很早的时间步，缓解梯度消失，从而学到长程记忆；gate 还允许模型自主选择写入、保留和输出哪些信息。

LSTM 只是缓解，不是彻底解决：

- 很长序列中 gate 乘积仍可能衰减。
- 仍然串行。
- 每步有四组门，计算和参数多于 vanilla RNN。
- 所有历史仍压缩在有限状态中。

GRU 用更少 gate 做相似折中，参数更少但表达取舍依任务而异。

---

## 4.1 LSTM 参数量

LSTM 有输入门、遗忘门、输出门和候选状态四组变换。输入维度为 $d$，隐藏维度为 $h$ 时，每组包含：

- 输入权重：$h\times d$。
- 循环权重：$h\times h$。
- 偏置：$h$。

因此单层、单向 LSTM 参数量为：

$$
4(hd+h^2+h)
$$

也可把输入和隐藏状态拼接，写成：

$$
4h(d+h)+4h
$$

### 工程细节

不同框架的偏置实现可能不同。例如 PyTorch LSTM 通常为输入侧和隐藏侧分别保存一组 bias，因此参数量会变为：

$$
4(hd+h^2+2h)
$$

双向 LSTM 还要乘以 2；堆叠层中，后续层输入维度取决于前一层方向数和 hidden size。面试计算前应先说明实现约定。

---

## 4.2 GRU

GRU（Gated Recurrent Unit）是 **LSTM 的轻量版**，做了两处精简：

- **状态合并**：把 LSTM 的 $c_t$ 和 $h_t$ 合并成单一隐状态 $h_t$，不再单独维护细胞状态。
- **门三变二**：只保留两个门。**更新门 $z_t$** 相当于把 LSTM 的遗忘门 + 输入门合成一个"跷跷板"——按下方公式，写入 $z_t$ 比例的新内容、保留 $1-z_t$ 比例的旧记忆，二者互补（一个变大另一个必然变小）；**重置门 $r_t$** 在算新候选记忆前决定用多少旧记忆，可清零以"切断上下文"。

$$
z_t=\sigma(W_z[x_t;h_{t-1}])
$$

$$
r_t=\sigma(W_r[x_t;h_{t-1}])
$$

$$
\tilde h_t=\tanh(W_h[x_t;r_t\odot h_{t-1}])
$$

$$
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t
$$

优点：参数比 LSTM 少约 $1/4$、训练更快、小数据上更不易过拟合，多数任务效果与 LSTM 相当；具体哪个更好依任务而异。

---

## 5. 如何缓解 RNN 梯度消失或爆炸？

### 梯度消失

- 使用 LSTM/GRU 的 gated additive memory。
- 合理初始化 recurrent matrix，例如 orthogonal initialization。
- 使用 residual/skip connection。
- LayerNorm 等归一化方法。
- 缩短依赖路径，例如引入 Attention。
- 对任务重新设计，使监督信号更密集。

### 梯度爆炸

- Gradient clipping，常按 global norm：

$$
g\leftarrow
g\cdot\min\left(1,\frac{\tau}{\|g\|}\right)
$$

- 降低学习率。
- 稳定初始化和归一化。
- 检查异常序列与 loss 数值。

Gradient clipping 主要处理爆炸，不能恢复已经消失的梯度。Truncated BPTT 节省计算和显存，但同时截断更远的信用分配，也不是长依赖的根本解法。

---

## 6. 什么是 Attention，为什么需要它？

Attention 根据 query 与一组 key 的相关性，对 value 做动态加权：

$$
\alpha_i
=\operatorname{softmax}_i(\operatorname{score}(q,k_i))
$$

$$
c=\sum_i\alpha_i v_i
$$

### Seq2Seq 中的动机

早期 Encoder-Decoder RNN 把整个输入压缩为最后一个固定长度向量，长句容易形成**信息瓶颈**（正是 §2 里 RNN 固定大小 hidden state 的老问题）。Attention 直接绕开瓶颈：让 Decoder 的**每一步都能回看全部输入**，按需**软对齐(soft alignment)** 到相关位置，而不是只依赖一个被压扁的向量。

- 动态选择当前输出所需的输入位置。
- 改善长序列信息传递。
- 学习翻译等任务中的软对齐。

至于 Q/K/V 的完整机制（多头、缩放、位置编码等），在 Transformer 相关章节展开，这里不重复。

### 常见 Score

- Dot product：$q^\top k$。
- Scaled dot product：$q^\top k/\sqrt d$。
- Additive/Bahdanau：

$$
v^\top\tanh(W_qq+W_kk)
$$

Self-Attention 的 Q/K/V 来自同一序列；Cross-Attention 的 Q 与 K/V 来自不同序列。Transformer 进一步用 Attention 取代主要 recurrent computation。

---

## 7. Language Model 的原理

Language Model 对 token 序列分配概率。根据 chain rule：

$$
p(w_{1:T})
=\prod_{t=1}^{T}
p(w_t\mid w_{<t})
$$

自回归神经语言模型学习 next-token distribution，训练时最小化负对数似然：

$$
\mathcal L
=-\sum_{t=1}^{T}
\log p_\theta(w_t\mid w_{<t})
$$

语言模型可用于生成，也可用 log-likelihood 或 perplexity 评价模型对文本的预测能力。

---

## 8. N-Gram Language Model

N-Gram 使用 $(n-1)$ 阶 Markov 假设：

$$
p(w_t\mid w_{<t})
\approx
p(w_t\mid w_{t-n+1:t-1})
$$

Maximum Likelihood count estimate：

$$
p(w_t\mid h)
=\frac{\operatorname{count}(h,w_t)}
{\operatorname{count}(h)}
$$

### 优点

- 简单、可解释、训练和查询快。
- 小数据和受限领域中可作为强 baseline。

### 局限

- 数据稀疏，未见 N-Gram 概率为 0。
- 上下文长度固定，无法泛化到语义相似历史。
- 词表和 $n$ 增大时存储迅速增长。

### Smoothing

- Add-k/Laplace。
- Backoff 与 interpolation。
- Good-Turing。
- Kneser-Ney，利用词在多少种上下文中出现来估计 continuation probability。

---

## 9. Word2Vec

Word2Vec 是学习静态词向量的一组浅层神经方法，主要包括 CBOW 和 Skip-Gram。它利用 distributional hypothesis：出现在相似上下文中的词具有相似语义。

### CBOW

根据周围上下文预测中心词：

$$
\max_\theta
\sum_t
\log p(w_t\mid w_{t-c:t-1},w_{t+1:t+c})
$$

通常将上下文 embedding 求和或平均。CBOW 训练快，对高频词表示通常较稳定。

### Skip-Gram

根据中心词预测窗口内上下文：

$$
\max_\theta
\sum_t\sum_{\substack{-c\le j\le c\\j\ne0}}
\log p(w_{t+j}\mid w_t)
$$

Skip-Gram 为每个中心词产生多个训练 pair，对低频词通常更有利，但训练成本更高。

### Full Softmax

给定中心词 $w$，预测上下文词 $c$：

$$
p(c\mid w)
=\frac{\exp(u_c^\top v_w)}
{\sum_{c'\in V}\exp(u_{c'}^\top v_w)}
$$

分母遍历整个词表，词表很大时昂贵。

---

## 10. Negative Sampling

Negative Sampling 把多分类预测近似为区分真实 word-context pair 与噪声 pair 的多个二分类任务。

对正样本 $(w,c)$ 和 $K$ 个负样本 $n_k$：

$$
\log\sigma(u_c^\top v_w)
+\sum_{k=1}^{K}
\log\sigma(-u_{n_k}^\top v_w)
$$

训练最大化该目标，或最小化其负值。

### 负样本如何采？

经典 Word2Vec 使用平滑后的 unigram distribution：

$$
P_{\text{neg}}(w)
\propto
\operatorname{count}(w)^{3/4}
$$

$3/4$ 次幂降低极高频词的支配，同时仍比均匀采样更多抽到常见词。

### 为什么有效？

- 每个训练 pair 只更新一个正词和少量负词，从 $O(|V|)$ 降到约 $O(K)$。
- 迫使真实共现 pair 的点积增大，噪声 pair 的点积减小。

Negative Sampling 的目标不是精确计算 normalized language-model probability，而是高效学习有用 embedding。若任务需要准确概率，应使用 full softmax、hierarchical softmax 或其他概率模型。

### 其他 Word2Vec 技巧

- 高频词 subsampling，减少大量低信息 stop words。
- Dynamic context window。
- 一个词通常有 input embedding 和 output embedding 两套参数；下游可选择其一或组合。
- Word2Vec 是 static embedding，同一词在不同上下文中表示相同，无法处理一词多义；contextual embedding 可缓解。

---

## 11. Stemming vs Lemmatization

两者都是把词变成更基础的形式，减少词形变化带来的稀疏性。

### Stemming

Stemming 用规则粗暴截断词缀：

```text
playing -> play
studies -> studi
```

优点是快、简单；缺点是结果可能不是合法单词。

### Lemmatization

Lemmatization 结合词典和词性，把词还原成 lemma：

```text
better -> good
running -> run
```

优点是更准确、可读；缺点是需要词性分析和语言资源，速度较慢。

### 什么时候用？

- 搜索、传统文本分类、关键词匹配：可以用 stemming 或 lemmatization。
- 现代 Transformer / LLM：通常直接使用 tokenizer，不一定需要手工 stemming。

面试一句话：

> Stemming 是按规则砍词尾，快但粗；lemmatization 是按语言学还原原词，准但慢。
