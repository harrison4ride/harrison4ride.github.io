# 第 7 章　梯度与初始化（Gradients and initialization）

[返回目录](README.md)

> **本章一句话**：要用 SGD 训练神经网络，每一步都得算出“损失对每个参数的梯度”——**反向传播**让这件事算得又快又省；而开训之前如何**初始化参数**，直接决定梯度会不会爆炸或消失，进而决定训练能不能稳定跑起来。

> **读这章的收获**：彻底搞懂反向传播是怎么用“前向存中间量 + 反向传中间梯度”把链式法则跑出来的、为什么它高效；理解梯度爆炸/消失是怎么由方差逐层放大/缩小造成的，以及 He 初始化（$\sigma^2 = 2/D_h$）凭什么能把它压住。

---

## 本章速览

- **要解决两个问题**：（1）如何**高效**计算损失对**海量参数**的梯度；（2）如何**初始化**参数让训练稳定。当今最大模型有约 $10^{12}$ 个参数，每次迭代、每个样本都要对每个参数求一次梯度。
- **梯度从哪来**：SGD 的每步更新都需要 $\partial \ell_i/\partial \boldsymbol{\beta}_k$ 和 $\partial \ell_i/\partial \boldsymbol{\Omega}_k$。反向传播就是专门高效算这些导数的算法。
- **两条直觉**：（1）一个权重的影响 = 它**乘的那个源神经元的激活值**有多大 → 所以要先跑一遍网络把激活值都存下来（前向传播）；（2）改一个参数会像涟漪一样一路传到损失 → 反向算时可以**复用**前一步已经算好的量。
- **反向传播 = 前向 + 反向两遍**：前向把所有中间量 $\mathbf{f}_k,\mathbf{h}_k$ 算出并存好；反向从输出端开始，沿网络倒着推，每步只是“乘权重转置 + 按 ReLU 阈值过滤”。
- **为何高效**：最费力的运算只是矩阵乘法（加法和乘法），且反向时**绝大多数项上一步已经算过**，无需重算。代价是费内存——前向的所有中间量都得留着。
- **现成框架代劳**：PyTorch / TensorFlow 通过**算法微分**自动求导，整个反向传播浓缩成一行 `loss.backward()`。
- **初始化是命门**：若权重方差太小，激活/梯度逐层指数衰减 → **梯度消失**；太大则逐层指数放大 → **梯度爆炸**。用方差传播推导出的 **He 初始化** $\sigma^2 = 2/D_h$ 能让方差逐层保持稳定。

---

## 7.1 问题定义（Problem definitions）

考虑一个有三层隐藏层的网络 $f[\mathbf{x},\boldsymbol{\phi}]$，输入为向量 $\mathbf{x}$，参数为 $\boldsymbol{\phi}$：

$$
\begin{aligned}
\mathbf{h}_1 &= a[\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0\mathbf{x}] \\
\mathbf{h}_2 &= a[\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1\mathbf{h}_1] \\
\mathbf{h}_3 &= a[\boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2\mathbf{h}_2] \\
f[\mathbf{x},\boldsymbol{\phi}] &= \boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3\mathbf{h}_3 ,
\end{aligned}
$$

其中 $a[\cdot]$ 把激活函数逐元素作用到输入上。参数 $\boldsymbol{\phi} = \{\boldsymbol{\beta}_0,\boldsymbol{\Omega}_0,\dots,\boldsymbol{\beta}_3,\boldsymbol{\Omega}_3\}$ 由每层之间的**偏置向量** $\boldsymbol{\beta}_k$ 和**权重矩阵** $\boldsymbol{\Omega}_k$ 组成。

训练用的损失是每个样本损失项 $\ell_i$ 的总和（$\ell_i$ 通常是“给定预测后真值的负对数似然”，例如最小二乘 $\ell_i = (f[\mathbf{x}_i,\boldsymbol{\phi}] - y_i)^2$）：

$$
L[\boldsymbol{\phi}] = \sum_{i=1}^{I}\ell_i .
$$

最常用的优化算法是**随机梯度下降（SGD）**：

$$
\boldsymbol{\phi}_{t+1} \leftarrow \boldsymbol{\phi}_t - \alpha \sum_{i\in\mathcal{B}_t}\frac{\partial \ell_i[\boldsymbol{\phi}_t]}{\partial \boldsymbol{\phi}} ,
$$

其中 $\alpha$ 是学习率，$\mathcal{B}_t$ 是第 $t$ 步这一批（batch）的样本下标。要更新参数，就必须对每层 $k$、每个样本 $i$ 算出：

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k} \quad\text{和}\quad \frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k} .
$$

本章前半部分讲**反向传播**（高效算这些导数），后半部分讲**参数初始化**（让训练稳定）。

---

## 7.2 计算导数（Computing derivatives）

导数告诉我们“参数动一点点，损失会怎么变”。反向传播算的就是这些导数。数学细节有点绕，先用两条直觉打底。

**观察 1（权重的影响 ∝ 源神经元激活）**：每个权重（$\boldsymbol{\Omega}_k$ 的一个元素）把**源**隐藏单元的激活值乘起来，加到**目标**隐藏单元上。所以对这个权重做一点小改动，效果会被**源单元的激活值**放大或缩小。

> 文字配图：把网络想成一张布满箭头的图，每个箭头是一个权重。某个箭头从“第 1 层第 2 个单元”出发——如果这个源单元的激活值翻倍，那么轻微调这个权重对下游的影响也翻倍。**结论：要算权重的导数，先得把各隐藏层的激活值算出来并存好。这一步就叫前向传播（forward pass）。**

**观察 2（涟漪效应 → 可复用）**：改动一个偏置或权重，会像涟漪一样往后传——先改它的目标单元，目标单元再改下一层，一直传到输出，最后传到损失。所以要知道“一个参数怎样改变损失”，就得知道“后面每一层的变化怎样改变它的后继层”。

关键在于：这些“后继层如何被前一层影响”的量，在计算**同层或更早层**的其他参数时**会被反复用到**。因此**只算一次、然后复用**即可。以喂入 $\mathbf{h}_3,\mathbf{h}_2,\mathbf{h}_1$ 的权重为例：

- 算喂入 $\mathbf{h}_3$ 的权重对损失的影响，需要：(i) $\mathbf{h}_3$ 变化如何改输出 $f$；(ii) $f$ 变化如何改损失 $\ell$。
- 算喂入 $\mathbf{h}_2$ 的，需要：(i) $\mathbf{h}_2$ 如何改 $\mathbf{h}_3$；外加上面那两步。
- 算喂入 $\mathbf{h}_1$ 的，再多一步 $\mathbf{h}_1$ 如何改 $\mathbf{h}_2$；其余与上面**完全重复**。

可见**越往后（输出端）的量越基础、越被复用**。于是从网络末端开始、倒着往前算导数，绝大多数项上一步都已算好——这就是**反向传播（backward pass）**。思路不难，难在推导要用矩阵微积分（偏置是向量、权重是矩阵）。下一节先用标量的玩具模型把机制讲透。

---

## 7.3 玩具示例（Toy example）

考虑一个只有 8 个标量参数 $\boldsymbol{\phi} = \{\beta_0,\omega_0,\dots,\beta_3,\omega_3\}$ 的模型，由 $\sin,\exp,\cos$ 复合而成：

$$
f[x,\boldsymbol{\phi}] = \beta_3 + \omega_3\cdot\cos\!\Big[\beta_2 + \omega_2\cdot\exp\!\big[\beta_1 + \omega_1\cdot\sin[\beta_0 + \omega_0\cdot x]\big]\Big],
$$

损失为最小二乘 $\ell_i = (f[x_i,\boldsymbol{\phi}] - y_i)^2$。可以把它看成一个“每层一个单元、激活函数分别是 $\sin/\exp/\cos$”的迷你神经网络。目标是算出 $\ell_i$ 对全部 8 个参数的导数。

当然可以手推。但有些表达式很恐怖，比如 $\partial\ell_i/\partial\omega_0$ 展开后有三层嵌套、还重复出现同一个 $\exp$ 项——手推易错，又**没利用到这种冗余**。反向传播能**一次性**高效算完所有导数：先**前向**算并存中间量，再**反向**从末端开始算导数、复用前面的结果。

**前向传播**：把损失的计算拆成一串有序步骤，并把每个中间量 $f_k,h_k$ 存下来：

$$
\begin{aligned}
f_0 &= \beta_0 + \omega_0\cdot x_i, & h_1 &= \sin[f_0], \\
f_1 &= \beta_1 + \omega_1\cdot h_1, & h_2 &= \exp[f_1], \\
f_2 &= \beta_2 + \omega_2\cdot h_2, & h_3 &= \cos[f_2], \\
f_3 &= \beta_3 + \omega_3\cdot h_3, & \ell_i &= (f_3 - y_i)^2 .
\end{aligned}
$$

**反向传播 #1（对中间量求导）**：从末端开始，**倒序**算 $\ell_i$ 对各中间量的导数。第一个最简单：

$$
\frac{\partial \ell_i}{\partial f_3} = 2(f_3 - y_i).
$$

下一个用**链式法则**：

$$
\frac{\partial \ell_i}{\partial h_3} = \frac{\partial f_3}{\partial h_3}\,\frac{\partial \ell_i}{\partial f_3}.
$$

左边问“$h_3$ 变化时 $\ell_i$ 怎么变”，右边把它拆成两段：(i) $h_3$ 怎么改 $f_3$，(ii) $f_3$ 怎么改 $\ell_i$。其中第二段**刚刚算过**，第一段就是 $\beta_3 + \omega_3 h_3$ 对 $h_3$ 求导 $= \omega_3$。照此倒推下去：

$$
\begin{aligned}
\frac{\partial \ell_i}{\partial f_2} &= \underbrace{\frac{\partial h_3}{\partial f_2}}_{\text{本步新增}}\;\underbrace{\frac{\partial f_3}{\partial h_3}\frac{\partial \ell_i}{\partial f_3}}_{\text{上一步已算}}, \\[4pt]
\frac{\partial \ell_i}{\partial h_2} &= \frac{\partial f_2}{\partial h_2}\Big[\frac{\partial h_3}{\partial f_2}\frac{\partial f_3}{\partial h_3}\frac{\partial \ell_i}{\partial f_3}\Big], \\[4pt]
&\;\;\vdots \\[4pt]
\frac{\partial \ell_i}{\partial f_0} &= \frac{\partial h_1}{\partial f_0}\Big[\frac{\partial f_1}{\partial h_1}\frac{\partial h_2}{\partial f_1}\frac{\partial f_2}{\partial h_2}\frac{\partial h_3}{\partial f_2}\frac{\partial f_3}{\partial h_3}\frac{\partial \ell_i}{\partial f_3}\Big].
\end{aligned}
$$

每一步，方括号里的东西上一步都算好了，只剩最前面那个新因子要算（形如 $\partial f_k/\partial h_k$ 或 $\partial h_k/\partial f_{k-1}$，都很简单）。这正是观察 2 的体现：**倒序计算就能复用之前的导数**。

**反向传播 #2（对参数求导）**：最后看损失怎么随参数 $\{\beta_k\},\{\omega_k\}$ 变，再用一次链式法则：

$$
\frac{\partial \ell_i}{\partial \beta_k} = \frac{\partial f_k}{\partial \beta_k}\frac{\partial \ell_i}{\partial f_k}, \qquad
\frac{\partial \ell_i}{\partial \omega_k} = \frac{\partial f_k}{\partial \omega_k}\frac{\partial \ell_i}{\partial f_k}.
$$

右边第二项 $\partial \ell_i/\partial f_k$ 在 #1 已经算出。而由 $f_k = \beta_k + \omega_k h_k$ 得：

$$
\frac{\partial f_k}{\partial \beta_k} = 1, \qquad \frac{\partial f_k}{\partial \omega_k} = h_k .
$$

这正好对应观察 1：**权重 $\omega_k$ 的导数与它所乘的源变量 $h_k$ 成正比**（$h_k$ 已在前向时存好）。最前端 $f_0 = \beta_0 + \omega_0 x_i$ 同理给出 $\partial f_0/\partial\beta_0 = 1$、$\partial f_0/\partial\omega_0 = x_i$。整套流程比逐个手推**既简单又高效**。

---

## 7.4 反向传播算法（Backpropagation algorithm）

现在把同样的思路搬到三层网络。直觉和大部分代数一模一样，区别只在于：中间量 $\mathbf{f}_k,\mathbf{h}_k$ 是向量、偏置 $\boldsymbol{\beta}_k$ 是向量、权重 $\boldsymbol{\Omega}_k$ 是矩阵，激活换成 ReLU。

**前向传播**：把网络写成一串有序计算，并存下所有中间量：

$$
\begin{aligned}
\mathbf{f}_0 &= \boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0\mathbf{x}_i, & \mathbf{h}_1 &= a[\mathbf{f}_0], \\
\mathbf{f}_1 &= \boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1\mathbf{h}_1, & \mathbf{h}_2 &= a[\mathbf{f}_1], \\
\mathbf{f}_2 &= \boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2\mathbf{h}_2, & \mathbf{h}_3 &= a[\mathbf{f}_2], \\
\mathbf{f}_3 &= \boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3\mathbf{h}_3, & \ell_i &= l[\mathbf{f}_3, y_i],
\end{aligned}
$$

其中 $\mathbf{f}_{k-1}$ 是第 $k$ 层**ReLU 之前**的“预激活值”，$\mathbf{h}_k$ 是 ReLU **之后**的激活值，$l[\cdot]$ 是损失函数。

**反向传播 #1（对预激活求导）**：看损失随 $\mathbf{f}_0,\mathbf{f}_1,\mathbf{f}_2$ 怎么变。以 $\mathbf{f}_2$ 为例，链式法则给出：

$$
\frac{\partial \ell_i}{\partial \mathbf{f}_2} = \frac{\partial \mathbf{h}_3}{\partial \mathbf{f}_2}\,\frac{\partial \mathbf{f}_3}{\partial \mathbf{h}_3}\,\frac{\partial \ell_i}{\partial \mathbf{f}_3}.
$$

更早的层同理，注意**方括号里的项都是上一步算好的，可直接复用**：

$$
\frac{\partial \ell_i}{\partial \mathbf{f}_1} = \frac{\partial \mathbf{h}_2}{\partial \mathbf{f}_1}\Big[\frac{\partial \mathbf{f}_2}{\partial \mathbf{h}_2}\frac{\partial \mathbf{h}_3}{\partial \mathbf{f}_2}\frac{\partial \mathbf{f}_3}{\partial \mathbf{h}_3}\frac{\partial \ell_i}{\partial \mathbf{f}_3}\Big],\qquad
\frac{\partial \ell_i}{\partial \mathbf{f}_0} = \frac{\partial \mathbf{h}_1}{\partial \mathbf{f}_0}\Big[\cdots\Big].
$$

而且每个新因子本身都很简单：

- $\partial \ell_i/\partial \mathbf{f}_3$ 取决于损失函数，但通常形式简单。
- 网络输出对隐藏层的导数：

$$
\frac{\partial \mathbf{f}_3}{\partial \mathbf{h}_3} = \frac{\partial}{\partial \mathbf{h}_3}(\boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3\mathbf{h}_3) = \boldsymbol{\Omega}_3^{\mathsf T}.
$$

- 激活输出对其输入的导数 $\partial \mathbf{h}_3/\partial \mathbf{f}_2$ 取决于激活函数，是个**对角矩阵**（每个激活只依赖对应的预激活）。对 ReLU，预激活小于 0 处为 0、否则为 1。实践中不去乘这个对角矩阵，而是把对角线抽成向量 $\mathbb{I}[\mathbf{f}_2 > 0]$ 做**逐元素相乘**，更高效。

> 文字配图（ReLU 的导数）：ReLU 在输入小于 0 时输出 0、否则输出输入本身（一条折线）；它的导数是一条阶跃——输入小于 0 处为 0（斜率为 0），大于 0 处为 1（斜率为 1）。

所以**沿网络往回走时，每步只是交替做两件事**：(i) 乘上权重矩阵的转置 $\boldsymbol{\Omega}_k^{\mathsf T}$；(ii) 按前向存好的输入 $\mathbf{f}_{k-1}$ 做 ReLU 阈值过滤。

**反向传播 #2（对权重、偏置求导）**：有了 $\partial \ell_i/\partial \mathbf{f}_k$，再用链式法则。对偏置：

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k} = \frac{\partial \mathbf{f}_k}{\partial \boldsymbol{\beta}_k}\frac{\partial \ell_i}{\partial \mathbf{f}_k} = \frac{\partial}{\partial \boldsymbol{\beta}_k}(\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k\mathbf{h}_k)\frac{\partial \ell_i}{\partial \mathbf{f}_k} = \frac{\partial \ell_i}{\partial \mathbf{f}_k}.
$$

对权重矩阵：

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k} = \frac{\partial \ell_i}{\partial \mathbf{f}_k}\,\mathbf{h}_k^{\mathsf T}.
$$

结果与同一大小的 $\boldsymbol{\Omega}_k$ 相符，且线性依赖 $\mathbf{h}_k$——再次印证观察 1：**权重的导数与它所乘的隐藏单元值 $\mathbf{h}_k$ 成正比**（$\mathbf{h}_k$ 已在前向时算好）。

### 7.4.1 算法总览

对一个有 $K$ 个 ReLU 隐藏层的网络，目标是算 $\partial \ell_i/\partial \boldsymbol{\beta}_k$ 与 $\partial \ell_i/\partial \boldsymbol{\Omega}_k$。

**前向传播**——计算并存储：

$$
\mathbf{f}_0 = \boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0\mathbf{x}_i,\qquad
\mathbf{h}_k = a[\mathbf{f}_{k-1}],\qquad
\mathbf{f}_k = \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k\mathbf{h}_k, \quad k\in\{1,\dots,K\}.
$$

**反向传播**——从末端 $\partial \ell_i/\partial \mathbf{f}_K$ 开始，对 $k = K, K-1,\dots,1$ 倒着算：

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k} = \frac{\partial \ell_i}{\partial \mathbf{f}_k},\qquad
\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k} = \frac{\partial \ell_i}{\partial \mathbf{f}_k}\mathbf{h}_k^{\mathsf T},
$$

$$
\frac{\partial \ell_i}{\partial \mathbf{f}_{k-1}} = \mathbb{I}[\mathbf{f}_{k-1} > 0]\odot\Big(\boldsymbol{\Omega}_k^{\mathsf T}\frac{\partial \ell_i}{\partial \mathbf{f}_k}\Big),
$$

其中 $\odot$ 是逐元素相乘，$\mathbb{I}[\mathbf{f}_{k-1} > 0]$ 是“预激活大于 0 处为 1、否则为 0”的指示向量。最后再补上第一层：

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_0} = \frac{\partial \ell_i}{\partial \mathbf{f}_0},\qquad
\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_0} = \frac{\partial \ell_i}{\partial \mathbf{f}_0}\mathbf{x}_i^{\mathsf T}.
$$

对 batch 里每个样本都这样算，再把梯度加起来，就得到 SGD 更新所需的梯度。

**为什么高效、代价是什么**：前向和反向中最费力的运算都只是**矩阵乘法**（乘 $\boldsymbol{\Omega}$ 或 $\boldsymbol{\Omega}^{\mathsf T}$，只用加法和乘法），且反向时大量项可复用——所以**计算上极其高效**。但它**不省内存**：前向的所有中间量都得留到反向用，这往往限制了能训练的模型规模。

### 7.4.2 算法微分（Algorithmic differentiation）

虽然理解反向传播很重要，但实践中你几乎不用自己写。PyTorch、TensorFlow 等框架会根据模型定义**自动求导**，这叫**算法微分**。每个功能模块（线性变换、ReLU、损失函数）都知道如何算自己的导数；框架又知道整个运算顺序，于是能自动跑前向和反向。它们还充分利用 GPU 的大规模并行——矩阵乘法天然适合并行，只要显存够，整个 batch 的前向/反向都能并行做。此时输入变成多维**张量**（tensor，矩阵向任意维度的推广：向量是 1D，矩阵是 2D，3D 张量就是数字的三维网格）。此前训练数据是 1D，所以反向传播的输入是 2D 张量（首维索引 batch 元素、次维索引数据维度）。再比如 RGB 图像样本是 3D（高×宽×通道），加上 batch 维就成了 4D 张量。

### 7.4.3 推广到任意计算图

上面描述的是天然**顺序**的网络。但模型可以有**分支结构**（如把一层的值分送两个子网再合并）。好消息是：只要计算图是**无环（acyclic）**的，反向传播的思路依然成立；现代框架能处理任意无环计算图。

---

## 7.5 参数初始化（Parameter initialization）

反向传播算出了 SGD/Adam 要用的梯度。但**开训之前**还得初始化参数，否则训练根本跑不稳。先看为什么。前向时每层预激活按下式计算：

$$
\mathbf{f}_k = \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k\mathbf{h}_k = \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k\,a[\mathbf{f}_{k-1}].
$$

设偏置全初始化为 0，权重 $\boldsymbol{\Omega}_k$ 的元素取自均值 0、方差 $\sigma^2$ 的正态分布。两种极端：

- **$\sigma^2$ 太小**（如 $10^{-5}$）：每个 $\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k\mathbf{h}_k$ 都是用极小权重做加权和，结果幅度比输入更小；再加上 ReLU 把负值砍掉、范围又减半——于是**预激活幅度逐层越来越小**。
- **$\sigma^2$ 太大**（如 $10^5$）：权重很大，加权和幅度比输入大得多；即便 ReLU 减半，幅度仍**逐层越来越大**。

两种情况下，预激活都可能小到/大到**浮点数无法表示**。反向时同理——每次梯度更新都要乘 $\boldsymbol{\Omega}^{\mathsf T}$，若初始化不当，梯度幅度会一路失控地**缩小**或**放大**。这就是**梯度消失（vanishing gradient）** 与**梯度爆炸（exploding gradient）** 问题：前者让更新小到可忽略，后者让训练发散不稳定。

### 7.5.1 前向传播的初始化（方差怎么传）

把上面的直觉做成数学。考虑相邻两层预激活 $\mathbf{f}\to\mathbf{f}'$（维度分别 $D_h, D_{h'}$）：

$$
\mathbf{h} = a[\mathbf{f}],\qquad \mathbf{f}' = \boldsymbol{\beta} + \boldsymbol{\Omega}\mathbf{h}.
$$

设输入层预激活 $f_j$ 的方差为 $\sigma_f^2$；偏置初始化为 0，权重 $\Omega_{ij}$ 取自均值 0、方差 $\sigma_\Omega^2$ 的正态分布。先看 $\mathbf{f}'$ 的**均值**：

$$
\mathbb{E}[f_i'] = \mathbb{E}\Big[\beta_i + \sum_{j=1}^{D_h}\Omega_{ij}h_j\Big] = \mathbb{E}[\beta_i] + \sum_{j=1}^{D_h}\mathbb{E}[\Omega_{ij}]\,\mathbb{E}[h_j] = 0,
$$

（用了权重与隐藏单元独立、且 $\mathbb{E}[\Omega_{ij}]=0$）。再看**方差**（因偏置初始化为 0，$\beta_i$ 项及含 $\beta_i$ 的交叉项均消失）：

$$
\sigma_{f'_i}^2 = \mathbb{E}[f_i'^2] - \mathbb{E}[f_i']^2 = \mathbb{E}\Big[\Big(\beta_i+\sum_{j=1}^{D_h}\Omega_{ij}h_j\Big)^2\Big] = \mathbb{E}\Big[\Big(\sum_{j=1}^{D_h}\Omega_{ij}h_j\Big)^2\Big] = \sum_{j=1}^{D_h}\mathbb{E}[\Omega_{ij}^2]\,\mathbb{E}[h_j^2] = \sigma_\Omega^2\sum_{j=1}^{D_h}\mathbb{E}[h_j^2].
$$

由于前一层预激活 $f_j$ 关于 0 对称，ReLU 会砍掉一半，所以二阶矩 $\mathbb{E}[h_j^2]$ 恰为 $f_j$ 方差的一半，即 $\sigma_f^2/2$。代入得：

$$
\sigma_{f'_i}^2 = \sigma_\Omega^2 \sum_{j=1}^{D_h}\frac{\sigma_f^2}{2} = \frac{1}{2}D_h\,\sigma_\Omega^2\,\sigma_f^2.
$$

要让下一层方差 $\sigma_{f'}^2$ 与本层方差 $\sigma_f^2$ **保持相等**（方差不放大也不缩小），就该设：

$$
\boxed{\;\sigma_\Omega^2 = \frac{2}{D_h}\;}
$$

其中 $D_h$ 是权重所作用的那一层的维度。这就是著名的 **He 初始化**。直觉上，那个分子里的 **2 正是用来补偿 ReLU 砍掉一半**带来的方差损失。

> 文字配图：取一个 50 层、每层 100 个单元的深网络，分别用 $\sigma_\Omega^2 \in \{0.001, 0.01, 0.02, 0.1, 1.0\}$ 初始化。画出“隐藏单元激活方差随层数”的曲线（对数坐标）：恰好取 He 值 $2/D_h = 0.02$ 时，方差**横着走、稳定不变**；偏大则一路飙升（爆炸），偏小则一路跳水（消失）。反向时梯度方差的曲线也呈同样趋势。

### 7.5.2 反向传播的初始化

对反向的梯度 $\partial l/\partial \mathbf{f}_k$ 做同样推导。反向时乘的是转置 $\boldsymbol{\Omega}^{\mathsf T}$，于是对应的稳定条件变成：

$$
\sigma_\Omega^2 = \frac{2}{D_{h'}},
$$

其中 $D_{h'}$ 是权重**喂入**的那一层的维度。

### 7.5.3 同时兼顾前向与反向

若权重矩阵不是方阵（相邻两层单元数 $D_h \ne D_{h'}$），就**无法同时满足**上面两式。一个折中是用两者均值 $(D_h + D_{h'})/2$ 当作“项数”，得到：

$$
\sigma_\Omega^2 = \frac{4}{D_h + D_{h'}}.
$$

实验表明，只要这样初始化，前向的激活方差和反向的梯度方差都能逐层保持稳定。

---

## 7.6 示例训练代码（Example training code）

本书重点是讲清原理，而非实现指南。但作者给了一段 PyTorch 代码示例，把目前为止的想法串起来。要点：

- **定义网络**：用 `nn.Sequential` 叠两个隐藏层（`nn.Linear` + `nn.ReLU`），输入/隐藏/输出维度为 `D_i, D_k, D_o`。
- **He 初始化**：写一个 `weights_init`，对每个 `nn.Linear` 调 `nn.init.kaiming_normal_` 初始化权重、并把偏置填 0，再用 `model.apply(weights_init)` 套到全网络。
- **损失与优化器**：最小二乘用 `nn.MSELoss`；优化器用带动量的 SGD（`lr=0.1, momentum=0.9`），并用 `StepLR` 让学习率**每 10 个 epoch 减半**。（原书正文叙述学习率从 0.01 起，但示例代码实为 `lr=0.1`，此处以代码为准。）
- **数据**：用 `TensorDataset` + `DataLoader` 把随机数据按 batch（大小 10）、打乱后喂入；共训练 100 个 epoch。
- **训练循环**：每个 batch 做四步——`optimizer.zero_grad()`（清空梯度）→ 前向 `pred = model(x_batch)` 算 `loss` → **`loss.backward()`** → `optimizer.step()`（更新参数）；epoch 末调 `scheduler.step()` 更新学习率。

> 核心 takeaway：尽管底层原理（尤其反向传播）相当复杂，**实现却很简单**——整套反向传播都藏在一行 `loss.backward()` 里。

---

## 本章小结

上一章引入了迭代优化算法 SGD（寻找损失最小值）；本章解决了它在神经网络上落地的两个专属问题。

- **梯度要算得快**：梯度需对海量参数、batch 中每个样本、每次迭代各算一遍，因此高效是刚需。**反向传播**通过“前向存中间量、反向传中间梯度、复用已算结果”做到了这一点——最费力的运算只是矩阵乘法，但代价是要存下前向的所有中间量（费内存）。现代框架用**算法微分**自动完成，一行 `loss.backward()` 即可。
- **参数要初始化得好**：前向时隐藏单元激活的幅度、反向时梯度的幅度，都可能随层数**指数级**衰减或膨胀，分别造成**梯度消失**和**梯度爆炸**，两者都会阻碍训练。按方差传播推导出的 **He 初始化**（$\sigma^2 = 2/D_h$，其中 2 用来补偿 ReLU）能让方差逐层稳定，从而避免这两个问题。

至此，模型、损失函数、训练算法都齐全了，可以为给定任务训练模型。下一章讨论如何**衡量模型性能**。

---

## 关键术语对照

| 中文 | English | 一句话含义 |
| --- | --- | --- |
| 反向传播 | Backpropagation | 高效计算损失对所有参数梯度的算法 |
| 前向传播 | Forward pass | 跑一遍网络，算出并存下各层中间量/激活 |
| 反向传播（过程） | Backward pass | 从输出端倒着推，复用已算结果求各梯度 |
| 链式法则 | Chain rule | 把复合函数的导数拆成各段导数相乘 |
| 算法微分 | Algorithmic differentiation | 框架按模型自动求导（反向模式） |
| 预激活 | Pre-activation | 激活函数作用之前的值 $\mathbf{f}_k$ |
| 梯度消失 | Vanishing gradient | 梯度逐层指数缩小，更新近乎为零 |
| 梯度爆炸 | Exploding gradient | 梯度逐层指数放大，训练发散 |
| He 初始化 | He initialization | 取 $\sigma^2 = 2/D_h$，让方差逐层稳定 |
| 张量 | Tensor | 矩阵向任意维度的推广 |
| 梯度检查点 | Gradient checkpointing | 只隔几层存激活，反向时重算以省内存 |

---

[返回目录](README.md) ｜ 上一章：[第 6 章 模型拟合](06-fitting-models.md) ｜ 下一章：[第 8 章 性能度量](08-measuring-performance.md)
