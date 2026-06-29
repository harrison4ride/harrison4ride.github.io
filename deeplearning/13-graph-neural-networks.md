# 第 13 章　图神经网络（Graph neural networks）

[返回目录](README.md)

> **本章一句话**：图神经网络是专门处理“图”（节点 + 边）的神经网络；它的核心套路是**消息传递**——每个节点不断从邻居那里收集信息、更新自己的表示，最终每个节点的向量都既懂自己、又懂它在整张图里的处境。

> **读这章的收获**：搞清楚什么是图、怎么用矩阵表示图，理解图上的三类任务（图级 / 节点级 / 边级），掌握图卷积网络（GCN）的“聚合邻居 + 更新自身”机制，弄懂为什么图模型必须满足**置换等变性**，以及**归纳式**与**直推式**学习的区别。

---

## 本章速览

- **图 = 节点 + 边**。很多现实事物天然是图：分子（原子 + 化学键）、社交网络（人 + 好友关系）、道路网、引文网、知识图谱、3D 点云……图通常很稀疏（大多数可能的边都不存在）。
- **三个新挑战**：① 拓扑结构可变（不同图节点数、连接方式都不同）；② 图可能极其巨大（社交网络可达十亿节点）；③ 常常只有**一整张图**，没有传统的“训练集 / 测试集”之分。
- **用三个矩阵表示图**：邻接矩阵 $\mathbf{A}$（谁连谁）、节点数据矩阵 $\mathbf{X}$（每个节点的特征）、边数据矩阵 $\mathbf{E}$（每条边的特征）。
- **三类任务**：图级（判断整张图的属性，如分子是否有毒）、节点级（给每个节点打标签，如点云的点属于机翼还是机身）、边级（预测两节点之间是否该有一条边，如好友推荐）。
- **核心机制是消息传递 / 邻居聚合**：每一层里，每个节点把邻居的向量汇总（求和、平均、取最大、或注意力加权），再与自己结合、做线性变换、过激活函数，得到新表示。多层叠加后，信息可以传播到更远的邻居。
- **置换等变性是硬约束**：节点的编号是任意的，重排编号不改变图，所以网络层必须对编号重排“无所谓”——这正是图卷积层的设计目标。
- **归纳式 vs 直推式**：有很多图、学一条“规则”再用到新图上，是归纳式；只有一张图、同时看已标注和未标注节点、直接给未知节点打标签，是直推式（半监督）。

---

## 13.1 什么是图（What is a graph）

**图（graph）** 是一种非常通用的结构：它由一组**节点（node，也叫顶点 vertex）** 构成，节点之间成对地用**边（edge，也叫连接 link）** 相连。图通常是**稀疏**的——所有可能的边里，只有一小部分真的存在。

很多现实事物天生就是图：

- **道路网**：节点是地点，边是连接它们的道路。
- **化学分子**：节点是原子，边是化学键。
- **电路**：节点是元件和接点，边是电连接。

还有很多数据虽然表面看不像图，其实也能用图表示：

- **社交网络**：节点是人，边是好友关系。
- **科学文献**：节点是论文，边是引用关系。
- **维基百科**：节点是词条，边是词条间的超链接。
- **程序代码**：节点是语法符号（变量在程序流中的不同位置），边是涉及这些变量的计算。
- **几何点云**：每个点是一个节点，与附近的点连边。
- **蛋白质相互作用**：节点是蛋白质，两者相互作用就连一条边。

此外，一个**集合**（无序列表）可以看成“每个成员都是节点、且彼此全部相连”的图；一张**图像**可以看成拓扑很规整的图——每个像素是节点，与相邻像素连边。

### 13.1.1 图的种类

图可以按多种方式分类：

- **无向图（undirected）**：边没有方向。例如社交网络里两人互相同意成为好友，关系是对称的。
- **有向图（directed）**：边有方向。例如引文网络里“甲论文引用乙论文”，是单向的。
- **异构多重图（heterogeneous multigraph）**：例如**知识图谱**，它通过定义对象之间的“关系”来编码一组事实。说它**异构**，是因为节点可以代表不同类型的实体（人、国家、公司）；说它**多重**，是因为任意两个节点之间可以有多条不同类型的边。

其他还有：**几何图（geometric graph）**——把点云按“连接 K 个最近邻”转成图，每个节点还带有 3D 空间坐标；**层次图（hierarchical graph）**——一个个小图（如桌子、灯、房间各自的结构图）本身又作为节点，组成一张更大的图。

> 所有种类的图都能用深度学习处理，但**本章聚焦最常见的无向图**（如社交网络）。

---

## 13.2 图的表示（Graph representation）

除了图的结构本身，节点上通常还附带信息。例如社交网络里每个人可以用一个定长向量描述其兴趣；有时边上也带信息，例如道路网里每条边可以记录长度、车道数、事故频率、限速。节点上的信息存为**节点嵌入（node embedding）**，边上的信息存为**边嵌入（edge embedding）**。

更正式地说，一张图有 $N$ 个节点、$E$ 条边，可以用三个矩阵编码：

- **邻接矩阵 $\mathbf{A}$**：编码图的**结构**。
- **节点数据矩阵 $\mathbf{X}$**：编码所有**节点嵌入**。
- **边数据矩阵 $\mathbf{E}$**：编码所有**边嵌入**。

**邻接矩阵**是一个 $N\times N$ 的矩阵：如果节点 $m$ 和节点 $n$ 之间有边，则第 $(m,n)$ 项为 1，否则为 0。对无向图，这个矩阵总是**对称**的。对大型稀疏图，可以只存一串连接 $(m,n)$ 来省内存。

第 $n$ 个节点的节点嵌入 $\mathbf{x}^{(n)}$ 长度为 $D$，把它们拼起来就得到 $D\times N$ 的节点数据矩阵 $\mathbf{X}$。类似地，第 $e$ 条边的边嵌入 $\mathbf{e}^{(e)}$ 长度为 $D_E$，拼成 $D_E\times E$ 的边数据矩阵 $\mathbf{E}$。为简单起见，前面几节先只考虑带节点嵌入的图，边嵌入留到 13.9 节再讲。

### 13.2.1 邻接矩阵的性质（Properties of the adjacency matrix）

邻接矩阵可以用线性代数直接“找邻居”。把第 $n$ 个节点的位置编码成一个**独热列向量**（只有第 $n$ 个位置是 1，其余全 0）。用邻接矩阵**左乘**这个向量，相当于取出 $\mathbf{A}$ 的第 $n$ 列，得到一个在“邻居位置”为 1 的向量——也就是“从第 $n$ 个节点走一步能到达的所有地方”。

> 直觉图示：把节点 6 表示成独热向量，乘一次 $\mathbf{A}$，结果告诉你“一步能走到节点 5、7、8”；再乘一次 $\mathbf{A}$（即乘 $\mathbf{A}^2$），结果告诉你“两步能走到哪些节点、各有几种走法”。

更一般地，把邻接矩阵取 $L$ 次幂，$\mathbf{A}^L$ 第 $(m,n)$ 项就是**从节点 $m$ 到节点 $n$、长度为 $L$ 的不同“游走（walk）”的条数**。（注意是“游走”不是“路径”——游走允许重复经过同一节点。）即便如此，$\mathbf{A}^L$ 仍蕴含大量连通性信息：第 $(m,n)$ 项非零，就说明从 $m$ 到 $n$ 的距离不超过 $L$。

### 13.2.2 节点编号的置换（Permutation）

图里**节点的编号是任意的**。重排节点编号，会使节点数据矩阵 $\mathbf{X}$ 的列发生重排，使邻接矩阵 $\mathbf{A}$ 的行和列都发生重排——但**底层的图完全没变**。这一点和图像、文本截然不同：打乱像素就是另一张图，打乱词语就是另一句话，但打乱节点编号还是同一张图。

交换节点编号的操作可以用**置换矩阵 $\mathbf{P}$** 表示（每行每列恰好有一个 1，其余为 0）。从一种编号映射到另一种编号：

$$
\mathbf{X}' = \mathbf{X}\mathbf{P}, \qquad \mathbf{A}' = \mathbf{P}^T\mathbf{A}\mathbf{P}.
$$

右乘 $\mathbf{P}$ 重排列，左乘 $\mathbf{P}^T$ 重排行。由此引出一条关键要求：**任何作用在图上的处理都必须对这种重排无动于衷**，否则结果就会依赖于我们随手选的节点编号——这显然不合理。

---

## 13.3 图神经网络、任务与损失函数（GNNs, tasks, and loss functions）

**图神经网络（GNN）** 是这样一个模型：它把节点嵌入 $\mathbf{X}$ 和邻接矩阵 $\mathbf{A}$ 作为输入，让它们穿过一系列 $K$ 层。节点嵌入在每一层都会被更新，产生中间的“隐藏”表示 $\mathbf{H}_k$，最终得到输出嵌入 $\mathbf{H}_K$。

> 直觉：网络开头，每个节点的向量只装着“它自己”的信息；网络末尾，每个节点的向量装进了“它自己 + 它在图中所处的上下文”。这和词向量穿过 Transformer 很像——开头只表示单词，结尾表示这个词在整句话中的含义。

### 13.3.1 任务与损失函数

有监督的图问题通常分三类：

**① 图级任务（graph-level）**：给整张图打一个标签或估一个值，同时利用结构和节点嵌入。例如预测分子的熔点（回归）、判断分子是否对人有毒（分类）。做法是把输出的节点嵌入**汇总**（如取平均），再用线性变换或神经网络映射到一个定长向量。回归用最小二乘损失；二分类则过 sigmoid，再用二元交叉熵损失（多类分类则把定长向量过 softmax 得到各类别概率）。图属于第 1 类的概率可写成：

$$
\Pr(y=1\mid \mathbf{X},\mathbf{A}) = \mathrm{sig}\!\left[\beta_K + \boldsymbol{\omega}_K \mathbf{H}_K \mathbf{1}/N\right],
$$

其中标量 $\beta_K$ 和 $1\times D$ 向量 $\boldsymbol{\omega}_K$ 是可学习参数。把输出嵌入矩阵 $\mathbf{H}_K$ 右乘全 1 列向量 $\mathbf{1}$，等于把所有节点的嵌入加起来，再除以节点数 $N$ 就是求平均——这叫**均值池化（mean pooling）**。

**② 节点级任务（node-level）**：给图的**每个节点**打标签（分类）或估值（回归）。例如把 3D 点云转成图后，判断每个点属于机翼还是机身。损失函数定义方式与图级相同，只是现在对每个节点 $n$ **独立**计算：

$$
\Pr(y^{(n)}=1\mid \mathbf{X},\mathbf{A}) = \mathrm{sig}\!\left[\beta_K + \boldsymbol{\omega}_K \mathbf{h}_K^{(n)}\right].
$$

**③ 边预测任务（edge prediction）**：预测节点 $n$ 和 $m$ 之间是否**应该**有一条边。例如社交网络里预测两人是否认识且投缘，进而推荐他们加好友。这是个二分类任务，需要把两个节点嵌入合成一个数（表示边存在的概率）。一种做法是取两节点嵌入的**点积**再过 sigmoid：

$$
\Pr(y^{(mn)}=1\mid \mathbf{X},\mathbf{A}) = \mathrm{sig}\!\left[\mathbf{h}^{(m)T}\mathbf{h}^{(n)}\right].
$$

---

## 13.4 图卷积网络（Graph convolutional networks, GCN）

图神经网络有很多种，本章聚焦**基于空间的卷积图神经网络**，简称 **GCN**。

- 说它**卷积**，是因为它通过**聚合邻近节点的信息**来更新每个节点——这引入了一种**关系归纳偏置（relational inductive bias）**，即优先采纳来自邻居的信息。
- 说它**基于空间**，是因为它直接使用原始的图结构（与之相对的是“基于谱”的方法，那类方法在傅里叶域里做卷积）。

GCN 的每一层都是一个带参数 $\boldsymbol{\phi}$ 的函数 $\mathbf{F}[\cdot]$，输入节点嵌入和邻接矩阵，输出新的节点嵌入。整个网络可写成：

$$
\begin{aligned}
\mathbf{H}_1 &= \mathbf{F}[\mathbf{X}, \mathbf{A}, \boldsymbol{\phi}_0]\\
\mathbf{H}_2 &= \mathbf{F}[\mathbf{H}_1, \mathbf{A}, \boldsymbol{\phi}_1]\\
&\ \vdots\\
\mathbf{H}_K &= \mathbf{F}[\mathbf{H}_{K-1}, \mathbf{A}, \boldsymbol{\phi}_{K-1}],
\end{aligned}
$$

其中 $\mathbf{X}$ 是输入，$\mathbf{A}$ 是邻接矩阵，$\mathbf{H}_k$ 是第 $k$ 层修改后的节点嵌入，$\boldsymbol{\phi}_k$ 是从第 $k$ 层映射到第 $k{+}1$ 层的参数。

### 13.4.1 等变性与不变性（Equivariance and invariance）

前面说过节点编号是任意的，重排编号不改变图，模型必须尊重这一点。于是每一层都必须对节点编号的重排是**等变（equivariant）** 的：如果重排节点编号，那么每一阶段的节点嵌入也会以**同样的方式**被重排。用置换矩阵 $\mathbf{P}$ 表示就是：

$$
\mathbf{H}_{k+1}\mathbf{P} = \mathbf{F}[\mathbf{H}_k\mathbf{P}, \mathbf{P}^T\mathbf{A}\mathbf{P}, \boldsymbol{\phi}_k].
$$

对节点分类和边预测任务，输出也应当对节点编号重排**等变**。但对**图级任务**，最后一层会把整张图的信息汇总成一个结果，所以输出对节点顺序是**不变（invariant）** 的。前面图级任务的输出层正好满足这一点：

$$
y = \mathrm{sig}\!\left[\beta_K + \boldsymbol{\omega}_K \mathbf{H}_K \mathbf{1}/N\right] = \mathrm{sig}\!\left[\beta_K + \boldsymbol{\omega}_K \mathbf{H}_K \mathbf{P}\mathbf{1}/N\right],
$$

对任意置换矩阵 $\mathbf{P}$ 都成立（因为求和/求平均与顺序无关）。

> 类比图像：分割应当对几何变换**等变**，分类应当**不变**。卷积和池化层只能对“平移”近似做到这一点，且无法对更一般的变换精确保证。但对图来说，我们可以设计出对“节点置换”**精确**等变或不变的网络。

### 13.4.2 参数共享（Parameter sharing）

第 10 章讲过，不该用全连接网络处理图像，否则网络得在每个像素位置分别学会识别同一物体。卷积层的做法是**对每个位置一视同仁地处理**，既减少参数，又引入“图像各处同等对待”的归纳偏置。

对图里的节点也可以这么论证。如果给每个节点配一套独立参数，网络就得在每个位置独立学习连接的含义，训练还得有大量拓扑相同的图——不现实。因此我们让模型**在每个节点上使用同一套参数**：既减少参数，又把“在某个节点学到的东西”共享到整张图。

回忆卷积的本质：通过对邻居信息做**加权求和**来更新一个变量。可以把它想成——每个邻居给目标变量**发一条消息**，目标变量把这些消息聚合起来完成更新。对图像而言，邻居是固定大小方块内的像素，每个位置的空间关系都一样；但在图里，每个节点的邻居数目可能不同，也没有“上方邻居”“下方邻居”这种一致的空间关系，所以无法像图像那样按方位区别加权。

### 13.4.3 GCN 层示例

上述考虑导出一个简单的 GCN 层。在第 $k$ 层的每个节点 $n$，先把邻居的节点嵌入**求和**，得到聚合向量：

$$
\mathrm{agg}[n,k] = \sum_{m\in \mathrm{ne}[n]} \mathbf{h}_k^{(m)},
$$

其中 $\mathrm{ne}[n]$ 表示节点 $n$ 的邻居编号集合。然后对当前节点嵌入 $\mathbf{h}_k^{(n)}$ 和这个聚合向量各做一次线性变换 $\boldsymbol{\Omega}_k$，加上偏置 $\boldsymbol{\beta}_k$，再过非线性激活 $\mathbf{a}[\cdot]$（逐元素作用，如 ReLU）：

$$
\mathbf{h}_{k+1}^{(n)} = \mathbf{a}\!\left[\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k\,\mathbf{h}_k^{(n)} + \boldsymbol{\Omega}_k\,\mathrm{agg}[n,k]\right].
$$

> 一层的五个步骤：(i) 聚合邻居成一个向量；(ii) 对聚合向量做线性变换 $\boldsymbol{\Omega}$；(iii) 对当前节点做**同样的**线性变换 $\boldsymbol{\Omega}$；(iv) 两者相加，再加偏置 $\boldsymbol{\beta}$；(v) 过激活函数。后续层重复这一过程，但每层用不同参数。

这个更新可以写得更紧凑。注意“矩阵右乘一个向量 = 取它各列的加权和”，而邻接矩阵 $\mathbf{A}$ 的第 $n$ 列恰好在邻居位置为 1。所以把节点嵌入收进 $D\times N$ 矩阵 $\mathbf{H}_k$、右乘 $\mathbf{A}$，结果的第 $n$ 列正是 $\mathrm{agg}[n,k]$。整层的更新于是变为：

$$
\mathbf{H}_{k+1} = \mathbf{a}\!\left[\boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}_k\mathbf{H}_k + \boldsymbol{\Omega}_k\mathbf{H}_k\mathbf{A}\right]
= \mathbf{a}\!\left[\boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}_k\mathbf{H}_k(\mathbf{A}+\mathbf{I})\right],
$$

其中 $\mathbf{1}$ 是 $N\times 1$ 全 1 向量，$\mathbf{A}+\mathbf{I}$ 里的 $\mathbf{I}$（单位矩阵）正好把“当前节点自己”也算进聚合（自环）。

这一层满足全部设计要求：对节点编号置换**等变**、能应付任意数目的邻居、利用图结构提供**关系归纳偏置**、并在整张图上**共享参数**。

---

## 13.5 图分类示例（Example: graph classification）

把上面的想法组合起来，造一个判断分子“有毒 / 无害”的网络。输入是邻接矩阵 $\mathbf{A}$ 和节点嵌入矩阵 $\mathbf{X}$：

- $\mathbf{A}\in\mathbb{R}^{N\times N}$ 来自分子结构。
- $\mathbf{X}\in\mathbb{R}^{118\times N}$ 的每一列是**独热向量**，指示该原子是周期表 118 种元素中的哪一种（长度 118，只有对应元素那一位是 1）。第一个权重矩阵 $\boldsymbol{\Omega}_0\in\mathbb{R}^{D\times 118}$ 把节点嵌入变换到任意维度 $D$。

网络方程为：

$$
\begin{aligned}
\mathbf{H}_1 &= \mathbf{a}\!\left[\boldsymbol{\beta}_0\mathbf{1}^T + \boldsymbol{\Omega}_0\mathbf{X}(\mathbf{A}+\mathbf{I})\right]\\
\mathbf{H}_2 &= \mathbf{a}\!\left[\boldsymbol{\beta}_1\mathbf{1}^T + \boldsymbol{\Omega}_1\mathbf{H}_1(\mathbf{A}+\mathbf{I})\right]\\
&\ \vdots\\
\mathbf{H}_K &= \mathbf{a}\!\left[\boldsymbol{\beta}_{K-1}\mathbf{1}^T + \boldsymbol{\Omega}_{K-1}\mathbf{H}_{K-1}(\mathbf{A}+\mathbf{I})\right]\\
\mathrm{f}[\mathbf{X},\mathbf{A},\boldsymbol{\Phi}] &= \mathrm{sig}\!\left[\beta_K + \boldsymbol{\omega}_K\mathbf{H}_K\mathbf{1}/N\right],
\end{aligned}
$$

网络输出 $\mathrm{f}[\mathbf{X},\mathbf{A},\boldsymbol{\Phi}]$ 是一个数，表示分子有毒的概率。

### 13.5.1 用批次训练

给定 $I$ 个训练图 $\{\mathbf{X}_i,\mathbf{A}_i\}$ 及其标签 $y_i$，参数 $\boldsymbol{\Phi}=\{\boldsymbol{\beta}_k,\boldsymbol{\Omega}_k\}_{k=0}^{K}$ 可用 SGD 加二元交叉熵损失来学。全连接网络、卷积网络、Transformer 都靠并行硬件**一次处理一整批**样本，做法是把批内元素拼成更高维的张量。

可问题来了：每张图的节点数可能不同，因此 $\mathbf{X}_i$ 和 $\mathbf{A}_i$ 尺寸各异，没法直接拼成 3D 张量。一个巧妙的小技巧解决了它：**把这一批图当成一张大图的若干互不相连的连通分量**，于是整批可以作为网络方程的单次实例并行跑完；只需让均值池化**只在各自的小图内部**进行，从而每张图得到一个表示，送入损失函数。

---

## 13.6 归纳式 vs 直推式（Inductive vs. transductive models）

本书此前所有模型都是**归纳式（inductive）** 的：用带标签的训练集学“输入→输出”的关系，再把它用到新的测试数据上。可以理解为：**学到一条规则，再拿去别处套用**。

而**直推式（transductive）** 模型同时考虑**已标注和未标注的数据**：它不产出一条规则，只是**直接给未知输出打上标签**。这有时被称为**半监督学习（semi-supervised learning）**。

- **优点**：能利用未标注数据里的模式来辅助判断。
- **缺点**：一旦加入新的未标注数据，模型就得**重新训练**。

两种问题在图上都很常见：

- 有时我们有**很多带标签的图**，学“图→标签”的映射。例如很多分子各自标了是否对人有毒，学出规则后用到新分子上——这是**归纳式**。
- 有时只有**一整张大图**。例如科学论文引文图，部分节点标了领域（物理、生物……），希望给其余节点也打上领域标签。此时训练数据和测试数据**不可分割地纠缠在一起**——这是**直推式**。

> 关键区分：**图级任务只出现在归纳式场景**（必须有训练图和测试图）。而**节点级任务和边预测任务在两种场景下都可能出现**。在直推式场景里，损失函数只在“已知真值”的地方最小化预测与真值的差距；新预测则通过前向传播、在“真值未知”的地方读取结果得到。

---

## 13.7 节点分类示例（Example: node classification）

作为第二个例子，考虑**直推式**场景下的二分类节点任务。我们有一张商业规模的大图，可能有上百万节点；部分节点有真值二元标签，目标是给其余未标注节点打标签。网络主体与上例相同，只是换一个最终层，输出一个 $1\times N$ 的向量：

$$
\mathrm{f}[\mathbf{X},\mathbf{A},\boldsymbol{\Phi}] = \mathrm{sig}\!\left[\beta_K\mathbf{1}^T + \boldsymbol{\omega}_K\mathbf{H}_K\right],
$$

其中 $\mathrm{sig}[\cdot]$ 对行向量每个元素独立作用。仍用二元交叉熵损失，但**只在已知真值标签 $y$ 的节点上**计算。这其实就是前面节点分类损失的向量化版本。

训练这种网络有两个麻烦：

1. **太大，存不下**：前向传播要存下每一层的节点嵌入，相当于反复存储和处理“几倍于整张图”的结构，可能不现实。
2. **只有一张图，怎么做 SGD？** 只有一个对象，怎么凑一个批次？

### 13.7.1 如何构造批次

一种办法：每个训练步随机选一**子集已标注节点**。每个节点依赖上一层的邻居，邻居又依赖更上一层的邻居……所以（和卷积网络一样）每个节点有一个**感受野（receptive field）**，这块区域叫 **k 跳邻域（k-hop neighborhood）**。于是我们可以只用“这批节点各自 k 跳邻域之并”所构成的子图来做一次梯度下降，其余输入不参与。

> 直觉图示：看第二隐藏层里的某个节点，它的输入来自第一隐藏层中它的 1 跳邻域；这些节点又各自从它们的邻居取输入；于是第二层这个节点最终从输入层的 2 跳邻域获取信息——这正是卷积网络“感受野”概念在图上的对应。

但如果层数很多、图又稠密，可能**每个输入节点都落进每个输出的感受野**，子图根本没缩小——这叫 **图扩张问题（graph expansion problem）**。两种应对方法：

- **邻域采样（neighborhood sampling）**：从输出层的批节点往回走，每往前一层只**随机采样固定数目的邻居**，再前一层又采样它们固定数目的邻居……图仍会逐层变大，但增长被控制住了。每个批次都重新采样，所以即使抽到同一批节点，参与的邻居也会不同——这有点像 dropout，自带一点**正则化**效果。
- **图划分（graph partitioning）**：先把原图**聚类**成互不相连的若干子集（彼此不连的小图）。可用标准算法选取这些子集，使内部连接尽量多。每个小图可直接当一个批次，也可以随机组合几个小图成一个批次（并重新接回它们之间原本的边）。

有了构造批次的办法，就能像归纳式那样训练参数，把已标注节点分成训练 / 验证 / 测试集——相当于**把一个直推式问题转化成了归纳式问题**。推理时只需根据未知节点的 k 跳邻域计算预测，且无需存中间表示，所以内存友好得多。

---

## 13.8 图卷积层的变体（Layers for GCNs）

前面的例子里，我们把邻居消息**求和**、再与变换后的当前节点相加，靠的是右乘 $\mathbf{A}+\mathbf{I}$。现在考虑两件事的不同做法：(i) 如何把当前嵌入与聚合后的邻居**结合**；(ii) **聚合**本身怎么做。

### 13.8.1 结合“当前节点”与“聚合邻居”

基础版直接相加：

$$
\mathbf{H}_{k+1} = \mathbf{a}\!\left[\boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}_k\mathbf{H}_k(\mathbf{A}+\mathbf{I})\right].
$$

**对角增强（diagonal enhancement）**：让当前节点先乘一个系数 $(1+\epsilon_k)$ 再参与求和，$\epsilon_k$ 是每层不同的可学习标量——相当于给“自己”更大的权重：

$$
\mathbf{H}_{k+1} = \mathbf{a}\!\left[\boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}_k\mathbf{H}_k(\mathbf{A}+(1+\epsilon_k)\mathbf{I})\right].
$$

**对当前节点用不同的线性变换 $\boldsymbol{\Psi}_k$**：把“聚合邻居”和“当前节点”用两套权重处理后再合并，可整理成把 $\boldsymbol{\Omega}_k$ 与 $\boldsymbol{\Psi}_k$ 拼成一个更大的权重 $\boldsymbol{\Omega}'_k$：

$$
\mathbf{H}_{k+1} = \mathbf{a}\!\left[\boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}_k\mathbf{H}_k\mathbf{A} + \boldsymbol{\Psi}_k\mathbf{H}_k\right]
= \mathbf{a}\!\left[\boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}'_k\begin{bmatrix}\mathbf{H}_k\mathbf{A}\\ \mathbf{H}_k\end{bmatrix}\right].
$$

### 13.8.2 残差连接（Residual connections）

把邻居聚合后的表示先变换、过激活，再与当前节点**拼接**（concatenate）或相加。拼接版的更新为：

$$
\mathbf{H}_{k+1} = \begin{bmatrix}\mathbf{a}\!\left[\boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}_k\mathbf{H}_k\mathbf{A}\right]\\ \mathbf{H}_k\end{bmatrix}.
$$

### 13.8.3 均值聚合（Mean aggregation）

前面是把邻居嵌入**求和**，但也可以**取平均**。当“嵌入内容”比“结构信息”更重要时，平均往往更好，因为这样邻居贡献的幅度就不会随邻居数目变化：

$$
\mathrm{agg}[n] = \frac{1}{|\mathrm{ne}[n]|}\sum_{m\in\mathrm{ne}[n]}\mathbf{h}_m.
$$

引入对角的 $N\times N$ **度矩阵 $\mathbf{D}$**（对角线上是每个节点的邻居数），其逆 $\mathbf{D}^{-1}$ 的对角元正是求平均所需的分母。于是矩阵形式为：

$$
\mathbf{H}_{k+1} = \mathbf{a}\!\left[\boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}_k\mathbf{H}_k(\mathbf{A}\mathbf{D}^{-1}+\mathbf{I})\right].
$$

### 13.8.4 Kipf 归一化（Kipf normalization）

在 Kipf 归一化里，邻居表示之和按**当前节点与邻居双方的度数**做归一化：

$$
\mathrm{agg}[n] = \sum_{m\in\mathrm{ne}[n]}\frac{\mathbf{h}_m}{\sqrt{|\mathrm{ne}[n]|\,|\mathrm{ne}[m]|}}.
$$

逻辑是：来自“邻居极多”的节点的信息应被**降权**，因为它连接太多、单条连接提供的独特信息较少。矩阵形式为：

$$
\mathbf{H}_{k+1} = \mathbf{a}\!\left[\boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}_k\mathbf{H}_k(\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}+\mathbf{I})\right].
$$

### 13.8.5 最大池化聚合（Max pooling aggregation）

另一个同样对置换不变的操作是**取最大**。最大池化聚合算子：

$$
\mathrm{agg}[n] = \max_{m\in\mathrm{ne}[n]}\big[\mathbf{h}_m\big],
$$

即对邻居向量逐元素取最大值。

### 13.8.6 注意力聚合（Aggregation by attention）

前面的聚合要么对邻居等权，要么按图的拓扑加权。而在**图注意力层（graph attention layer）** 里，权重取决于**节点上的数据本身**。先对当前节点嵌入做线性变换：

$$
\mathbf{H}'_k = \boldsymbol{\beta}_k\mathbf{1}^T + \boldsymbol{\Omega}_k\mathbf{H}_k.
$$

再算每对变换后嵌入 $\mathbf{h}'_m$ 与 $\mathbf{h}'_n$ 的相似度：把这对向量拼接、与可学习参数列向量 $\boldsymbol{\phi}_k$ 取点积、过激活：

$$
s_{mn} = \mathbf{a}\!\left[\boldsymbol{\phi}_k^T\begin{bmatrix}\mathbf{h}'_m\\ \mathbf{h}'_n\end{bmatrix}\right].
$$

这些值存进 $N\times N$ 矩阵 $\mathbf{S}$。和点积自注意力一样，用 softmax 把权重归一化为非负且和为 1；但**只有当前节点及其邻居**才该参与。最后把注意力权重作用到变换后的嵌入上：

$$
\mathbf{H}_{k+1} = \mathbf{a}\!\left[\mathbf{H}'_k\cdot\mathrm{Softmask}[\mathbf{S}, \mathbf{A}+\mathbf{I}]\right].
$$

$\mathrm{Softmask}[\cdot,\cdot]$ 先把第二个参数 $\mathbf{A}+\mathbf{I}$ 为 0 的位置在 $\mathbf{S}$ 里设成负无穷（使它们不贡献），再对每一列做 softmax——这就保证了对**非邻居**节点的注意力为零。

> 这与 Transformer 的点积自注意力非常像，区别在于：(i) 键、查询、值都是同一个；(ii) 相似度度量不同；(iii) 注意力被**掩码**，每个节点只关注自己和邻居。和 Transformer 一样，它也能扩展成**多头**并行再重组。

---

## 13.9 边图（Edge graphs）

前面一直在处理**节点嵌入**——它们穿过网络逐步演化，到末尾就既表示节点、又表示节点在图中的上下文。现在考虑信息附在**边**上的情形。

借助**边图（edge graph，又叫伴随图 adjoint graph 或线图 line graph）**，可以把处理节点嵌入的那套机器直接搬来处理边嵌入。边图是这样构造的互补图：

- **原图里的每一条边，变成边图里的一个节点**；
- 原图里**共享一个公共节点的两条边**，在边图里就连成一条边。

一般来说，原图可以从它的边图还原回来，所以两种表示之间可以来回切换。

> 直觉图示：原图有 6 个节点若干条边；给每条原边画一个新节点（圆圈），凡是“在原图里挨着同一个节点”的两条边，就把对应的新节点连起来——这就得到边图。

要处理边嵌入，就把图转成它的边图，然后用**完全相同**的技术：在每个新节点处聚合邻居信息、再与当前表示结合。当节点和边嵌入**同时存在**时，可以在两张图之间来回翻译，于是有四种可能的更新：

1. 节点更新节点；
2. 节点更新边；
3. 边更新节点；
4. 边更新边。

这四种更新可以按需交替进行；稍加改造，还能让节点**同时**从“节点”和“边”两方面被更新。

---

## 本章小结

- **图 = 节点 + 边**，节点和边都可以附带数据，分别叫**节点嵌入**和**边嵌入**。很多现实问题都能表述成图任务：判断**整张图**的属性、判断**每个节点 / 边**的属性、或预测图中**是否该多一条边**。
- 图用三个矩阵编码：**邻接矩阵 $\mathbf{A}$**（结构）、**节点数据矩阵 $\mathbf{X}$**、**边数据矩阵 $\mathbf{E}$**。邻接矩阵的幂 $\mathbf{A}^L$ 还能告诉你节点间长度为 $L$ 的游走条数。
- 因为节点编号是任意的，GNN 的每一层都必须对节点置换**等变**；图级任务的输出层则做到了**不变**。
- **基于空间的卷积图网络（GCN）** 是一类核心 GNN：在每一层，每个节点**聚合邻居信息**（求和 / 平均 / 取最大 / Kipf 归一化 / 注意力加权），再与自身结合、做线性变换、过激活函数。整层可紧凑写成 $\mathbf{H}_{k+1}=\mathbf{a}[\boldsymbol{\beta}_k\mathbf{1}^T+\boldsymbol{\Omega}_k\mathbf{H}_k(\mathbf{A}+\mathbf{I})]$。这正是**消息传递**的具体实现：堆叠多层，信息便能从更远的邻居传播过来。
- 图处理的一大特点是常出现在**直推式**场景：只有一张部分标注的大图，而非成套的训练 / 测试图。这张图可能极大，由此催生了**邻域采样**和**图划分**等技术。
- 把图转成**边图**（每条边变成一个节点），同一套 GNN 机器就能用来更新边嵌入。

---

## 关键术语对照

| 中文 | English | 一句话含义 |
| --- | --- | --- |
| 图 | Graph | 由节点和边组成的通用结构 |
| 节点 / 边 | Node / Edge | 图的基本单元 / 节点间的连接 |
| 邻接矩阵 | Adjacency matrix | $N\times N$ 矩阵，记录谁连谁 |
| 节点嵌入 / 边嵌入 | Node / Edge embedding | 附在节点 / 边上的特征向量 |
| 图神经网络 | Graph neural network (GNN) | 处理图数据的深度模型 |
| 图卷积网络 | Graph convolutional network (GCN) | 靠聚合邻居更新节点的 GNN |
| 消息传递 / 邻居聚合 | Message passing / Aggregation | 节点从邻居收集并汇总信息 |
| 关系归纳偏置 | Relational inductive bias | 优先采纳来自邻居的信息 |
| 置换等变 / 不变 | Permutation equivariance / invariance | 重排节点编号后输出同步重排 / 完全不变 |
| 置换矩阵 | Permutation matrix | 表示节点重新编号的 0/1 矩阵 |
| 度矩阵 | Degree matrix | 对角线为各节点邻居数的矩阵 |
| 图级 / 节点级 / 边级任务 | Graph- / Node- / Edge-level task | 预测整图 / 每节点 / 是否有边 |
| 均值池化 | Mean pooling | 把所有节点嵌入取平均 |
| 归纳式 / 直推式 | Inductive / Transductive | 学规则套新图 / 直接给未知节点打标签 |
| k 跳邻域 / 感受野 | k-hop neighborhood / Receptive field | 影响某节点的图区域 |
| 邻域采样 / 图划分 | Neighborhood sampling / Graph partitioning | 控制大图训练规模的两种方法 |
| 图注意力网络 | Graph attention network | 用数据本身决定邻居权重 |
| 边图（线图） | Edge graph (line graph) | 每条边变成节点的互补图 |

---

[返回目录](README.md) ｜ 上一章：[第 12 章 Transformer](12-transformers.md) ｜ 下一章：[第 14 章 无监督学习](14-unsupervised-learning.md)
