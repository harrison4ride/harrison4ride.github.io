# 06. 训练与系统

[返回目录](README.md)

## 1. 常见并行策略

训练时，GPU 内存主要用于保存：

- **Parameters**：模型参数；
- **Gradients**：每个参数的梯度；
- **Optimizer States**：例如 AdamW 的一阶和二阶动量；
- **Activations**：前向传播产生、反向传播需要的中间结果；
- **Temporary Buffers**：Attention、通信和算子使用的临时空间。

不同并行方法切分的是不同内容：

| 并行方法                       | 主要切分对象                 | 主要解决的问题       |
| ------------------------------ | ---------------------------- | -------------------- |
| Data Parallelism               | 训练数据                     | 提高整体吞吐         |
| ZeRO / FSDP                    | 参数、梯度、Optimizer States | 减少单卡模型状态     |
| Tensor Parallelism             | 单层中的矩阵计算             | 单层模型放不下       |
| Pipeline Parallelism           | 不同网络层                   | 整个模型放不下       |
| Sequence / Context Parallelism | 序列和激活值                 | 长上下文激活占用过大 |
| Expert Parallelism             | MoE Experts                  | MoE Expert 参数过大  |

实际训练通常不会只使用一种，而是根据内存、通信和集群拓扑组合使用。

### Data Parallelism

Data Parallelism 的基本做法是：

1. 每张 GPU 保存一份完整模型；
2. 不同 GPU 处理不同数据；
3. 每张 GPU 独立完成前向和反向传播；
4. 通过 All-Reduce 同步梯度；
5. 所有 GPU 使用相同梯度更新参数。

### DP 和 DDP 的区别

在 PyTorch 中：

- `DataParallel` 通常由一个进程控制多张 GPU，主卡容易成为瓶颈；
- `DistributedDataParallel` 通常每张 GPU 使用一个独立进程，通过 All-Reduce 同步梯度。

实际多 GPU 训练一般使用 DDP，而不是旧式的 `DataParallel`。

需要注意：

> Data Parallelism 并不会减少单张 GPU 保存的模型状态，只是把数据分给不同 GPU。

如果模型本身无法装入单卡，需要使用 FSDP、ZeRO、TP 或 PP。

### ZeRO 与 FSDP

普通 Data Parallelism 在每张 GPU 上都保存完整的：

- Parameters；
- Gradients；
- Optimizer States。

ZeRO 的核心思想是：

> 既然不同 Data-Parallel Ranks 最终维护的是同一个模型，就不需要让每张 GPU 永久保存所有模型状态。

### Tensor Parallelism（TP）

把单层矩阵乘法沿维度切到多卡。例如列并行和行并行线性层。适合模型单层过大时，但层内频繁 collective，对高速互联要求高。

Tensor Parallelism 的优点：

- 可以切分单个超大线性层；
- 降低每张 GPU 保存的参数和部分激活值；
- 适合单层参数已经无法放入一张 GPU 的模型。

主要代价：

- 几乎每一层都需要 Collective Communication；
- 对 GPU 间带宽和延迟要求很高；
- TP Degree 越大，单张 GPU 的矩阵越小，计算效率可能下降。


### Pipeline Parallelism（PP）

Pipeline Parallelism 按网络深度切分模型。

例如，一个 48 层模型可以分成 4 个 Stage：

- Stage 1：第 1–12 层；
- Stage 2：第 13–24 层；
- Stage 3：第 25–36 层；
- Stage 4：第 37–48 层。

一个完整 Batch 会进一步切成多个 Micro-batches，在不同 Stage 之间流水执行。

Pipeline Parallelism 的优点：

- 每张 GPU 只保存部分网络层；
- 通信内容主要是 Stage 之间的激活值和梯度；
- 可以将模型扩展到多个节点。

主要问题是 Pipeline Bubble。流水线启动和结束阶段，会有部分 GPU 等待其他 Stage。

### 实际选型

通常组合为 DP/FSDP + TP + PP，MoE 再加入 EP。原则是：

- TP 尽量限制在高速互联域内。
- PP 跨更慢节点也相对可行，但要减少 bubble。
- DP 扩全局吞吐。
- 并行度不是越高越好，通信、batch、内存和容错必须联合优化。

---

[返回目录](README.md)