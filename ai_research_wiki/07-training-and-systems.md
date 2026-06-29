# 05. 训练与系统

[返回目录](README.md)

## 1. 训练百亿/千亿参数 LLM 的主要挑战

### 30 秒回答

挑战可分成五类：

1. **显存**：模型参数、梯度、优化器状态和激活远超单卡容量。
2. **计算**：训练 FLOPs 巨大，需要高利用率的 GPU 集群。
3. **通信**：数据并行同步、张量并行 collective、流水线传输和 MoE all-to-all 都可能成为瓶颈。
4. **稳定性**：loss spike、梯度爆炸、数值溢出、坏数据和路由失衡可能导致训练失败。
5. **数据与实验治理**：去重、质量、污染、版权、多语言配比，以及 checkpoint 和故障恢复。

解决方案通常是混合精度、activation checkpointing、ZeRO/FSDP、3D 并行、FlashAttention、高效 kernel、稳定初始化和完善的监控恢复机制。

---

## 2. 显存由什么构成？

以 AdamW 混合精度训练作粗略估算，每个参数可能需要：

- BF16/FP16 参数：2 bytes。
- 梯度：2 bytes。
- FP32 master weight：4 bytes，具体框架可能省略或实现不同。
- Adam 一阶和二阶状态：8 bytes。

合计常用粗估约 12--16 bytes/parameter，还未包含激活、临时 buffer、通信 bucket 和碎片。

一个 70B 模型若完全复制到单个 data-parallel rank，仅模型状态就可能超过 840 GB。实际必须切分。

### 激活为什么也很大？

激活大致随：

$$
\text{batch} \times \text{sequence length}
\times \text{hidden size} \times \text{layers}
$$

增长。长序列训练中，激活可能超过模型状态。Activation Checkpointing 只保存部分边界激活，反向时重算中间结果，用计算换显存。

---

## 3. 常见并行策略

### Data Parallelism（DP）

每张卡有完整模型，处理不同 mini-batch，再同步梯度。

- 实现简单，扩展 batch。
- 模型必须装得下；all-reduce 通信随模型增大。

### ZeRO / FSDP

在 data-parallel rank 间切分状态：

- ZeRO-1：切 optimizer states。
- ZeRO-2：再切 gradients。
- ZeRO-3 / FSDP full shard：再切 parameters，计算前 all-gather，用后释放或 reshard。

优点是显著降低单卡模型状态；代价是更频繁的通信和调度复杂度。

### Tensor Parallelism（TP）

把单层矩阵乘法沿维度切到多卡。例如列并行和行并行线性层。适合模型单层过大时，但层内频繁 collective，对高速互联要求高。

### Pipeline Parallelism（PP）

按层切成多个 stage，用 micro-batch 流水执行。问题是 pipeline bubble、stage 不均衡，以及前后向调度复杂。

### Sequence / Context Parallelism

沿序列维度切分激活或 Attention，适合超长上下文。需要处理 Q/K/V 跨卡通信和 causal 依赖。

### Expert Parallelism（EP）

MoE 的不同 expert 放在不同设备，token 通过 all-to-all 路由。负载均衡和网络拓扑非常关键。

### 实际选型

通常组合为 DP/FSDP + TP + PP，MoE 再加入 EP。原则是：

- TP 尽量限制在高速互联域内。
- PP 跨更慢节点也相对可行，但要减少 bubble。
- DP 扩全局吞吐。
- 并行度不是越高越好，通信、batch、内存和容错必须联合优化。

---

## 4. 训练吞吐和 MFU

Model FLOPs Utilization（MFU）近似衡量实际有效模型计算相对硬件峰值的比例：

$$
\text{MFU}=
\frac{\text{achieved model FLOPs/s}}
{\text{theoretical peak FLOPs/s}}
$$

MFU 受以下因素影响：

- 小矩阵、短序列或 micro-batch 太小，GPU 未吃满。
- 通信与计算没有重叠。
- 数据加载、CPU 预处理或存储成为瓶颈。
- Padding 浪费。
- Kernel launch 和中间张量读写过多。
- Pipeline bubble、MoE 负载不均。

常见优化包括 sequence packing、FlashAttention、fused optimizer/RMSNorm/MLP、通信重叠、拓扑感知放置和异步 checkpoint。

---

## 5. 为什么会训练不稳定？

### 常见原因

- 学习率过大或 warmup 不足。
- 初始化、残差尺度和深度不匹配。
- FP16 动态范围不足造成 overflow；BF16 通常更稳。
- 数据异常：超长重复、乱码、极端 token 分布、错误 mask。
- Attention logit 或 softmax 数值异常。
- MoE router 塌缩或 expert 负载失衡。
- 分布式 bug 导致部分 rank 参数或梯度不一致。
- Checkpoint 恢复时 optimizer/scheduler/RNG 状态不完整。

### 监控与处理

监控总 loss 和分 domain loss、梯度范数、参数范数、学习率、每层激活/梯度统计、NaN/Inf、吞吐、通信时间、expert load 和数据 batch ID。

应对措施包括 gradient clipping、降低学习率、延长 warmup、跳过坏 batch、稳定 softmax、使用 BF16/FP32 accumulation、QK norm、调整初始化与残差缩放。

**不要看到 loss spike 就直接回滚。** 先判断它是可恢复尖峰、单个坏 batch、数值错误还是持续发散。回滚必须配合根因定位，否则还会重复出现。

---

## 6. 数据工程挑战

### 预训练数据流水线

1. 获取网页、书籍、代码、论文、多语言和领域数据。
2. 文档解析、语言识别、PII/安全过滤。
3. 质量分类或打分。
4. URL、文档和近似重复去重。
5. Benchmark contamination 检测。
6. 按领域和语言设定 mixture。
7. Tokenize、分片、shuffle、packing。
8. 保存数据版本和 lineage，保证实验可复现。

### 为什么去重重要？

- 减少记忆和隐私泄漏。
- 防止重复文档主导梯度。
- 降低测试集污染。
- 提升每个训练 token 的信息密度。

### 数据质量与数据量

高质量数据通常更高效，但“质量”依赖目标：代码、数学、多语言、对话和事实知识需要不同的数据分布。实践中不是只做一个全局质量分，而是分 domain 建模和配比。

---

## 7. 常见开源训练与推理框架

### 训练

- **PyTorch FSDP**：原生参数分片，适合 PyTorch 生态。
- **DeepSpeed**：ZeRO、并行训练和推理优化较完整。
- **Megatron-LM / Megatron-Core**：大规模 TP/PP/CP 和高性能 kernel。
- **Hugging Face Transformers + Accelerate**：模型和训练接口丰富，研究迭代方便。
- **Colossal-AI**：多种并行和大模型训练工具。
- **verl / TRL / OpenRLHF**：LLM 后训练和 RL。

### 推理

- **vLLM**：PagedAttention、continuous batching，高吞吐服务。
- **SGLang**：结构化生成、radix cache 和复杂 LLM 程序执行。
- **TensorRT-LLM**：NVIDIA 平台高性能 kernel 和量化部署。
- **llama.cpp**：CPU/边缘设备和 GGUF 量化生态。

面试中不要只罗列名称。应说明你为什么选它、模型规模、硬件、吞吐/延迟目标、遇到什么问题，以及如何验证。

---

## 8. Qwen 系列可讲的创新点

下面以 Qwen2/Qwen2.5 时代的公开技术报告为主，面试时应明确具体版本。

### 架构和训练

- Decoder-only Transformer，使用 RoPE、RMSNorm、SwiGLU、GQA 等成熟设计。
- 扩大高质量预训练数据，Qwen2.5 报告中从约 7T 扩到 18T tokens。
- 多尺寸、稠密/MoE、通用/代码/数学/视觉语言等完整模型谱系。
- 大规模 SFT 与多阶段强化学习，强化指令遵循、结构化输出、数学和代码能力。

### 长上下文

Qwen2.5-1M 结合长文本数据合成、渐进式长上下文训练、RoPE 外推和稀疏注意力/Chunked Prefill 等推理优化。

### 如何评价

优势是完整生态、多语言和中文能力、尺寸覆盖、专业模型与部署友好。讨论创新时要区分：

- 架构首创。
- 对已有方法的大规模、系统化工程实现。
- 数据和后训练带来的能力提升。

不要把使用 GQA、RoPE 本身说成 Qwen 首创。

---

## 9. DeepSeek 可讲的创新点

### DeepSeek-V2/V3 架构

**Multi-head Latent Attention（MLA）**

把 K/V 压缩为低维 latent 表示并缓存，需要时通过投影恢复相关表示，显著降低 KV Cache。还要正确处理 RoPE 位置相关部分，避免位置旋转妨碍低秩吸收。

**DeepSeekMoE**

使用细粒度 routed experts 和 shared experts，提高 expert 专门化与知识共享。

**Auxiliary-loss-free load balancing**

传统 MoE 通过辅助损失平衡路由，但可能干扰语言模型主目标。V3 使用动态调整的 expert bias 促进负载均衡，避免把平衡项直接加入主损失。

**Multi-Token Prediction（MTP）**

训练时预测未来多个 token，提供更密集训练信号；推理时也可与 speculative decoding 等方向结合。

**系统工程**

V3 报告强调 FP8 混合精度、通信计算重叠、DualPipe 等工程优化。回答时应把“算法创新”和“系统优化”分开。

### DeepSeek-R1

- R1-Zero 展示了不用预先 SFT、直接对 base model 做大规模 RL 也能激发推理行为。
- 使用 GRPO 和可验证的准确率/格式奖励。
- 纯 RL 版本存在可读性和语言混杂问题；R1 加入 cold-start 数据、多阶段 RL/SFT。
- 用强模型生成推理数据蒸馏到较小的 Qwen/Llama 模型，说明推理模式可通过 SFT 有效迁移。

### 批判性评价

- 可验证奖励最适合数学和代码，对开放域任务仍依赖 reward model 或 AI feedback。
- 长 CoT 不等于正确推理，可能出现过度思考和 reward hacking。
- 公开报告没有披露所有训练细节，复现结论要区分论文事实与社区推测。

---
