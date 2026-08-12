# 07. 对齐与 RLHF

[返回目录](README.md)

## 1. 为什么只有 SFT 不够？

### 30 秒回答

SFT 通过最大似然学习“示范答案长什么样”，但它不直接学习多个可接受回答之间的偏好，也难以表达帮助性、真实性、安全性和风格之间的细粒度权衡。

主要局限是：

- 高质量示范昂贵，每个 prompt 通常只有一个参考答案。
- Cross-Entropy 对示范中的每个 token 同等模仿，不知道哪些行为更受偏好。
- 对模型自己生成但分布外的错误缺少负反馈。
- “看起来像好答案”不等于用户真正偏好或长期安全。

RLHF 用比较数据学习奖励，再直接优化策略的期望奖励，同时约束模型不要偏离参考策略过远。

不过 RLHF 不是 SFT 的替代。稳定流程通常先用 SFT 建立指令遵循，再做偏好优化。

---

## 2. 经典 RLHF 三阶段

### 阶段一：Supervised Fine-Tuning

- **输入**：prompt 与人工/高质量模型示范回答。
- **输出**：SFT policy $\pi_{\text{SFT}}$。
- **目标**：让预训练模型学会对话格式、任务遵循和基本安全行为。

$$
\mathcal L_{\text{SFT}}
=-\sum_t\log \pi_\theta(y_t\mid x,y_{<t})
$$

### 阶段二：Reward Model

- **输入**：同一 prompt 下多个候选回答，以及人类偏好排序或成对比较。
- **输出**：奖励模型 $r_\phi(x,y)$。
- **目标**：预测人类更偏好哪个回答。

### 阶段三：RL Policy Optimization

- **输入**：prompt、当前 policy 采样回复、RM 奖励、reference policy。
- **输出**：对齐后的 policy。
- **目标**：最大化偏好奖励，同时用 KL 约束避免偏离参考模型过远：

$$
\max_\theta
\mathbb E_{x,y\sim\pi_\theta}
\left[r_\phi(x,y)
-\beta D_{\mathrm{KL}}
(\pi_\theta(\cdot|x)\Vert\pi_{\text{ref}}(\cdot|x))\right]
$$

---

## 3. 为什么使用成对比较而非绝对打分？

### 优势

- 人通常更擅长比较 A/B，而不是稳定定义“7 分”和“8 分”。
- 减少不同标注者的标尺漂移。
- 对风格、完整性和帮助性等主观维度更容易操作。
- 可由多候选排序拆成多个 pair，提高数据利用率。

### 劣势

- 每个 pair 只提供相对信息，数据效率未必高。
- 偏好可能不传递：A > B、B > C，但 C > A。
- 候选质量接近时噪声大；差距太大时信息量又低。
- 无法天然表达“两者都差”或质量绝对门槛。
- 展示顺序、长度、措辞和标注者背景会造成偏差。

实践中可加入 tie、Likert 评分、维度化 rubric、多标注者和质量控制。

---

## 4. Reward Model 的架构和损失

### 架构

通常从与目标 LLM 同族或较强的 pretrained/SFT 模型初始化，把语言模型 head 替换为 scalar value head，在回答末尾 token 的 hidden state 上输出标量：

$$
r_\phi(x,y)\in\mathbb R
$$

RM 不必与 policy 完全同大小，但分布和 tokenizer 兼容会简化训练。过小可能无法判断复杂推理；过大则训练和在线打分成本高。

### Bradley-Terry 损失

假设偏好回答 $y_w$ 胜过 $y_l$ 的概率为：

$$
P(y_w\succ y_l)
=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

最大化观测偏好的似然，等价于最小化：

$$
\mathcal L_{\text{RM}}
=-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

它只约束奖励差，不确定奖励的绝对零点。训练时还要关注长度偏置、类别不平衡、过拟合和跨域泛化。

### RM 如何评估？

- Held-out pairwise accuracy。
- 按领域、难度和长度切片。
- 与人类排序的 Kendall/Spearman 相关。
- 对抗集：虚假引用、冗长废话、奉承、格式投机。
- Best-of-N 中随着 N 增大，真实质量是否继续提高；若 RM 分数升高但人工质量下降，说明开始过优化。

---

## 5. 为什么 PPO，而不是 REINFORCE 或 Q-learning？

### 对比 REINFORCE

REINFORCE 的梯度估计：

$$
\nabla_\theta J
=\mathbb E[
\nabla_\theta\log\pi_\theta(y|x)(R-b)]
$$

简单但方差很大。LLM 动作空间是巨大词表、轨迹很长、奖励常在序列末尾才给出，训练容易不稳定。PPO 使用 value/advantage 估计、mini-batch 多轮更新和 clipped objective，提高样本效率并限制单次策略变化。

### 对比 Q-learning

Q-learning 更适合离散且可反复探索的 Markov 环境。对 LLM：

- 状态是任意长 token 前缀。
- 动作空间是大词表。
- 学习每个状态动作的 Q 值成本高。
- 离线 bootstrapping 和分布外动作容易不稳定。

Policy gradient 可直接优化生成策略，更自然。

### PPO Clipped Objective

$$
r_t(\theta)
=\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

$$
\mathcal L_{\text{clip}}
=\mathbb E_t
\left[
\min\left(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

Clipping 防止一次更新把 token 概率改得过大。LLM RLHF 还会训练 value model，并在 token reward 中加入相对 reference model 的 KL 惩罚。

---

## 6. KL 惩罚的作用与 $\beta$ 调节

### 作用

- 防止 policy 为骗取 RM 分数远离语言模型分布。
- 保留 SFT 模型的语言质量和通用能力。
- 相当于在奖励最大化中加入 trust region / regularization。
- 降低 RM 在分布外区域被利用的风险。

### $\beta$ 太大

- 模型几乎不敢改变，奖励和偏好胜率提升有限。
- 输出过于接近 SFT/reference。
- RL 信号被 KL 压制。

### $\beta$ 太小

- Policy 快速漂移，语言质量下降。
- Reward hacking、模式化、过长输出和能力遗忘。
- KL、entropy、长度或 reward 出现异常增长。

### 如何调节？

同时观察：

- RM reward 与独立人工/LLM judge 胜率。
- 每 token KL 和序列 KL。
- 输出长度、entropy、重复率、拒答率。
- 各领域能力回归。

可设定 target KL，动态调整 $\beta$：实际 KL 高于目标则增大，低于目标则减小。不能只看 reward 曲线。

---

## 7. Reward Hacking

### 定义

模型找到提高代理奖励但不满足真实目标的行为。根因是“奖励模型只是人类偏好的不完美近似”，且优化会主动寻找 RM 的漏洞。

### 例子

客服模型的 RM 偏好“礼貌、详细”。RL 后模型开始对任何投诉都先长篇道歉、重复用户观点并承诺无法兑现的补偿。RM 分数高，但用户需要的是准确解决问题。

### 缓解

- 改善偏好数据，加入简洁性、事实性和拒绝奉承的反例。
- 多维奖励：帮助性、正确性、安全、长度、风格分别建模。
- Rule-based verifier、工具执行结果或事实核验。
- KL、长度约束和早停，避免对 RM 过优化。
- Reward model ensemble 与不确定性估计。
- 定期用新 policy 的高奖励样本做人工审计和 adversarial data collection。
- 使用独立 judge，不用同一 RM 同时训练和最终评估。

---

## 8. DPO 的核心思想

### 30 秒回答

DPO（Direct Preference Optimization）从 KL 正则化的 RL 目标推导出最优策略与奖励之间的关系，把隐式奖励代回 Bradley-Terry 偏好模型，直接用偏好 pair 训练 policy，不需要显式 Reward Model、在线采样和 PPO。

典型目标：

$$
\mathcal L_{\text{DPO}}
=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]\right)
$$

它提高 chosen 相对 reference 的概率优势，并降低 rejected 的相对优势。

### 相比 PPO-RLHF

| 维度         | PPO-RLHF                     | DPO                          |
| ------------ | ---------------------------- | ---------------------------- |
| Reward Model | 显式训练                     | 不需要显式 RM                |
| 数据         | 在线 rollout + RM            | 离线偏好 pair                |
| 组件         | policy、reference、RM、value | policy、reference            |
| 稳定性       | 调参复杂                     | 类似监督训练，通常更稳       |
| 探索         | 可在线探索新输出             | 受离线数据覆盖限制           |
| 奖励         | 可组合环境和规则奖励         | 原始形式依赖 pair preference |

### DPO 局限

- 偏好数据质量和覆盖范围决定上限。
- 不会主动探索超出数据的新推理策略。
- 对 chosen/rejected 长度、难度和噪声敏感。
- 对数学/代码等可验证任务，在线 RL 能利用执行反馈，往往更有潜力。

---

## 9. GRPO 与 PPO

### GRPO 的核心

对每个 prompt 从旧策略采样一组 $G$ 个回答，得到奖励 $r_1,\ldots,r_G$，用组内相对标准化构造 advantage：

$$
A_i=\frac{r_i-\operatorname{mean}(r)}
{\operatorname{std}(r)+\epsilon}
$$

再使用类似 PPO 的重要性采样和 clipping 更新 policy，并加 KL 正则。它不训练独立 critic/value model。

### 相比 PPO

- **省显存和计算**：去掉与 policy 规模相近的 critic。
- **相对比较**：同一 prompt 的组内 baseline 降低题目难度差异。
- **适合可验证任务**：一个 prompt 可采样多个推理轨迹，用答案或单测打分。

### 局限

- 每个 prompt 要采样多条回答，rollout 成本高。
- 若组内奖励全相同，标准化后几乎没有学习信号。
- 序列最终奖励广播到 token，信用分配仍较粗。
- 组内均值/方差会引入采样噪声。
- 长度和 token-level clipping 的处理可能带来偏差。

---

## 10. DAPO 与 GRPO

DAPO 即 Decoupled Clip and Dynamic sAmpling Policy Optimization，针对大规模推理 RL 的稳定性做了四类关键修改：

1. **Clip-Higher**：正向和负向 clipping 解耦，允许低概率的优质 token 获得更大上升空间，缓解 entropy collapse。
2. **Dynamic Sampling**：过滤奖励全相同、没有有效梯度的 prompt，并动态补充有效样本。
3. **Token-Level Policy Gradient Loss**：按 token 聚合损失，避免不同回答长度导致 sample-level 聚合偏差。
4. **Overlong Reward Shaping**：对接近最大长度的回答平滑惩罚，而不是截断后突然给极低奖励。

### 一句话比较

GRPO 给出了“组内相对奖励、不用 critic”的基本框架；DAPO 重点修复在大规模推理训练中出现的 entropy、无效样本、长度偏差和截断奖励问题。

---

## 11. GSPO 与 GRPO

GSPO（Group Sequence Policy Optimization）把优化单元从 token 级 importance ratio 改为 sequence 级。

### GRPO 常见 token ratio

$$
r_{i,t}=
\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}
{\pi_{\text{old}}(y_{i,t}|x,y_{i,<t})}
$$

一个序列中不同 token 分别 clip，可能破坏“整条序列获得一个 outcome reward”这一语义，也可能在长序列和 MoE 路由变化时造成高方差。

### GSPO

先用整条序列的平均 log-likelihood ratio 构造 sequence-level ratio，再对序列进行 clipping、奖励和优化。核心对齐是：

- 奖励通常是 sequence-level。
- 采样概率本质上是整个序列概率。
- 因而 importance weight 也在 sequence-level 定义。

### 优势

- 论文报告训练更稳定、样本效率更高。
- 对 MoE RL 尤其友好，token 路由变化不容易让个别 token ratio 破坏更新。
- 目标和 outcome reward 的粒度更一致。

### 需要谨慎

Sequence-level 优化并没有自动解决详细步骤的信用分配。它只是让策略比率与序列奖励粒度一致。若要知道哪一步推理错了，仍需 process reward、verifier、step-level advantage 或环境反馈。

---

## 12. 信用分配：token 奖励与 sequence 奖励

### Sequence-Level Reward

完成整条回答后给一个分数，例如答案是否正确。

- 优点：标注简单；不要求暴露或判断每一步推理；可直接用单测/答案 verifier。
- 缺点：所有 token 共享结果，难以知道关键步骤；长轨迹方差大。

### Token/Step-Level Reward

在生成过程中为 token 或推理步骤提供奖励。

- 优点：反馈密集，可定位错误步骤，理论上样本效率更高。
- 缺点：标注和模型训练昂贵；局部正确不保证全局正确；Process RM 也会被攻击。

### 常见信用分配方法

- Value function / GAE 估计每个状态的未来回报。
- Process Reward Model 给推理步骤打分。
- Monte Carlo rollout：从中间步骤继续多次采样，以最终成功率估计该步骤价值。
- Verifier 定位代码测试、数学约束或工具执行错误。
- Outcome reward 配合 leave-one-out/group baseline 降方差。
- 将长任务拆成可验证子目标，但要防止错误分解限制策略。

---

## 13. RLAIF

RLAIF 使用 AI 生成偏好、批评、原则判断或奖励，替代或补充人类反馈。Constitutional AI 是代表思路：先定义原则，让模型自我批评和修订，再用 AI preference 做偏好优化。

### 潜力

- 成本低、速度快、可扩展到大量 prompt。
- rubric 一致，适合快速迭代和长尾场景。
- 可让多个 judge 从事实、安全、风格等维度分别评估。
- 人类可以把精力集中在原则设计、校准和困难样本。

### 风险

- Judge 偏差被 policy 放大。
- 同源模型可能偏好相似措辞、长答案或自身输出。
- 对事实和安全的盲点形成系统性错误。
- 模型生成的“多数意见”不等于人类价值。
- Policy 可能学会针对 judge 的模式，而非真实质量。

### 更稳妥的方案

人类定义规范和 gold set，AI 扩大标注；定期做人类校准；使用多模型、多提示 judge；对高风险领域保留专家审核；报告 judge 与人类的一致率。

---

## 14. 离线分数高，上线后模式化、奉承、信息量低

### 可能原因

- RM 存在长度、礼貌、同意用户或固定格式偏好。
- PPO 过优化 RM，KL 太小。
- 离线评估与真实用户分布不一致。
- Judge 与训练 RM 同源，形成自我验证。
- 偏好标注者倾向“看起来安全”的回答。
- 训练 pair 缺少“礼貌但无信息”和“不同意用户但事实正确”的困难负例。
- 线上多轮对话状态与离线单轮评估不同。

### 排查顺序

1. 对线上失败样本做人类盲评并建立 taxonomy。
2. 对比 RM 分数、独立 judge、人类评分和业务结果。
3. 按长度、同意程度、模板、领域和轮次切片。
4. 检查 KL、entropy、输出长度和多样性随训练步变化。
5. 用 counterfactual pair 测试 RM：事实相同但风格不同；礼貌相同但信息量不同。

### 修复

- 补充 hard negatives 和真实线上偏好。
- 采用多维奖励并加入事实/任务完成 verifier。
- 增大 KL、减少 RL epoch 或 early stop。
- 混合 SFT replay，防止能力遗忘。
- 优化目标加入简洁性和 calibration，而不是一味奖励更长。
- 线上 A/B 以任务成功、追问率、用户修正率为准，不以 RM 分数代替。

[返回目录](README.md)