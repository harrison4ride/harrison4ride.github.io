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

就是逐 token 的 cross-entropy：给定 prompt $x$ 和已生成的前缀 $y_{<t}$，让模型把示范答案的下一个 token $y_t$ 的概率推高。注意它对示范里的每个 token 一视同仁，这正是 SFT 学不到偏好的原因。

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

这个目标函数是后面所有内容的总纲，两项在拔河：前一项要**分数高**（RM 觉得好），后一项要**别跑远**（离参考模型不能太离谱），$\beta$ 是这场拔河的力度旋钮。注意 $y\sim\pi_\theta$——回答是模型自己采样出来的，不是数据集里给定的，这正是它属于 RL 而不是监督学习的地方。

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

读法：两个回答的**分数差**经 sigmoid 变成一个概率。差得越多，人类越可能偏好 $y_w$；分数相等时概率是 0.5，也就是「五五开」。

最大化观测偏好的似然，等价于最小化：

$$
\mathcal L_{\text{RM}}
=-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$

这就是把上面那个概率取负对数——本质上是一个二分类的 log loss，只不过分类的对象是「哪个回答更好」。它逼着 RM 把 chosen 的分数拉高、rejected 的压低，且两者拉得越开惩罚越小。

由于式子里只出现分数差，它只约束奖励差，不确定奖励的绝对零点（整体加减一个常数，loss 不变）。训练时还要关注长度偏置、类别不平衡、过拟合和跨域泛化。

### RM 如何评估？

- Held-out pairwise accuracy。
- 按领域、难度和长度切片。
- 与人类排序的 Kendall/Spearman 相关。
- 对抗集：虚假引用、冗长废话、奉承、格式投机。
- Best-of-N 中随着 N 增大，真实质量是否继续提高；若 RM 分数升高但人工质量下降，说明开始过优化。

---

## 5. PPO：经典 RLHF 的基石

### 先看难在哪

RL 在游戏和机器人上跑得好好的，搬到 LLM 上却格外难训，原因有三个：

- **动作空间极大**：每一步要从整个词表（几万到十几万个 token）里挑一个。
- **轨迹很长**：一条回答几百上千个 token，等于几百上千步连续决策。
- **奖励来得极晚**：数学题、代码题往往要整段生成完、跑完单测才知道对不对，这叫 outcome reward。中间几百步全是「不知道好坏」。

三件事叠在一起的直接后果是**策略梯度的方差极大**：同一个 prompt 采样两次，回报可能天差地别，参数被推向完全不同的方向，训练极易不稳定。PPO 的每一个设计，都是围绕「把方差压下去、把单次更新限制住」展开的。

### 为什么不是 REINFORCE 或 Q-learning？

**对比 REINFORCE**

$$
\nabla_\theta J
=\mathbb E[
\nabla_\theta\log\pi_\theta(y|x)(R-b)]
$$

读法：$\nabla_\theta\log\pi_\theta(y|x)$ 指向「让这条回答更容易被生成」的方向，前面乘的标量 $(R-b)$ 决定推还是拉——回报高于基线 $b$ 就正向推一把，低于基线就往回拽。$b$ 只为降方差，不改变梯度的期望。

它足够简洁，但整条回答只有一个标量在指挥几百步 token 的更新，方差大到难以直接使用。PPO 在此之上加了 value/advantage 估计、mini-batch 多轮复用同一批采样，以及 clipped objective，样本效率和稳定性都好得多。

**对比 Q-learning**

Q-learning 更适合离散且可反复探索的 Markov 环境。对 LLM：

- 状态是任意长的 token 前缀。
- 动作空间是整个大词表。
- 给每个「状态—动作」学一个 Q 值成本极高。
- 离线 bootstrapping 碰上分布外动作容易发散。

直接优化生成策略的 policy gradient 要自然得多。

### 核心机制三件套

**① Clipped Objective：不许一步迈太远**

先定义新旧策略在同一个 token 上的概率比（importance sampling ratio）：

$$
r_t(\theta)
=\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

$r_t>1$ 表示更新后模型更愿意吐出这个 token，$r_t<1$ 表示更不愿意。它衡量的就是「这一步把策略改动了多少」。

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

$A_t$ 是优势（advantage），表示这个 token 比预期好多少。式子读作：正常按 $r_tA_t$ 更新，但一旦 $r_t$ 跑出 $[1-\epsilon,1+\epsilon]$（常用 $\epsilon=0.2$）就换成截断版本，外面再套一个 $\min$ 取更保守的那个。

效果是**好 token 的概率最多提到 1.2 倍就不再给奖励，坏 token 最多压到 0.8 倍**，想再往前冲梯度直接归零。这正是名字里 proximal（近端）的由来：每次只在旧策略附近挪一小步，防止单次更新过猛毁掉模型能力。

**② Actor-Critic 双模型：把方差压下去**

除了负责生成的 policy（actor），PPO 还要训练一个与 policy 规模相近的 value model / critic，用来估计每个状态的价值 $V(s_t)$，进而算出优势：

$$
A_t\approx R_t-V(s_t)
$$

也就是「实际拿到的回报」减去「本来预期能拿多少」（实践中用 GAE 做更平滑的估计）。有了这个减法，梯度信号从「这条回答的绝对得分」变成「比预期好还是差」，方差大幅下降。

它其实就是 REINFORCE 里那个基线 $b$ 的升级版：$b$ 是一个全局常数，$V(s_t)$ 是随状态变化、并且学出来的基线。

**③ KL 惩罚：别跑离语言模型太远**

在 reward 里减去相对 reference model（通常就是 SFT 模型）的 KL 惩罚：

$$
\tilde r
=r_\phi(x,y)
-\beta D_{\mathrm{KL}}
(\pi_\theta\Vert\pi_{\text{ref}})
$$

防止模型为了刷高 RM 分数而偏离正常语言分布。这一项的调节是实操中最容易翻车的地方，下一节单独讲。

### 痛点与局限

- **显存和计算开销极高**：训练时要同时驻留 4 个大模型——policy（actor，训练中）、critic（训练中）、reward model（推理）、reference model（推理）。其中 critic 通常和 policy 规模相近，等于凭空多出一个正在训练的大模型，显存直接爆炸。
- **调参复杂**：PPO 的成败很大程度上取决于 critic 拟合得好不好。critic 估不准 → advantage 有偏 → actor 被带偏 → 采样分布随之变化 → critic 更估不准。actor 与 critic 的这场博弈很容易导致训练崩盘。

后面的 DPO 和 GRPO，本质上都是在削减这套架构的成本。

---

## 6. RLHF 训练的两大核心难题

上一节三件套里的 KL 惩罚，和它要防的 reward hacking，是实际跑 RLHF 时最容易翻车的两处。两者是一体两面：KL 是手段，防的就是 hacking。

### ① KL 惩罚与 $\beta$ 调节

**作用**：扮演一根安全绳，防止模型为了拿高 RM 分数而丢掉原有的语言通顺度、语法能力和泛化能力。换个说法，它相当于在奖励最大化中加了 trust region / 正则项，同时降低 RM 在分布外区域被钻空子的风险。

**$\beta$ 过大**：模型过于保守，几乎不敢改变，输出黏在 SFT/reference 上，RL 信号被 KL 压死，奖励和胜率都提不动——等于 RL 优化无效。

**$\beta$ 过小**：策略快速漂移，语言质量下降，随之而来的是 reward hacking、模式化重复、输出越写越长、原有能力遗忘；监控上会看到 KL、entropy、长度或 reward 异常增长。

**工程实践**：不能只盯 reward 曲线，要同时观察

- RM reward 与独立人工 / LLM judge 的胜率；
- 每 token KL 和整条序列的 KL；
- 输出长度、entropy、重复率、拒答率；
- 各领域的能力回归测试。

常用做法是设一个 **target KL**，据此动态缩放 $\beta$：实际 KL 高于目标就调大 $\beta$ 收紧，低于目标就调小放松。

### ② Reward Hacking

**现象**：RM 只是人类偏好的不完美代理（proxy）。RL 优化会主动去找 RM 的漏洞——只要某种行为能骗到高分，模型就会把它做到极致。

典型例子：客服模型的 RM 偏好「礼貌、详细」。RL 之后，模型对任何投诉都先长篇道歉、重复一遍用户的观点、再承诺一个兑现不了的补偿。RM 分数一路走高，而用户真正需要的「把问题解决掉」反而消失了。

**应对手段**：

- **数据层面**：偏好数据里加入针对冗长废话、讨好奉承（sycophancy）、虚假引用、格式投机的反例和 hard negatives，以及强调简洁性和事实性的正例。
- **奖励层面**：多维奖励（帮助性、正确性、安全、长度、风格分开建模）；长度惩罚（length penalty）；rule-based verifier、工具执行结果或事实核验做硬校验；RM ensemble 配不确定性估计。
- **约束层面**：KL 约束、长度约束和早停，避免对 RM 过优化。
- **评估隔离**：**严禁**用训练用的那个 RM 评估最终模型，必须换独立第三方 judge（更强的模型）或人工盲评；同时定期把新 policy 的高奖励样本捞出来人工审计，做 adversarial data collection。

一个实用的过优化信号：Best-of-N 里随着 N 增大，RM 分数继续升高但人工判定的质量开始下降，说明已经在刷分了。

---

## 7. DPO 的核心思想

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

怎么读这个式子：括号里两项形状完全一样，都是「当前模型相对参考模型，把这条回答的概率抬高了多少」——DPO 把这个比值的对数当作**隐式奖励**。于是 chosen 减 rejected，就是一个 Bradley-Terry 里的分数差，外面套上 $-\log\sigma(\cdot)$，正好就是第 4 节那个 RM 损失的形状。

区别只在于优化对象换了：RM 训练时调的是 $r_\phi$，DPO 直接调 $\pi_\theta$。所以它提高 chosen 相对 reference 的概率优势，同时压低 rejected 的相对优势。

这也意味着 DPO 的训练循环和 SFT 几乎一样：读一批离线 pair、前向、算 loss、反向——没有采样，没有 RM，没有 critic。$\beta$ 依然在（它来自原 RL 目标里的 KL 系数），控制允许偏离 reference 多远。

### DPO 局限

- 偏好数据质量和覆盖范围决定上限。
- 不会主动探索超出数据的新推理策略。
- 对 chosen/rejected 长度、难度和噪声敏感。
- 对数学/代码等可验证任务，在线 RL 能利用执行反馈，往往更有潜力。

---

## 8. GRPO：面向 Reasoning 的轻量化革命

针对 PPO 的显存和调参痛点，GRPO（Group Relative Policy Optimization）做了一个关键革新：**用同组样本的平均表现，替掉那个学出来的 critic**。

### 核心机制

**① 取消 Critic**：完全摒弃独立的 value / critic model，直接省掉接近一半的训练显存与计算量。

**② Group Sampling（组内采样）**：对同一个 prompt，让旧策略一次性采样出一组 $G$ 个候选回答，各自得到奖励 $r_1,r_2,\ldots,r_G$。

**③ Group Relative Advantage（组内相对归一化）**：直接用这组奖励的均值和标准差算出每个回答的相对优势：

$$
A_i=\frac{r_i-\operatorname{mean}(r)}
{\operatorname{std}(r)+\epsilon}
$$

读法：$\operatorname{mean}(r)$ 充当 baseline，$\operatorname{std}(r)$ 把不同题目的奖励尺度拉齐，$\epsilon$ 防除零。答得比同组平均好，$A_i>0$，这条轨迹被强化；比平均差就被压下去。

关键在于**这个 baseline 是自适应的**：难题上大家都答不好、平均分低，你只要比同组稍好就是正优势；简单题上大家都对，优势自然趋近 0。它抵消的正是不同 prompt 之间的难度差异——而这本来是 PPO 里 critic 的职责。

**④ PPO 式更新**：拿着这个组内 advantage $A_i$，继续套用 PPO 的 ratio clipping 和 KL 正则去更新 policy。所以 GRPO 不是另起炉灶，而是把 PPO 的 advantage 来源换了。

### 相比 PPO

- **省显存和计算**：去掉与 policy 规模相近的 critic，训练时只剩 policy、reference 和 RM/verifier。
- **相对比较更稳**：同一 prompt 的组内 baseline 天然消化了题目难度差异，也不存在 critic 估不准带崩 actor 的问题。
- **适合可验证任务**：一个 prompt 可以采样多条推理轨迹，用标准答案或单元测试直接打分，连 RM 都可以省掉。

### 局限

- 每个 prompt 要采样 $G$ 条回答，rollout 成本高。
- 若组内奖励全相同（全对或全错），标准化后 $A_i$ 全是 0，这批数据几乎没有学习信号。
- 序列最终奖励广播到每个 token，信用分配仍然较粗。
- 组内均值/方差本身带采样噪声，$G$ 太小时不稳。
- 长度和 token-level clipping 的处理可能引入偏差。

---

## 9. 推理任务的奖励设计

前面几节的奖励都来自一个学出来的 RM。到了数学、代码这类**答案可验证**的推理任务，奖励的设计方式完全不同，也是当前推理模型（o1、R1 这条线）的核心。

### 可验证奖励（RLVR）

奖励不再是模型打分，而是一个**程序**：数学题比对最终答案、代码跑单元测试、形式化证明送进检查器，再加一点格式奖励（比如是否把推理过程写进指定标签里）。

好处是根本性的：规则不会被说服。第 6 节讲的 reward hacking 之所以存在，是因为 RM 只是人类偏好的近似；而单元测试没有「被讨好」这个概念，模型只能真的把题做对。这就是为什么这条路线能放心地把 RL 跑得很久。

代价是适用面窄——只能用在答案可自动判定的领域，开放式写作、对话依然得靠 RM。

### Outcome Reward vs. Process Reward

| 维度     | ORM（Outcome Reward Model） | PRM（Process Reward Model）    |
| -------- | --------------------------- | ------------------------------ |
| 打分对象 | 只看最终答案对不对          | 给推理过程的**每一步**打分     |
| 标注成本 | 低，有标准答案就能自动判    | 高，要人标或自动标每一步       |
| 信号密度 | 稀疏，整条轨迹只有一个分    | 稠密，每步都有反馈             |
| 主要问题 | 蒙对也给满分                | 标注贵、步骤边界难定义、易被 hack |

**ORM 的隐患**：模型完全可能靠一堆错误推理凑出正确答案，ORM 照样给满分——等于在奖励错误的推理模式。信用分配也粗：整条几千 token 的推理只有一个分，模型不知道是哪一步做对了。

**PRM 怎么标**：早期靠人工逐步标注（OpenAI 的 PRM800K）。后来有了自动化办法——从某一步出发多次 rollout，如果从这里出发能高频到达正确答案，就认为这一步是好的（Math-Shepherd 的思路），把「步骤质量」转化成「后续成功率」。

**现在的实践取向**：R1 的经验是，**outcome reward 加规则验证就足以激发长链推理**，PRM 在大规模 RL 里训练复杂且容易被 hack（模型学会写出「看起来很像正确步骤」的话）。所以主流 RL 训练转向可验证的 outcome reward。但 PRM 在**推理时的重排序**（Best-of-N 里给候选打分，见 [09. 推理与部署](09-inference-and-serving.md)）依然好用——那里不做梯度更新，模型没有机会去 hack 它。

### RL 激发推理能力

DeepSeek-R1 报告里两个值得记住的观察：

- **R1-Zero**：完全跳过 SFT，直接在 base model 上用可验证奖励做 RL，模型**自己涌现出**了长链推理、回头检查、换一种解法这些行为——没人教它「要反思」，只是因为反思能提高答对率。代价是可读性差、中英文混杂。
- **R1**：先用少量高质量长推理数据做冷启动 SFT，再做 RL，兼顾了可读性和能力。

要注意 RL 的作用边界：它**强化的是模型已有的行为**，而不是凭空创造新能力。这也解释了为什么小模型直接做 RL 效果远不如从大模型蒸馏（见 [07. 训练与系统](07-training-and-systems.md)）——小模型的采样里压根就没出现过那些好行为，自然无从强化。

---

## 10. 路线选型：PPO vs. DPO vs. GRPO

| 路线        | 数据与采样              | 训练时要加载的模型                             | 强项                                     | 短板                                   |
| ----------- | ----------------------- | ---------------------------------------------- | ---------------------------------------- | -------------------------------------- |
| PPO-RLHF    | 在线 rollout + RM 打分  | policy、reference、RM、critic（4 个）          | 在线探索能力强，可组合环境奖励和规则奖励 | 显存与算力极贵，调参复杂、容易崩       |
| DPO         | 离线偏好 pair           | policy、reference（2 个）                      | 像 SFT 一样稳和便宜，无需 RM 和采样      | 受离线数据覆盖限制，缺乏自我探索       |
| GRPO / DAPO | 在线采样一组 $G$ 个回答 | policy、reference、RM 或 verifier（无 critic） | 省掉 critic，适合有标准答案的推理任务    | rollout 成本高，组内奖励雷同就没有信号 |

一句话选型：

- 通用偏好对齐、预算有限 → 从 DPO 起步。
- 需要在线探索、要接入规则或环境奖励 → PPO。
- 数学、代码这类答案能自动验证的推理任务 → 走 GRPO 这条线，DAPO、GSPO 都是它的后续改进，也是目前复杂推理对齐的主流方向。

---

# 目录

| 章节                                         |
| -------------------------------------------- |
| [00. 机器学习核心概念](00-ml-concepts.md)    |
| [01. 基础与神经网络机制](01-foundations.md)  |
| [02. 模型评估与指标](02-evaluation.md)       |
| [04. 经典机器学习](04-classical-ml.md)       |
| [05. NLP、RNN 与词向量](05-nlp-rnn.md)       |
| [06. LLM 基础](06-llm-foundations.md)        |
| [07. 训练与系统](07-training-and-systems.md) |
| [08. 对齐与 RLHF](08-alignment-and-rlhf.md)  |
| [09. 推理与部署](09-inference-and-serving.md) |
| [12. ML Coding](12-ml-coding.md)             |
| [参考资料](references.md)                    |