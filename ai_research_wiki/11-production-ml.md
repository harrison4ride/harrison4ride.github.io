# 09. 项目经验与 Production ML

[返回目录](README.md)

## 1. 离线训练好的模型在线上不工作，可能是什么原因？

### 30 秒回答

我会先确认“不工作”的定义和退化发生在哪一层，再按以下顺序排查：

1. 指标、流量和实验设置是否正确。
2. 训练与线上特征流水线是否一致。
3. 数据分布、标签定义或用户行为是否变化。
4. 离线指标是否与业务目标错位。
5. 阈值、校准、延迟和系统约束是否破坏模型效果。

不能一开始就假设“模型泛化差”，因为生产问题经常来自数据管道、实验分流或业务集成。

### 数据与标签问题

- Train/validation 存在 leakage，离线成绩虚高。
- 训练数据不代表线上用户、时间段、语言或设备。
- 标签延迟、标签噪声或线上标签定义不同。
- 线上出现新类别、冷启动用户或长尾输入。
- Sampling strategy 改变了类别先验。
- 训练数据经过人工清洗，线上原始数据更脏。

### Training-Serving Skew

- 特征计算代码、窗口、时区、单位或归一化不同。
- 类别 ID、词表、缺失值编码或 schema 不一致。
- 线上特征过期，event-time 与 processing-time 混淆。
- Join key 错误、重复记录或默认值比例异常。
- 模型、preprocessor、feature store 版本不匹配。

应对关键特征做离线/在线 parity test：给定相同原始事件，两条流水线必须产生相同特征。

### 目标和评估错位

- 优化 AUC，但业务关心固定容量下的 Precision@K。
- 平均指标提高，但重要用户切片退化。
- 离线数据没有模拟曝光偏差、反馈回路或竞争环境。
- 模型概率未校准，线上阈值沿用旧模型。
- Proxy label 与真实长期目标不一致。

### 追问：离线指标高，但业务没提升，先怀疑什么？

优先怀疑 metric mismatch 和数据问题：

- 离线指标和业务目标不一致，例如 AUC 提升但 Precision@K 没提升。
- validation/test 分布不代表线上流量。
- 概率没校准，阈值沿用旧模型。
- 平均指标提升，但关键用户切片退化。
- 线上延迟、fallback、日志归因影响真实结果。

### 系统问题

- 延迟超时导致模型结果被 fallback 覆盖。
- 特征服务、缓存或下游规则修改模型输出。
- 量化、模型转换或不同推理库造成数值偏差。
- A/B 分流、日志或指标归因错误。
- 吞吐压力下发生丢请求、批处理 padding 或状态串扰。

### 推荐排查流程

1. 用少量 request ID 端到端重放，核对原始输入、特征、logit、概率、阈值和最终动作。
2. 比较 offline replay、shadow traffic 与正式线上结果。
3. 检查 schema、missing rate、特征分布和预测分布。
4. 按用户、时间、设备、地区和标签切片。
5. 人工审查典型 false positive / false negative。
6. 定位后再决定修数据、调阈值、重新校准还是重新训练。

---

## 2. Loss 变成 Inf 或 NaN 的原因

### 优化与梯度

- 学习率过大，参数或激活迅速爆炸。
- 梯度爆炸，特别是深层网络、RNN 或长序列。
- 没有 gradient clipping，或 clipping 在错误时机执行。
- 初始化、残差尺度或 optimizer hyperparameter 不合理。
- 恢复 checkpoint 时 optimizer state、scheduler 或 loss scaler 不一致。

### 数值不稳定

- 直接计算 `exp(large_value)`。
- 计算 $\log(0)$、除以 0、$\sqrt{x<0}$。
- Softmax/Log Loss 没使用 log-sum-exp 或 logits fused implementation。
- Normalization 的方差太小且 $\epsilon$ 不足。
- FP16 overflow/underflow；BF16 动态范围通常更大，但也不能避免所有问题。
- 自定义 loss 中先求极小概率，再取 log。

### 数据问题

- 输入、label 或 sample weight 已包含 NaN/Inf。
- 异常大值、错误单位或未归一化特征。
- 非法标签，例如分类 label 越界。
- 空 batch、全 mask、长度为 0 或分母计数为 0。
- 数据增强产生无效值。

### 分布式与实现问题

- 某个 rank 先出现 NaN，经 all-reduce 污染全部 rank。
- Mask 符号或广播错误，把全部 logits 设为 $-\infty$。
- In-place operation 破坏 autograd 所需张量。
- 混合精度下错误的 loss scaling 或 cast。
- 自定义 kernel 越界、竞态或未初始化内存。

### 排查顺序

1. 定位第一个出现非有限值的 step 和 batch，而不是只看最终 loss。
2. 检查原始输入、label 和每个 loss component 是否 finite。
3. 记录逐层 activation、logit、parameter、gradient norm。
4. 暂时切到 FP32、降低学习率、关闭复杂优化和数据增强，缩小问题。
5. 开启 `torch.autograd.detect_anomaly()` 做小规模定位。
6. 若只在分布式出现，比较各 rank 的第一个异常 tensor。
7. 修复根因后从异常前的健康 checkpoint 恢复。

只加一个很大的 $\epsilon$ 或跳过所有 NaN batch 可能掩盖问题，不应作为最终修复。

---

## 3. 如何检测 Data Shift？

### 三类 Shift

- **Covariate shift**：$P(X)$ 改变，但 $P(Y\mid X)$ 基本不变。
- **Label shift**：$P(Y)$ 改变，但类条件分布 $P(X\mid Y)$ 基本不变。
- **Concept drift**：$P(Y\mid X)$ 改变，相同输入与目标的关系发生变化。

仅检测输入分布不能证明 concept drift；后者通常需要延迟到达的真实标签或可靠 proxy。

### 数据质量与 Schema

首先监控：

- 类型、取值范围、类别集合和 schema 版本。
- Missing rate、默认值率、唯一值数量。
- 数据量、延迟、重复率和 join success rate。
- 特征新鲜度和时间窗口。

这一步能快速发现管道故障，不应直接用复杂 drift detector 替代。

### 追问：Model drift 和 data bug 怎么区分？

- data bug 通常是突变：某个字段 missing rate 突然 90%，类别全变 unknown，join success rate 掉了。
- drift 通常是分布逐渐或周期性变化，例如季节、用户行为、产品策略改变。

先排 data quality，再谈模型漂移。

### 单变量检测

- 连续变量：KS test、Wasserstein distance、PSI、分位数变化。
- 类别变量：Chi-square、Jensen-Shannon divergence、类别频率变化。
- 预测：score distribution、entropy、positive rate。

大样本下很小差异也可能统计显著，因此必须同时设置 effect-size 和业务阈值。

### 多变量检测

- 训练一个 domain classifier 区分 reference 和 current data；AUC 明显高于 0.5 表示联合分布可区分。
- 对 embedding 或高维特征使用 MMD、energy distance 或聚类覆盖。
- 按重要业务切片监控，而不是只看全局平均。

### 有标签后的性能监控

- Accuracy、AUC、PR-AUC、calibration、业务收益。
- 各切片的错误率和置信区间。
- 标签延迟分层，避免只评估最快到达的一部分标签。
- Champion/challenger 和 shadow evaluation。

### Drift 告警设计

不要让每个特征的轻微变化都触发告警。应组合：

- 数据质量硬规则。
- Drift score 持续多个时间窗口超阈值。
- 重要特征加权。
- 预测或业务指标同步异常。
- 人工可解释的 top shifted features。

---

## 4. Data Shift 如何补救？

### 先判断根因

- 数据 bug：修复 pipeline 并回填数据。
- 季节性或短期事件：调阈值、扩大窗口或临时 fallback。
- 类别先验变化：在假设成立时做 prior correction / label-shift adjustment。
- Covariate shift：importance weighting、补采样新分布数据。
- Concept drift：收集新标签并重新训练或持续学习。

### 常见方案

- 重新校准概率或重新选择 decision threshold。
- 使用近期数据 fine-tune/retrain，并保留历史 replay 防遗忘。
- 增加新类别、unknown bucket 和 OOD detection。
- Domain adaptation 或 invariant representation。
- Online learning，但必须防反馈污染和灾难性遗忘。
- 回退到更简单、鲁棒或规则化的 baseline。

### 生产机制

- 数据和模型版本化。
- 定期 retraining 与 event-triggered retraining 结合。
- Canary、shadow、A/B test 后再全量发布。
- 保留 rollback 和人工审批。
- 将每次 drift incident 变成 regression test。

---

## 5. 标注数据有限时如何训练？

### 优先级

1. 使用预训练模型或 foundation model 做 transfer learning。
2. 设计高质量、覆盖真实分布的 validation/test set。
3. 用 active learning 把人工标注花在最有信息量的样本上。
4. 再考虑 weak supervision、semi-supervised learning 和合成数据。

测试集不能全部被主动学习或伪标签过程污染，否则无法知道是否真正提升。

### Transfer Learning

- 冻结大部分 backbone，只训练 linear head。
- 逐步 unfreeze。
- 使用 PEFT，例如 LoRA、adapter、prompt tuning。
- 从相近领域 checkpoint 开始。

### Self-Supervised / Semi-Supervised

- 在无标注领域数据上继续预训练。
- Pseudo-labeling：用教师模型标注高置信样本。
- Consistency regularization：扰动前后预测保持一致。
- Teacher-student / self-training。

伪标签错误会被放大，应使用置信度、类别平衡、人工抽检和迭代更新。

### Active Learning

选择：

- 不确定性高的样本。
- 彼此多样且覆盖新区域的样本。
- 模型之间分歧大的样本。
- 业务风险高的样本。

仅采最不确定样本可能集中在噪声和异常点，通常要把 uncertainty 与 diversity、业务价值结合。

### Weak Supervision 与数据增强

- 规则、词典、远程监督和多个 noisy labeling functions。
- 图像增强、文本改写、回译、mixup 等。
- 强模型生成样本，再由规则、执行器或人工验证。

合成数据不能替代真实数据分布。要检查重复、模式单一、标签泄漏和 teacher bias。

### Few-Shot 场景的评估

- 多个随机 seed 和数据切分。
- 报告均值、方差和置信区间。
- 与简单 baseline 比较。
- 做 learning curve，判断收益来自方法还是增加标注量。

---

## 6. 上线前发现重要特征缺失，但暂时不能重训怎么办？

### 第一步：确认缺失性质

- 是单次请求缺失，还是该特征永久无法在线获取？
- 训练时该特征是否出现过 missing value？
- 缺失是随机的，还是与用户/标签相关？
- 是否能从 feature store、缓存或上游系统可靠回填？

### 可选方案

按可靠程度排序：

1. **恢复或回填特征**：修复在线计算、使用最后有效值，但要限制 TTL。
2. **延迟或拒绝预测**：高风险业务中等待特征，比错误自动决策更安全。
3. **使用预先验证的 fallback model**：该模型不依赖缺失特征。
4. **使用训练时定义的 missing/default 表示**：前提是模型见过这种值并经过评估。
5. **规则或人工 fallback**：降低覆盖率，保住安全和业务底线。
6. **临时调整阈值/校准**：必须基于模拟缺失特征的离线 replay 验证。

### 不能直接做什么？

- 不能随意填 0。0 可能是有业务含义的合法值。
- 不能只用总体均值填充后直接上线，除非训练和评估时使用相同策略。
- 不能假装重要特征永久缺失也无需重训。若缺失是长期状态，临时方案只能争取时间，最终应重新训练并验证。

例子：`过去 7 天消费金额 = 0` 和 `特征没拿到` 含义完全不同。随便填 0 会让模型误以为用户真的没消费。

### 上线前验证

在离线测试流量中强制 mask 该特征，重放完整 serving pipeline，比较：

- 主指标和重要切片。
- 预测分布、校准和阈值通过率。
- Safety / business guardrail。
- Missing rate 增长时的 degradation curve。

未来可在训练时加入 feature dropout、missingness indicator，并长期维护不依赖高风险特征的 fallback model。

---

## 7. 上线后的模型监控

上线后不能只看服务是否 200，还要监控三层：

### 数据层

- Schema、类型、取值范围。
- Missing rate、默认值率、唯一值数量。
- 特征分布、类别分布、数据延迟。
- 训练/线上 feature parity。

### 模型层

- 预测分布、平均分、正例率、entropy。
- Calibration、AUC、PR-AUC、F1 等延迟标签指标。
- 重要切片表现，例如地区、设备、用户类型。

### 系统/业务层

- 延迟、吞吐、错误率、fallback 率。
- CTR、转化、留存、投诉、人工审核量等业务指标。
- Guardrail 指标，例如安全、合规、公平性、成本。

面试一句话：

> 生产监控要同时看 data quality、model behavior、business impact；没有标签时先看输入和预测分布，有标签后再看真实性能。

### 追问：没有线上 label 时，怎么判断模型是不是坏了？

先看 proxy 和无标签信号：

- 输入分布、missing rate、schema 是否异常。
- 预测分布、平均分、positive rate 是否漂移。
- 用户行为 proxy，例如点击、停留、跳出、投诉。
- fallback 率、延迟、错误率是否异常。
- 重要切片是否变化。

等延迟标签回来后，再确认真实 AUC、PR-AUC、校准和业务指标。

---

## 8. A/B Testing in ML

A/B test 用真实线上流量比较两个策略，而不是只看离线测试集。

基本流程：

1. 定义主指标和 guardrail 指标。
2. 随机分流，确保用户不会在 A/B 间频繁切换。
3. 预估样本量和实验时长。
4. 监控数据质量、延迟、错误率和业务指标。
5. 达到统计和业务显著后再扩大流量。

常见坑：

- 提前偷看结果并反复停止实验。
- 分流单位错了，例如同一用户同时进 A/B。
- 新模型改变曝光后，标签分布也随之改变。
- 只看平均值，不看关键切片和负反馈。

### 追问：A/B test 指标涨了，但投诉也涨了怎么办？

不能只看主指标。要看 guardrail：

- 投诉率、隐藏/拉黑/取消关注。
- 延迟和错误率。
- 长期留存。
- 内容多样性、公平性、安全指标。

如果主指标提升来自短期刺激但损害长期体验，应降流量或回滚。

---

## 9. Cold Start

Cold start 指新用户、新物品或新场景缺少历史数据。

### 新用户

- 用注册信息、地理位置、入口来源等上下文特征。
- 先推荐热门、多样化或探索型内容。
- 通过少量 onboarding 问题获取偏好。

### 新物品

- 使用内容特征，例如文本、图片、类目、价格。
- 让新物品获得一定探索流量。
- 用相似物品的 embedding 或规则冷启动。

### 新场景

- 使用迁移学习、相近市场数据、人工规则 baseline。
- 从小流量 shadow/canary 开始收集反馈。

核心权衡：探索能收集数据，但可能牺牲短期体验；利用历史热门更安全，但会压制新内容。

### 追问：Cold Start 怎么评估？

不要只看整体 CTR。要单独看新用户/新物品切片：

- 新用户首日/首周留存。
- 首次正反馈时间。
- 新物品曝光覆盖率。
- 探索流量带来的长期收益。
- 负反馈率。

整体指标可能被老用户老物品掩盖。

---

## 10. Feature Store

Feature store 是统一管理特征定义、计算、存储和服务的系统。

它解决的问题：

- 训练和线上特征不一致。
- 多个团队重复写同一特征。
- 特征版本、血缘、TTL、权限难管理。
- 离线 batch 特征和在线 real-time 特征难对齐。

关键能力：

- Offline store：训练、回放、批量特征。
- Online store：低延迟 serving。
- Point-in-time correctness：训练样本只能拿当时可见的历史特征。
- Feature versioning 和 schema validation。

面试一句话：

> Feature store 的核心价值不是“存特征”，而是保证训练和线上用的是同一套、同一时间语义的特征。

### 追问：Feature Store 和普通数据库有什么区别？

普通数据库只管存取数据。Feature store 还要管：

- 特征定义和版本。
- 离线训练与在线 serving 一致。
- point-in-time correctness。
- TTL、freshness、血缘、权限。

核心不是“存一张表”，而是避免 training-serving skew。

### 追问：Point-in-time correctness 举个例子

预测用户今天是否会流失时，不能使用“未来 7 天是否活跃”这种未来信息。训练样本在某个时间点 $t$，所有特征都必须只来自 $t$ 之前。

常见错误：用全量历史聚合特征，导致训练时偷偷看到未来。

---

## 11. 为什么时间序列不能用普通 K-Fold？

普通 K-fold 会随机打乱数据，容易让未来信息泄漏到训练中。时间序列应使用按时间前滚的验证方式：

```text
Train: [过去]       Val: [未来一段]
Train: [更长过去]   Val: [再未来一段]
```

注意：

- 所有 rolling/window 特征只能用预测时间之前的数据。
- 标准化、填补、特征选择也要按训练窗口 fit。
- 如果同一用户/设备有强相关性，还要结合 group split。

### 追问：Time-series validation 最常见的坑是什么？

把未来信息带进训练：

- 随机 K-fold。
- 标准化用全量数据 fit。
- rolling 特征用了未来窗口。
- target encoding 用了未来 label。

时间序列要按时间切，所有 preprocessing 也只能看训练窗口。

---

## 12. Bias / Fairness 如何检测和缓解？

先定义敏感或重要分组，例如性别、年龄、地区、语言、设备、收入段。然后按组检查：

- 数据覆盖率和标签质量。
- Accuracy、FPR、FNR、Precision、Recall。
- Calibration。
- 通过率、拒绝率、排序曝光率。

常见公平性指标：

- **Demographic parity**：不同组的正向决策率接近。
- **Equal opportunity**：不同组的 TPR 接近。
- **Equalized odds**：不同组的 TPR 和 FPR 都接近。

缓解方法：

- 补充低覆盖组数据，改善标注质量。
- 调整采样、重加权或阈值。
- 删除或约束明显不合规特征，但要注意 proxy feature。
- 上线后持续按组监控。

公平性没有唯一答案，必须结合业务、法律和错误代价说明取舍。

### 追问：Fairness 指标冲突怎么办？

不同公平性定义可能无法同时满足。例如 demographic parity 要求不同组通过率接近，equalized odds 要求错误率接近。实际要根据业务和法律风险选择主约束，并报告 trade-off。

[返回目录](README.md)