# 08. RAG 与搜索

[返回目录](README.md)

## 1. RAG 的工作原理

Retrieval-Augmented Generation 在回答前从外部知识源检索相关证据，把证据连同问题交给 LLM 生成答案：

$$
q\rightarrow \operatorname{Retrieve}(q)
\rightarrow \{d_1,\ldots,d_k\}
\rightarrow \operatorname{Generate}(q,d_{1:k})
$$

### 相比直接微调

RAG 主要解决知识更新、私有知识、来源追溯和长尾事实问题。

**优势**

- 更新知识库即可生效，不必重新训练。
- 可展示引用和证据。
- 私有数据不必写入模型参数。
- 对事实型任务可降低幻觉。

**局限**

- 检索不到或检索错，生成端无法补救。
- 上下文增加延迟、token 成本和 prompt injection 面。
- LLM 仍可能忽略、曲解或混合证据。

### RAG 与微调不是互斥

- RAG 提供知识。
- 微调改变行为、格式、领域语言和工具使用方式。
- 常见组合是：微调模型学会使用证据，RAG 动态提供事实。

---

## 2. 完整 RAG 流水线

### 离线索引

1. 数据源连接：网页、PDF、数据库、工单、代码。
2. 解析与清洗：OCR、表格、标题、页码、权限元数据。
3. 文档去重和版本管理。
4. Chunking。
5. 生成 embedding。
6. 写入向量索引、倒排索引或知识图谱。

### 在线查询

1. Query 理解、分类或改写。
2. 召回：dense、sparse、metadata filter、graph traversal。
3. 融合与去重。
4. Rerank。
5. Context selection/compression/order。
6. 基于证据生成。
7. 引用绑定、事实校验和拒答。
8. 记录反馈用于评估迭代。

每个阶段都要单独评估，否则只看最终答案无法定位是 retrieval 还是 generation 出错。

---

## 3. 如何选择切块大小和重叠？

### 权衡

**块太小**

- 语义不完整，标题、定义和论据分离。
- 召回更精确但缺上下文。
- chunk 数和索引成本增加。

**块太大**

- embedding 混合多个主题，召回精度下降。
- 占用更多 context。
- 相关句子被无关内容稀释。

**Overlap**

可防止答案跨边界被切断，但过大导致索引膨胀、重复召回和上下文浪费。

### 选择方法

从文档结构出发，不应只按固定字符数：

- FAQ：每个问答一个 chunk。
- 技术文档：标题层级 + 段落，保留 breadcrumb。
- 论文：章节、段落、图表说明。
- 代码：函数、类、模块和调用关系。
- 法律：条款和引用关系。

先建立真实 query-answer-evidence 集，在多个 chunk size、overlap 下比较 Recall@K、MRR、最终 faithfulness、延迟和 token 成本。常见起点可以是 200--500 tokens、10%--20% overlap，但不是通用最优值。

### Parent-Child Retrieval

用小 chunk 做精确检索，命中后返回更大的 parent chunk 给生成模型，兼顾召回精度和上下文完整性。

---

## 4. 如何选择 Embedding 模型？

### 考虑因素

- 语言和领域：中文、多语言、代码、法律、医学。
- Query/document 是否使用不同 instruction 或 prefix。
- 最大输入长度。
- 向量维度、存储和检索成本。
- 延迟、吞吐、部署许可。
- 是否支持 fine-tuning。

### 离线指标

- Recall@K：相关文档是否进入前 K。
- Precision@K：前 K 中相关比例。
- MRR：第一个相关结果排名的倒数。
- MAP：多个相关结果时的平均精度。
- NDCG：考虑分级相关性和排名位置。

### 不要只看公开榜单

MTEB 等 benchmark 可做初筛，但必须用自己的 query、文档、语言、长度和 hard negatives 评估。还要联合测试 chunk 策略与 reranker，因为 embedding 单项最优不保证端到端最优。

---

## 5. 提升检索质量的方法

### Hybrid Search

结合 BM25 和 dense retrieval：

- Sparse 对精确术语、编号、人名、错误码强。
- Dense 对语义改写和同义表达强。
- 可用 Reciprocal Rank Fusion（RRF）合并排名：

$$
\operatorname{RRF}(d)=
\sum_j\frac{1}{k+\operatorname{rank}_j(d)}
$$

### Reranking

先用便宜 retriever 召回几十到几百个候选，再用 cross-encoder 或 LLM 对 query-document 联合打分，提升前 K 精度。代价是延迟和计算。

### Query Transformation

- Query rewrite：补全指代和上下文。
- Multi-query：生成多个角度的查询，合并召回。
- HyDE：先生成假想答案/文档，再对其 embedding 检索。
- Query decomposition：多跳问题拆成子问题。

### Metadata Filtering

按租户、权限、时间、产品版本、语言过滤。权限过滤必须在检索层生效，不能只让 LLM“不要引用”。

### Contextual Retrieval

给 chunk 加入文档标题、章节和简短上下文后再 embedding，避免孤立片段语义不足。

### Fine-tuning

用领域 query-positive-hard negative 对训练 bi-encoder；hard negatives 应“看似相关但不能回答”，比随机负例更有价值。

---

## 6. Lost in the Middle

### 现象

当相关证据位于长上下文中部时，模型利用率常低于证据在开头或结尾的情况。即便上下文窗口足够长，也不代表模型对所有位置同等有效。

### 缓解

- 提升检索和 rerank，只放真正相关内容。
- 把最关键证据放在开头和靠近问题的位置。
- 去重、压缩和分组上下文。
- 分段回答或 map-reduce，而非一次塞入全部内容。
- 让模型先提取每段证据，再综合。
- 对长上下文模型做真实位置敏感评估。

更根本的原则是：不要用“更长 context”替代检索质量。

---

## 7. 如何全面评估 RAG？

### 检索阶段

- Recall@K、Precision@K、MRR、MAP、NDCG。
- Answer-containing recall：召回内容是否包含答案。
- 权限过滤正确率。
- 按 query 类型、领域、时间和难度切片。

### 生成阶段

- **Correctness**：答案是否正确。
- **Faithfulness / Groundedness**：陈述是否能由检索证据支持。
- **Answer relevance**：是否回答问题。
- **Citation precision/recall**：引用是否支持对应 claim、是否漏引。
- **Completeness**：是否覆盖问题要求。
- 拒答准确率和 calibration。

### 系统阶段

- 端到端成功率。
- P50/P95 延迟、吞吐和成本。
- 新鲜度、索引延迟。
- 用户追问率、点击引用率和人工纠正率。

### 评估集构建

应包含真实用户 query、答案、支持证据和不可回答问题。避免只用 LLM 合成简单问题；合成数据可扩规模，但要有人类校准和真实日志覆盖。

---

## 8. 什么时候用知识图谱/图数据库？

适合：

- 问题需要多跳关系，如“某药物影响的通路关联哪些疾病？”
- 实体、关系和约束清晰。
- 需要可解释路径和精确过滤。
- 关系更新、实体消歧比文本相似度更重要。
- 欺诈、供应链、组织关系、推荐等图结构场景。

不适合仅因为“图听起来更高级”就替换向量库。开放文本知识抽取成本高，图谱会有遗漏、错误和维护负担。

常见混合方案：

1. 从 query 识别实体。
2. 图遍历找候选实体/关系。
3. 回到文本库检索支持证据。
4. LLM 基于结构化路径和原文生成。

---

## 9. 更复杂的 RAG 范式

### Iterative / Multi-Hop RAG

根据第一轮证据生成下一轮 query，逐跳检索，适合跨文档问题。风险是早期错误导致 query drift。

### Self-RAG

模型学习决定是否检索，并生成 reflection token 评估证据相关性和回答支持度。

### Corrective RAG（CRAG）

先评估检索结果质量；若不足，改写查询、扩大搜索或使用 web source，再生成。

### Adaptive RAG

按问题复杂度选择不检索、单步检索或多步检索，减少无必要成本。

### Agentic RAG

Agent 可选择不同数据源、拆问题、反复检索、验证和修订。能力更强，但延迟、不可预测性和评估难度更高。

---

## 10. RAG 的实际部署挑战

- 文档解析错误，特别是 PDF、表格、扫描件。
- 数据更新、删除和版本一致性。
- 多租户权限与敏感数据泄漏。
- Query 分布变化和新术语。
- Embedding 模型升级导致全量重建索引。
- 检索和生成延迟叠加。
- 引用看似存在但不支持 claim。
- 外部文档 prompt injection。
- 评估集与线上问题脱节。
- 成本：embedding、存储、rerank、长 context。

需要完整 observability：query rewrite、召回列表、分数、rerank、最终 context、引用和模型输出都可追踪。

---

## 11. 搜索系统和 RAG 的区别

### 搜索

目标通常是返回排序后的文档、网页或商品，由用户阅读和判断。核心关注 relevance、ranking、点击和覆盖。

### RAG

检索只是中间步骤，最终由生成模型综合答案。除 relevance 外，还要关注证据使用、事实忠实、引用和生成幻觉。

### 联系

高质量 RAG 依赖成熟搜索技术：倒排索引、BM25、Learning to Rank、query understanding、freshness、权限和反作弊。RAG 不是用向量数据库替代搜索，而是在检索之上增加生成和交互。

---

## 12. 开源 RAG 框架和选型

### 常见框架

- **RAGFlow**：偏完整文档理解、知识库和可视化 RAG 流程，适合快速构建文档型应用。
- **LlamaIndex**：数据连接、索引、retriever 和 query engine 组件丰富。
- **Haystack**：pipeline 化检索与生成，适合可组合后端服务。
- **LangChain/LangGraph**：适合把 RAG 放进复杂 Agent 工作流。
- **LightRAG / GraphRAG 类方案**：强调实体关系、全局/局部图检索。

### 选型问题

- 文档类型和解析质量。
- 是否需要 UI、多租户和权限。
- 自定义 retriever/reranker 的自由度。
- 增量索引和版本。
- 可观测、评估和部署能力。
- 团队是否更需要开箱即用，还是研究可控性。

对核心检索质量敏感的研究项目，应能单独替换 parser、chunker、embedding、retriever 和 reranker，而不是被框架默认值锁死。
