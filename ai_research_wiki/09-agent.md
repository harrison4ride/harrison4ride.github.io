# 07. Agent

[返回目录](README.md)

## 1. 如何定义 LLM Agent？

### 30 秒回答

LLM Agent 是以语言模型为策略或推理核心，能够维护状态、制定计划、调用工具、观察环境反馈并循环行动，直到完成目标或触发终止条件的系统。

区别于普通聊天模型的关键不是“会调用一次 API”，而是形成闭环：

$$
\text{Observation}
\rightarrow \text{Reason/Plan}
\rightarrow \text{Action}
\rightarrow \text{Environment Feedback}
\rightarrow \text{State Update}
$$

### 核心组件

- **Model/Policy**：理解任务、推理、选择动作。
- **Planning**：任务分解、顺序安排、重规划。
- **Memory/State**：保存对话、工作区、历史行动和长期知识。
- **Tools**：搜索、代码执行、数据库、浏览器、业务 API。
- **Environment**：工具返回结果或可交互世界。
- **Controller/Orchestrator**：执行循环、超时、重试、并发和终止。
- **Guardrails**：权限、确认、沙箱、策略检查和审计。
- **Evaluator**：判断完成度、错误和成本。

---

## 1.1 Agent Harness 是什么？

### 30 秒回答

Agent harness 是包在 LLM 外面的执行系统。模型只负责“想”和“选动作”，harness 负责把任务真正跑起来：

- 给模型准备 prompt、工具说明和上下文。
- 维护状态、文件、记忆和中间产物。
- 执行工具调用、命令、浏览器或 API。
- 管理权限、沙箱、超时、重试和预算。
- 记录 trace，方便复盘和评估。
- 判断任务是否完成，必要时让模型继续、回退或请求人工确认。

一句话：

> **LLM 是大脑，harness 是操作系统 + 工作流控制器 + 安全壳。**

### 为什么 Harness 很重要？

同一个模型，换不同 harness，效果可能差很多。原因是 agent 的表现不只取决于模型能力，还取决于：

- 工具接口是否好用。
- 文件编辑和命令执行是否可靠。
- 观察结果是否清晰。
- 错误是否能恢复。
- 是否能安全地限制危险动作。
- 是否能保存和重放完整轨迹。

比如 coding agent 不是只把 repo 塞给 LLM。好的 harness 会提供：

- 搜索、读文件、编辑、运行测试的工具。
- 隔离环境或 sandbox。
- diff 和测试结果反馈。
- 失败重试和回滚。
- 最终 patch 验证。

SWE-agent 的核心观点之一就是 Agent-Computer Interface 设计会显著影响软件工程任务表现；OpenHands 这类平台也强调 sandbox、runtime、工具、评估和人机交互，而不只是换一个更强模型。

### Harness 和 Agent Framework 的区别

- **Framework**：开发框架，例如 LangGraph、OpenHands、AutoGen、LlamaIndex。
- **Harness**：你为某个任务实际搭出来的执行壳，包括 prompt、工具、状态机、权限、评估和恢复策略。

可以用框架搭 harness，但二者不是一回事。

### 面试怎么说？

> 我理解 agent harness 是模型外面的运行时系统，负责工具调用、状态管理、安全权限、沙箱、trace、重试和评估。现在很多 Agent 的差距不只是模型差距，而是 harness 差距。尤其是 coding、browser、computer-use 任务，工具接口和执行环境设计会直接影响成功率。

---

## 1.2 最近常见的 Agent Loop 设计

### 基础 Agent Loop

最基础的 loop 是：

$$
\text{Observe}
\rightarrow \text{Think/Plan}
\rightarrow \text{Act}
\rightarrow \text{Observe}
\rightarrow \cdots
$$

每一轮模型读取当前状态，决定下一步工具调用，然后根据 observation 继续。ReAct 就是这个范式的代表。

### 现在为什么不只讲简单 Loop？

简单 loop 好用，但有几个问题：

- 上下文越来越长，历史越来越乱。
- 依赖关系隐含在对话里，难 debug。
- 出错后容易无限重试。
- 很难判断哪些中间产物应该保留、哪些应该重算。
- 多工具、多子任务时，单 loop 容易变成黑盒。

所以近期更常见的方向是：**从隐式 loop 走向显式 state machine / graph / harness**。

### 常见 Loop 形态

#### 1. ReAct Loop

```text
Thought -> Action -> Observation -> Thought -> ...
```

适合短任务、工具调用和快速探索。缺点是容易循环、上下文变长、状态不够结构化。

#### 2. Plan-Execute-Verify Loop

```text
Plan -> Execute step -> Verify -> Replan or Finish
```

适合 coding、RAG、多步骤业务流程。关键是每步要有 verifier，例如测试、schema check、retrieval score、规则检查。

#### 3. Reflection / Repair Loop

```text
Run -> Fail -> Diagnose -> Patch -> Retry
```

适合代码修复、数据处理、网页操作。注意：reflection 只有在有真实失败信号时才有价值；没有测试或环境反馈时，反思可能只是自我合理化。

#### 4. Supervisor + Worker Loop

```text
Supervisor splits task -> Workers execute -> Supervisor merges/verifies
```

适合任务可拆分、可并行的场景。风险是通信成本、重复劳动和 supervisor 瓶颈。

#### 5. Graph / State-Machine Loop

把流程写成显式节点：

```text
Retrieve -> Draft -> Check -> Revise -> Final
```

每个节点有明确输入、输出、失败处理和终止条件。LangGraph 这类工具适合这种模式。相比自由 loop，它更容易测试、重放和上线。

### 近期趋势

- **Harness engineering**：优化模型外部的工具、状态、权限、sandbox 和评估。
- **Explicit graph over implicit chat history**：把控制流从对话上下文里拿出来，写成显式图或状态机。
- **Verifier-driven loop**：用测试、执行结果、规则、judge 或环境状态决定是否继续。
- **Human-in-the-loop**：高风险动作前暂停确认。
- **Parallel sub-agents**：并行探索候选方案，但必须有去重、合并和最终验证。
- **Long-running/background agents**：支持任务挂起、恢复、定期执行和持久化状态，但更需要权限边界和审计。

### 面试怎么说？

> 早期 Agent 常被描述成一个 ReAct loop：观察、思考、行动、再观察。最近更工程化的做法是把 loop 放进 harness 里，显式管理状态、工具、权限、验证和恢复。对生产系统来说，关键不是让模型无限循环，而是限制循环、有 verifier、有终止条件，并把流程做成可重放、可调试的 graph 或 state machine。

---

## 2. ReAct 框架

ReAct 将 reasoning trace 和 action 交错：

1. **Thought**：基于当前观察分析下一步。
2. **Action**：按工具协议执行动作。
3. **Observation**：环境返回结果。
4. 重复直到给出 Final Answer。

### 为什么有效？

- 纯 CoT 只能基于模型内部知识，容易幻觉。
- 纯 action policy 缺少显式任务分解和上下文整合。
- ReAct 让推理决定何时查证，观察又能纠正后续推理。

### 失败模式

- 无限循环或重复搜索。
- 因早期错误观察形成错误计划。
- 把工具输出中的恶意文本当作指令，即 prompt injection。
- Thought 很长但没有提高决策质量。
- 工具参数格式错误或调用不存在的工具。

### 工程改进

- 限制最大步骤、时间和预算。
- 用结构化 action schema，不解析自由文本。
- 对观察做 provenance 标记和不可信内容隔离。
- 保存状态机和 error taxonomy，针对可恢复错误重试。
- 使用 verifier 判断完成，而不是让同一模型仅凭自信终止。

---

## 3. 如何赋予 Agent 规划能力？

### Chain-of-Thought（CoT）

线性分解和推理。成本低，但错误会沿单路径传播。

### Plan-and-Execute

先生成高层计划，再逐步执行；执行中可重规划。适合长任务，但计划可能过早固定，环境变化时需显式修订。

### Tree of Thoughts（ToT）

在每一步生成多个候选 thought，评估并搜索，如 BFS/DFS/beam search。提高探索能力，但调用成本高，评估器质量决定结果。

### Graph of Thoughts（GoT）

允许 thought 间合并、回边和复用，比树更适合多来源综合、迭代修订，但控制和去重更复杂。

### Self-Consistency

采样多条推理链，通过多数投票或 verifier 选择。适合有确定答案的任务，成本随采样数增加。

### Reflection / Reflexion

执行失败后总结原因，把反思写入短期或长期记忆，再重试。有效前提是有可靠失败信号；否则可能只是产生合理化解释。

### Search / Classical Planner

让 LLM 生成动作或启发函数，使用 A*、MCTS、约束求解器或 PDDL planner 搜索。在动作空间明确、可模拟时比自由 CoT 更可控。

### 选型原则

- 短、低风险任务：ReAct/CoT。
- 长任务：plan-and-execute + checkpoint。
- 可验证且分支有限：ToT、beam、MCTS。
- 强约束调度：经典 planner/solver。
- 高成本工具：先规划并估算预算，再执行。

---

## 4. 短期记忆和长期记忆

### 短期记忆

保存当前任务所需的工作状态：

- 对话上下文。
- 当前计划、已完成步骤、工具输出。
- Scratchpad、代码 diff、临时变量。

实现可以是 context window、结构化 state object、数据库中的任务 checkpoint。长对话要做摘要，但摘要会丢信息，因此关键事实应结构化保存而不是只做自然语言压缩。

### 长期记忆

- **Semantic memory**：事实、偏好、领域知识，常用向量库/搜索索引。
- **Episodic memory**：过去任务、行动和结果。
- **Procedural memory**：成功工作流、工具使用规则和技能。

### 写入策略比存储介质更重要

需要回答：

- 什么信息值得写？
- 如何去重、更新和遗忘？
- 谁能读取，权限和隐私如何控制？
- 何时召回，如何避免无关记忆污染当前决策？

可用 salience、novelty、用户明确确认、任务结果和时间衰减决定写入。召回时结合语义相似度、时间、实体和任务过滤。

### 常见外部技术

PostgreSQL/Redis 保存结构化状态；向量数据库做语义召回；知识图谱保存实体关系；对象存储保存大文件；事件日志支持审计和重放。

---

## 5. LLM 如何学会 Tool Use / Function Calling？

### 数据格式

模型看到：

- 工具名称和自然语言描述。
- 参数 JSON Schema。
- 用户请求。
- 正确的 tool call。
- 工具结果。
- 基于结果生成的最终回答。

通过 SFT 学习“何时调用、选哪个工具、参数如何填”。还可用执行成功率或结果正确性做 RL。

### 推理流程

1. 应用把工具 schema 放入模型上下文。
2. 模型输出结构化工具调用，而不是直接执行。
3. Orchestrator 校验权限和参数。
4. 外部系统执行并返回 observation。
5. 模型读取结果，继续调用或回答。

### 关键风险

- Schema 合法不等于语义正确。
- 工具描述过长或相似会导致选择混淆。
- 外部内容可能包含 prompt injection。
- 写操作、支付、发送消息需要用户确认、幂等键和最小权限。
- 模型不能自行决定自己是否有权限。

### 评价指标

Tool selection accuracy、参数正确率、执行成功率、任务成功率、无效调用数、重试次数、延迟、成本，以及未授权操作率。

---

## 6. LangChain 与 LlamaIndex

### LangChain / LangGraph

重点是 LLM 应用编排：模型、工具、prompt、状态、Agent loop。LangGraph 更适合显式状态图、分支、循环、持久化和 human-in-the-loop。

### LlamaIndex

起点更偏数据和知识：数据连接器、文档解析、索引、retriever、query engine、RAG workflow。也提供 Agent 能力，但核心优势常在 knowledge-centric 应用。

### 如何选？

- 复杂有状态工作流、多工具编排：LangGraph。
- 文档/RAG、数据连接和索引实验：LlamaIndex。
- 简单生产服务：直接使用模型 SDK + 自己的状态机可能更透明。

面试中最好说明：框架不是能力来源。关键是可观测性、可测试性、状态持久化、错误恢复、供应商锁定和团队维护成本。

---

[返回目录](README.md)