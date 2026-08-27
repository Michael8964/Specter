# Specter 项目计划 v8 —— 精读五篇自适应控制论文原文后的引用勘误版

v7 写完之后,用户指出"想让计划文档达到参考文档的密度,不能靠'再写详细点'这种单一指令,而要分阶段做不同性质的动作——先读完真实材料(比如论文的Limitations章节)、再手算公式、再写故事化walkthrough、再自审"，并授权("好，你自己评估、自己review、自己干吧")对5篇自适应控制论文(GammaTune/SpecKV/AdaEDL/BanditSpec/Nightjar)做一次**回到原文而非依赖摘要转述**的精读，逐条核对v6/v7里引用的具体数字和归因是否准确。

结果：**发现了一处需要更正的错误引用**，以及若干可以把"转述性引用"升级为"精确引用"的地方。v8 只做这一件事——不引入新的架构/技术决策，v7 的所有设计决策原样保留。

**发现的错误**：v6/v7 反复引用的"SpecKV发现最优γ随目标模型量化程度偏移最高达41.2%"是**误归因**。读完SpecKV全文(arXiv 2605.02888)后确认：41.2%这个数字出现在SpecKV论文里，但那是SpecKV**引用另一篇论文**(Smurfs, Chen/Liang/Wang 2024, arXiv 2405.05955)在讨论"γ会随不同模型/batch size/数据集变化"这个*一般性*论点时提到的数字，不是SpecKV自己关于*压缩程度*的实验结果。SpecKV自己的真实实验结果(其Table 1)是：最优γ在FP16下是2，INT8下是8——偏移量级是4倍(300%)，压缩方式是BitsAndBytes的FP16/INT8/NF4，不是AWQ。SpecKV的真正headline结果是自己的MLP-16控制器让"每步期望token数"比固定γ=4提升56.0%，控制器开销0.34ms(不到一步耗时的0.5%)。这处错误已经在下面所有引用到的地方(指标体系表、P5.2、坑10、附录E)更正，并把41.2%正确地重新归位给Smurfs。这个错误恰好验证了用户坚持"回到原始材料"这一步骤的必要性——如果只读转述/摘要，这处误归因不会被发现。

**顺带发现的第二处措辞问题**：坑12把BanditSpec描述成"理论严谨但实现成本高(regret bound分析)"，暗示不复现的原因是"太复杂"。读完全文后这个说法不够准确——BanditSpec论文自己在附录B.2明确说"UCBSpec是最简单的一类UCB算法之一，只需要维护经验均值和UCB置信半径"(相对Thompson Sampling/KL-UCB更复杂的替代方案，作者是刻意选择了这个更简单的版本)。**复杂的是论文里17页的regret bound证明过程，不是UCBSpec这个算法本身**——如果真要实现，核心逻辑大概10行代码就能写完。v8把坑12的措辞改成更诚实的真实理由：不复现的原因不是"太复杂"，而是(a) BanditSpec的K-armed bandit框架不把batch size作为上下文特征，这是Nightjar论文和BanditSpec自己的"未来工作"章节都承认的局限；(b) BanditSpec的理论分析完全没有建模"从γ=0切换回γ>0"的KV cache重建开销，这正是P5.3熔断器要弥补的地方——GammaTune的EMA方案已经覆盖了"自适应步长"这个核心目标，再引入一套独立的bandit框架和regret-bound分析框架，对项目目标而言是不成比例的额外抽象层，不是"做不到"。

另外补充了几处从精读中获得、但之前只有转述、现在可以给出精确数字/原文依据的引用（AdaEDL论文里另一组独立实验也证明"死亡区间"现象、Nightjar论文实测的KV cache切换开销具体量级）。

---

## 0. 版本信息

| 版本 | 内容 |
|---|---|
| v1 | 三支柱构想，基于摘要级调研 |
| v2 | 精读论文原文后修正三处理论风险 |
| v3 | 生产复盘后新增支柱4（batch size 交叉点） |
| v4 | 发现平台兼容性硬约束，新增支柱5（自适应控制器），补充12个具体已知坑 |
| v5 | 按工程设计文档骨架重组，补齐目标/非目标、考虑过的其他方案、依赖假设、里程碑判据、悬而未决问题 |
| v6 | 对标参考文档前18页的密度：三层指标体系、机制级拆分（P1.0-P5.4）、端到端走读、叙事收尾、三份附录 |
| v7 | 读完参考文档全部28页后补四处颗粒度缺口：走读加入失败/恢复场景、公式配数值算例、真实文献列表内嵌（而非外链）、附录C加可执行的诊断分析代码；新增"研究诚信护栏"一节 |
| **v8**（本版本） | 精读5篇自适应控制论文原文(而非依赖摘要/转述)，发现并更正一处错误引用(SpecKV的"41.2%"实为误归因，真实数字是2→8/4倍)，重新措辞一处不准确的复杂度判断(BanditSpec不是"太复杂"而是"不建模batch/切换开销")，补充若干可精确溯源的新引用 |

---

## 1. 摘要

Specter 是一个面向 Agent 场景的本地推理加速引擎：手写实现投机解码（含严格正确性证明）和 AWQ 风格量化，用一套系统原理（内存带宽优化 vs 计算瓶颈切换点）统一两者的失效边界，再加一个训练-free 的自适应投机步长控制器。目标硬件：个人 24GB 统一内存 Mac（开发/调试，$0）+ 短期云端 GPU 租用（规模化验证，$30-50）。当前状态：计划已经过两轮论文精读（含一轮针对5篇自适应控制论文的原文核对）、一轮生产实践调研、一轮针对性踩坑调研，尚未开始任何代码实现。

---

## 2. 背景与问题陈述

### 2.1 为什么现在做这件事有意义（不只是"学新技术"）

一个残酷的招聘现状,恰好也是 Time x Mart 那份文档里引用过的数据(Gartner 预测到 2028 年全球四分之一的候选人档案将是伪造的; LinkedIn 每分钟吞下约11,000份几乎一模一样的申请):当 AI 能替任何人生成一份看起来专业的项目描述时,简历上的一行 bullet 本身已经不构成信号。真正稀缺的是**可验证的技术判断力痕迹**——你在实现之前有没有读懂边界条件、有没有踩过真实的坑、有没有诚实地说清楚"我为什么没选那个更优的方案"。

这正是 Specter 的定位:不是"我实现了投机解码"这种可以被任何人复制粘贴的陈述,而是一份**留下判断痕迹的工程记录**——12个具体的、有来源的已知坑,7处"考虑过但拒绝"的替代方案并写明理由,一个从平台兼容性调研中反推出来的架构决定。这些东西 AI 可以帮你写代码,但编不出"我调研之后发现 vLLM 在 Mac 上跑不动,所以把方案拆成两个阶段"这种只有真正做过调研的人才会有的具体转折。

### 2.2 技术现状与缺口

投机解码和量化在 2025-2026 已经是主流推理框架的标配(vLLM/SGLang 一个 flag 就能开)——这意味着"会调库"本身已经没有技术含量。缺口在于:几乎没有人系统性地验证过——(a) 手写实现能否达到和调库一致的正确性保证,(b) 这些技巧在什么条件下失效(batch size、压缩程度),(c) 主流生产库还没内置的"自适应控制"能做到什么程度。这个项目要填的就是这个"调库拿不到"的部分。

---

## 3. 目标与非目标

### 目标
1. 手写实现投机解码核心采样算法,贪心模式逐 token 精确验证正确性,采样模式统计层面验证分布等价。
2. 手写实现 AWQ 风格激活感知量化校准,诚实区分"自己写的算法"和"复用的现成 kernel"。
3. 用统一系统原理(内存带宽 vs 计算瓶颈)解释投机解码和量化各自的失效边界,实测出具体交叉点。
4. 实现训练-free 自适应投机步长控制器,验证其在压缩模型和生产基线(HF Transformers 内置动态投机)面前的表现。
5. 用 AgentBench 方法论改编的 agent 工具调用评测集,验证结构化输出场景下投机解码的优势。

### 非目标(显式声明,防止范围蔓延)
1. 不重新实现 GPTQ——太复杂,用 LLM Compressor 做对比基线。
2. 不做完整 AgentBench 八个环境——只改编 OS/bash 子环境。
3. 不重新实现 BanditSpec 的 bandit 理论方法或 AdaEDL 的熵基提前停止——只作文献对比引用（BanditSpec 的 UCBSpec 算法本身并不复杂，不复现的理由见坑12的v8更正版：它不建模 batch size 和切换开销，而不是"太复杂"）。
4. 不训练专用 speculative head(EAGLE/Medusa 那种)——没有多卡训练资源,经典两模型方案是预算约束下的合理取舍。
5. 不追求在 Mac 上跑出真实速度数字——Mac 阶段只产出正确性/压缩率数字。
6. 不做多租户/高并发服务化——关注单用户/低 batch 的本地 agent 场景。

---

## 4. 指导原则(Tenets)

1. **能在免费环境验证的,绝不烧云端预算**——决定 Mac/云端两阶段分工。
2. **诚实边界优于假装SOTA**——每处方案选择都要写清楚"生产界最优解是什么、我们为什么没选它"，理由必须是真实理由，不是听起来更体面的理由（v8的坑12更正是这条原则的一次自我检验）。
3. **正确性验证要有可复现的数学/统计依据**——贪心逐token精确比对、采样统计检验+任务指标双重验证。
4. **每个技巧不仅要证明"有效",还要测出"什么时候失效"**——batch size交叉点实验和量化-步长耦合实验的存在理由。
5. **基线出来之前不编造目标数字**(借用 Time x Mart 文档的原则)——第5节的所有阈值都标注"预期区间"而非"承诺值",实际数字以阶段0-6跑出来的结果为准。
6. **测量本身也要接受怀疑**——一个"符合预期"的实验结果不会自动免检;必须先证明测量工具本身是可靠的(测试的测试),再采信结果。
7. **引用别人的数字之前,先确认自己真的读对了(v8新增)**——转述/摘要会丢失或扭曲归因关系(比如把论文A引用论文B的数字误记成论文A自己的结果)；任何要写进计划或最终报告的具体数字，引用前必须能标注清楚"这句话在原文哪一节、是谁的实验结果、不是谁引用谁的背景数字"。

---

## 5. 指标体系(北极星 / 伴随 / 护栏)

> 沿用 Time x Mart 文档 §6 的三层指标框架。**以下数字均来自已读论文的原始实验结果,是"这个方法在别人的实验里做到过什么量级"，不是 Specter 自己的实测承诺**——Specter 自己跑出来的数字以最终 README 为准,可能高于也可能低于这些参考区间,这正是实验要回答的问题。

### 北极星指标
**正确性优先于速度**:贪心模式下投机解码输出与目标模型直接推理的 token 级一致率——目标 100%(允许可忽略的浮点误差),这是唯一不允许"这次没做到但下次再说"的指标。

### 伴随指标(每个支柱一个,参考区间来自文献)

| 指标 | 参考区间(来自文献,非承诺值) | 来源 |
|---|---|---|
| 投机解码接受率 α | ≥0.65 才有净加速(低于则收益递减) | mlx-lm 生产实践阈值 |
| 投机解码整体加速比 | 2-3x(生产环境实测) | IBM/PyTorch 官方生产博客 |
| 4-bit量化显存降低 | ~70%,perplexity涨幅<5%(同分布校准) | AWQ论文 |
| 跨分布校准 perplexity 涨幅差异 | AWQ +0.5-0.6 vs GPTQ +2.3-4.9 | AWQ论文实验 |
| GammaTune 风格自适应控制器 vs 固定γ | 15%±5% | GammaTune论文 SpecBench结果 |
| 量化程度对最优γ的偏移幅度 | 最高4倍(FP16下γ=2 → INT8下γ=8，BitsAndBytes压缩，非AWQ) | SpecKV论文 Table 1（v8更正：此前误引"41.2%"，该数字实为SpecKV转述Smurfs论文[arXiv 2405.05955]关于γ随模型/batch/数据集变化的一般性论点，非SpecKV自己的压缩实验结果） |
| 高负载下投机解码相对无投机的倒退幅度 | 最高30.25% | Nightjar论文生产实测(RTX 4090, DeepSeek-R1-Distill-Qwen-7B, ShareGPT) |

### 护栏指标(不能突破的底线)
- 采样模式下游任务指标差距(HumanEval pass@1 / ROUGE)必须 <2 个点,否则视为"分布等价"验证失败,需要回头检查采样实现而不是接受结果。
- 云端预算硬顶 $50,超支即停止规模化验证,用已有结果收尾,不追加开支。
- 任何"自适应控制器让情况变得更差"的场景(GammaTune论文承认的边界)必须被记录进报告,不能因为不好看就不提——诚实边界本身是护栏。

---

## 6. 现有方案与文献综述

条件性结论(完整分类文献列表见**附录E**,不再用外链指针代替内容): 投机解码/量化已是主流框架标配, 真正区分度在手写实现+正确性验证+边界刻画; IBM/PyTorch生产复盘证实代码/结构化输出场景效果最好, batch>64开始吞吐下降; AWQ相对GPTQ对校准集分布更鲁棒; 2024-2026自适应投机控制是活跃前沿方向, 且动态γ调整已是HF Transformers默认行为(4.45.0起)。

---

## 7. 方案与架构(机制级拆分,对标 R0-R17 的颗粒度)

### 架构总览

```
Mac 开发阶段（阶段 0-4, $0）：HF Transformers (PyTorch, MPS backend)
云端 GPU 阶段（阶段 5-6, $30-50）：vLLM 对比基线 + LLM Compressor
```

### 支柱1：手写投机解码 + 正确性验证

**P1.0 — 前置存活性 gate（S，半天）**
断言 `draft_tokenizer.get_vocab() == target_tokenizer.get_vocab()`；测 α，两级门槛：<0.4 换模型对；0.4-0.65 降低预期继续；≥0.65 正常推进。用 mlx-lm 跑同一模型对交叉验证 α 数量级。

**P1.1 — Rejection Sampling 核心算法（M，2-3天）**
标准算法：草稿模型自回归生成 γ 个候选 token，目标模型单次前向对全部候选打分；接受判据 $p_{TM}(x) \geq p_{DM}(x)$ 时接受，否则以 $1 - p_{TM}(x)/p_{DM}(x)$ 概率拒绝，从调整分布 $p'_{TM}(x) = \text{norm}(\max(0, p_{TM}(x) - p_{DM}(x)))$ 重采样。**关键实现检查点**：bonus token 必须来自目标模型分布，不是草稿模型（坑2）。数值算例见**附录A.1**。

**P1.2 — 贪心模式验证器（S，1天）**
逐 token 严格比对投机解码输出 vs 目标模型直接推理输出，要求 100% 一致（或仅浮点误差）。

**P1.3 — 采样模式验证器（M，2天）**
统计层面验证实测 α 与理论公式 $\alpha = E[\min(p,q)]$ 吻合；下游任务指标 parity(HumanEval pass@1 / ROUGE，差距<2点)。

**P1.4 — γ 扫描（S，1天）**
γ ∈ {1,3,5,7,10}，记录接受长度分布（预期形状参考 AdaEDL 论文 Figure 1 / Appendix Fig 7c：Dolly-15k创作数据集,目标模型Llama2-7B,草稿模型是对齐微调过的115M模型,不同 DL 下方差递增，DL=3→std≈1.2，DL=7→std≈1.92，DL=16→std≈2.35，这是"为什么需要自适应γ"的实证基础，支柱5会复现类似分布）。

### 支柱2：AWQ 量化

**P2.0 — 激活统计采集（S，1天）**
在校准集上跑前向，收集每层激活值的逐通道统计量。

**P2.1 — 逐通道缩放校准算法（M，2天）**
AWQ 核心思想：找到一组逐通道缩放因子，使量化后误差最小化，同时保护"显著性权重通道"（由激活值幅度决定）不被粗暴量化。直觉性数值算例见**附录A.3**。

**P2.2 — 跨分布鲁棒性实验（S，1天，坑5）**
两个分布明显不同的校准集（代码语料 vs 自然语言）分别校准，交叉在两种评测集上测 perplexity，验证"AWQ 跨分布涨幅远小于 GPTQ"这一 AWQ 论文核心论点是否在自己的模型上复现。

**P2.3 — 校准集大小消融（S，半天，坑6）**
"校准集大小 vs perplexity"曲线，验证过小样本(几十条量级)是否出现过拟合迹象。

**P2.4 — 云端速度对比（M，云端阶段）**
LLM Compressor 跑 GPTQ 基线，测真实加速比——Mac 阶段无法产出这个数字（平台约束）。

### 支柱3：AgentBench-OS 子集评测

**P3.0 — 任务集改编（M，2天）**
15-20个 bash/工具调用任务，改编自 AgentBench OS 子环境，具体任务清单见附录B。

**P3.1 — 接受率对比实验（S，1天）**
同一模型对，工具调用/结构化输出场景 vs 自由文本对话场景，对比 α 差异，验证"结构化输出接受率更高"这一生产界共识。

### 支柱4：Batch Size 交叉点实验

**P4.0 — Mac 小模型扫描（S，1天）**
batch ∈ {1,4,8,16,32,64}，7B级别模型，投机解码和量化各测一条吞吐提升曲线。

**P4.1 — 显存占用记录（S，与P4.0同步）**
每个 batch size 下草稿模型权重和 KV cache 各自估算显存占用(坑8)。

**P4.2 — 云端大模型验证（M，云端阶段）**
更大模型/更高 batch 验证交叉点是否随模型尺寸偏移，对标 Nightjar 30.25% 倒退量级(坑7)。

### 支柱5：自适应投机步长控制器

**P5.0 — GammaTune 算法实现（M，2天）**
Algorithm 1：接受数 A 与当前 γ 相等时扩窗（$\gamma \leftarrow A + \delta$）；否则 EMA 更新 $\bar\gamma \leftarrow \text{clip}(\gamma_{\min},\gamma_{\max},(1-\eta)\bar\gamma + \eta A)$，$\gamma \leftarrow \lceil \bar\gamma \rceil$。三轮数值算例见**附录A.2**。

**P5.1 — 波动场景鲁棒性测试（S，1天，坑9）**
混合代码生成+开放聊天的 prompt 序列，中途切换任务类型，验证 GammaTune 在非稳态场景下是否还有优势，还是像论文承认的那样收益打折。

**P5.2 — 量化-γ耦合实验（M，云端阶段，坑10）**
复用 P2 的量化模型(FP16 vs 自己的4-bit AWQ版本)，对同一 draft/target 组合分别扫描最优γ，验证"量化后最优γ是否偏移"，对标 SpecKV 自己实测的 2→8(4倍)偏移量级(v8更正：不是此前误引的41.2%，见坑10)。

**P5.3 — Batch-aware 熔断器（M，2天，坑11）**
高 batch 时自动降级/禁用投机解码；**必须包含**周期性重探测机制(避免 DSD 式"重激活难题")；**必须实测**切换开销(KV cache重建成本，避免 BanditSpec 式"假设切换免费"的问题；Nightjar 实测同类切换开销量级为17.87-102.03ms，随输入长度/batch size变化，可作为自己实测数字是否合理的参考量级)。

**P5.4 — 生产基线对比（S，1天）**
对标 HF Transformers 内置动态投机解码(4.45.0起默认行为)，给出定量对比结论(不要求超越，要求诚实数字)。

---

## 8. 考虑过的其他方案

| 决策点 | 选择的方案 | 考虑过的替代方案 | 拒绝理由 |
|---|---|---|---|
| 投机解码架构 | 经典独立草稿模型 | 训练专用speculative head(EAGLE/Medusa) | 需要多卡训练资源(IBM用FSDP两阶段训练)，个人项目没有这个预算 |
| 量化自研对象 | AWQ | GPTQ | 实现复杂度(Hessian/Cholesky分解)和时间线不匹配；AWQ对校准集分布更鲁棒 |
| 量化对比基线工具 | LLM Compressor | 分别装AutoGPTQ+AutoAWQ | 行业已转向统一工具；且两者在Mac上都不可用 |
| Mac开发阶段主框架 | HF Transformers(MPS backend) | (a)vLLM CPU (b)vLLM-metal (c)MLX主实现 | (a)比llama.cpp慢20-30倍 (b)2026年仍是早期版本 (c)HF Transformers有论文先例(SpecKV方法论)，MLX降级为交叉验证工具 |
| Agent评测范围 | AgentBench OS子环境 | 完整八个环境 | 完整版对时间线太重；OS子环境恰好是投机解码效果最好的场景 |
| 支柱5控制器算法 | GammaTune(EMA) | BanditSpec(bandit理论)/AdaEDL(熵基早停) | 不是"太复杂做不到"（BanditSpec的UCBSpec核心算法其实很简单，见坑12的v8更正）；真实理由是这两套框架各自的核心目标(regret-bound理论保证 / 熵阈值提前停止)和Specter支柱5的核心目标(自适应步长)只有部分重叠,引入完整框架的抽象成本和项目目标不成比例；两者保留为文献对比对象 |
| 正确性验证标准 | 贪心逐token严格比对+采样统计检验 | 对采样模式也要求KL散度严格为0 | 不同设备/框架间不保证bit-exact输出，严格要求KL=0不现实且无必要 |

---

## 9. 已知风险与应对

### 9.1 平台兼容性硬约束

**风险A：vLLM 在 Mac 上不能用**——vLLM CPU模式实测 3-5 tokens/s(M5 Max, Llama-3.1-8B, batch=4)，对比 llama.cpp Metal 是 92 tokens/s，差距20-30倍；`vllm-metal`插件截至目前仍是早期版本，模型覆盖不全。来源：[vLLM Issue #1441](https://github.com/vllm-project/vllm/issues/1441)、[vllm-metal GitHub](https://github.com/vllm-project/vllm-metal)。**应对**：Mac阶段用HF Transformers手写实现(SpecKV论文方法论原文如此)，云端阶段才引入vLLM做对比基线。

**风险B：AutoAWQ/AutoGPTQ/LLM Compressor 全部要求CUDA**——AutoAWQ要求"Compute Capability≥7.5, CUDA≥11.8"；AutoGPTQ维护者明确需要专门写MPS kernel(未实现)；LLM Compressor底层依赖和vLLM生态一致。来源：[AutoGPTQ Issue #223](https://github.com/PanQiWei/AutoGPTQ/issues/223)。**应对**：支柱2拆两半，Mac阶段只产出正确性/压缩率数字，云端阶段才测真实速度。

**参照系：MLX 生态经验阈值**——`mlx-lm`文档：接受率α>0.65才有净加速。**应对**：阶段0 α-gate改两级判断(见P1.0)。

### 9.2 支柱1已知坑

**坑1**：tokenizer/vocab不一致使接受率静默归零，不报错。同一模型家族内部也可能不一致(Qwen2 1.5B vocab_size=151936, 72B=152064)。**应对**：P1.0显式断言检查。

**坑2**：bonus token采样源搞错——已知真实bug(DSD)：错误地从草稿模型分布采样bonus token，违反正确性保证但不会崩溃。**应对**：单元测试专门检查bonus token采样代码路径来自哪个模型的logits。

**坑3**：batch>1时ragged tensor同步问题——不同序列接受的草稿token数不同，position id/attention mask/KV cache长度参差不齐。**应对**：支柱4设计里手动维护per-sequence状态，不依赖框架自动padding。

**坑4**：draft/target尺寸"死亡区间"——尺寸差距不够大(2-3倍)可能比不用还慢。**应对**：诊断标准——α尚可但墙钟时间变慢，先查草稿模型自身延迟占比。独立佐证(v8新增)：AdaEDL论文的对照实验里，用未经微调的TinyLlama-1B给Llama2-7B做草稿模型、固定投机长度=7时，静态投机解码(Base-SPD)反而比不用投机解码的自回归基线慢16%；而同样的模型对，只要加上自适应提前停止(AdaEDL/Max-Confidence-SPD)，就变成比自回归快43%——同一组模型，"静态步长"和"自适应步长"的差别决定了投机解码是帮倒忙还是真加速，这是坑4现象的一个独立数据点（AdaEDL论文本身不复现，只作为佐证引用，见§3非目标3）。

### 9.3 支柱2已知坑

**坑5**：校准集分布不匹配使GPTQ严重过拟合，AWQ相对稳健——跨分布(PubMed→Enron)时AWQ涨0.5-0.6，GPTQ涨2.3-4.9。**应对**：P2.2交叉验证实验复现这个矩阵。

**坑6**：校准集太小会过拟合——标准实践C4数据集、group size 128。**应对**：P2.3做校准集大小消融。

### 9.4 支柱4已知坑

**坑7**：Nightjar实测高负载下投机解码相对无投机倒退最多30.25%，比"batch>64开始下降"更极端更具体。**应对**：P4.2对标这个数字。

**坑8**：草稿模型权重和KV cache抢显存，独立于计算瓶颈之外的第二个原因——与我们自己24GB Mac的资源约束是同一原理的缩影。**应对**：P4.1记录显存占用维度。

### 9.5 支柱5已知坑

**坑9**：GammaTune自己承认的边界——draft/target匹配度高(方差小)时收益有限，依赖历史接受率使其在对抗性/剧烈波动场景下退化。**应对**：P5.1设计波动场景测试。

**坑10（价值最高的一条，v8已更正）**：SpecKV自己的实验(Table 1)发现最优γ随目标模型压缩程度(BitsAndBytes的FP16/INT8/NF4，不是AWQ)从FP16下的2偏移到INT8下的8，偏移量级4倍——量化和自适应控制不是独立技巧。**v8勘误说明**：v6/v7此前引用的"偏移最高达41.2%"是误归因——那个数字实际出现在SpecKV论文里，但是SpecKV自己转述另一篇论文(Smurfs, arXiv 2405.05955)关于"γ会随不同模型/batch size/数据集变化"这个一般性观察时提到的，不是SpecKV关于压缩程度这个具体维度的实验结果。真正该引用的是SpecKV Table 1的2→8这组数字，以及其MLP-16控制器让每步期望token数比固定γ=4提升56.0%(开销0.34ms，占单步耗时不到0.5%)这个headline结果。**应对**：P5.2复用支柱2的量化模型验证偏移，对标2→8(4倍)而非41.2%。

**坑11**：Nightjar批评DSD"禁用后不再收集观测数据，难以重新启用"，批评BanditSpec"忽视KV cache重建开销"。**应对**：P5.3熔断器必须含周期性重探测机制+实测切换开销；Nightjar自己实测的同类开销在RTX 4090+DeepSeek-R1-Distill-Qwen-7B配置下是17.87ms(短输入/小batch)到102.03ms(长输入/大batch)，可作为自己实测数字量级是否合理的参照。

**坑12（v8重新措辞）**：不复现BanditSpec的理由不是"太复杂"——BanditSpec论文自己在附录B.2说UCBSpec是"最简单的一类UCB算法之一"(只维护经验均值+置信半径，作者刻意避开了更复杂的Thompson Sampling/KL-UCB)，核心算法本身并不难写。真正的理由是：(a) 它的K-armed bandit框架没有把batch size当作决策的上下文特征，这个局限BanditSpec自己在"未来工作"章节(Contextual Bandits方向)和Nightjar的批评里都承认；(b) 它的regret-bound理论分析完全没有建模"重新启用投机解码需要重建KV cache"这个切换成本，这正是P5.3要弥补的地方。引入BanditSpec完整的bandit框架和regret-bound分析，对"实现一个自适应步长控制器"这个具体目标而言是不成比例的额外抽象层——GammaTune的EMA方案已经用远小的复杂度覆盖了同一个核心目标。**应对**：只作文献对比引用(§3非目标3)，不重新实现；但P5.3的切换开销实测直接吸收了这个批评里最有价值的部分。

### 9.6 研究诚信：防止自己欺骗自己

> 参考文档有一份"风险台账与反刷设计"，防止平台被用户系统性钻空子。Specter 没有外部用户，类比风险不是"被别人刷"，而是**评测/调参过程中无意识地让结果看起来比实际更好**——这在没有同行评审、只有自己既是研究者又是评审者的个人项目里尤其容易发生，必须显式设计对策，而不是假设自己足够客观。v8发现的SpecKV"41.2%"误归因，本质上也是这类风险的一个变种——不是刻意造假，而是"转述别人的转述"在多手传递中悄悄扭曲了归因关系，值得作为风险5补充。

**风险1：评测集/校准集泄漏到调参过程**——如果 P5.0 的 GammaTune 超参数(η, δ, γ_min/max)是在 P3.0 的 AgentBench-OS 任务集上反复试出来的，那么 P5.1/P5.4 用同一套任务集"验证"控制器有效，验证的只是过拟合。**应对**：P3.0 的 15-20 个任务在设计阶段就固定切出一部分(建议 3-5 个)作为 held-out set，只在所有超参数定下来之后跑一次，不回头调整。

**风险2：Best-of-N 挑好看的一次运行报告**——batch 扫描(P4)、γ 扫描(P1.4)、量化对比(P2.4)这类实验都有随机性(采样温度、校准集shuffle顺序)，如果每个数字都是"跑了几次挑最好的一次"，报告出来的加速比会系统性偏高。**应对**：凡是进最终报告的数字，至少跑3次取均值±标准差；如果标准差大到影响结论(比如GammaTune 15%提速的置信区间跨过了0)，如实写出"这次没有统计显著性"，而不是只报均值。

**风险3：正确性验证器本身有bug却一直"通过"**——P1.2的贪心比对器如果实现有误(比如比较逻辑写反、或者两条推理路径共享了同一份缓存导致"殊途同归"式的假阳性)，会一直静默通过，给人"正确性已验证"的错觉，比不验证更危险。**应对**：故意注入已知会破坏正确性的bug(比如手动让bonus token从错误的模型采样、或者手动打乱一个draft token)，确认验证器**必须报错**——"测试验证器本身"是P1.2完成判据的一部分，不是事后可选项。

**风险4：符合预期的结果比违反预期的结果少一道复核**——如果P2.2真的复现出"AWQ比GPTQ更抗跨分布过拟合"，因为这和AWQ论文的结论一致，很容易不假思索地直接采信；但如果结果恰好和论文数字接近到可疑的程度，反而应该多查一遍(比如检查是不是校准代码不小心复用了同一份数据、或者perplexity计算窗口和论文不一致导致巧合)。**应对**：任何"和文献高度吻合"的结果，都过一遍"如果这是bug造成的假象，最可能是哪个环节"的反向检查清单，再写进报告。

**风险5：引用链条比原文长两层以上时,归因关系会悄悄失真(v8新增)**——多手转述(读了别人写的转述,而不是原文;或者笔记里记的是"论文A提到41.2%"而没记清楚这个数字是A自己的结果还是A引用B的结果)是最不容易被自己发现的错误类型,因为读起来完全通顺、不会触发"这里有问题"的直觉。SpecKV的41.2%误归因就是这样进入v6/v7的。**应对**：任何要写进计划或最终报告、且来自"精读论文"这个动作的具体数字，标注时至少包含"论文名+这是第几节/哪张表+这是该论文自己的结果还是它转述的背景引用"三项，缺一项就视为未完成溯源，不能直接采信。

---

## 10. 依赖与假设

### 硬件/环境
- 开发机：Mac,24GB统一内存,无独立GPU
- 云端：Vast.ai/RunPod按需租用NVIDIA GPU实例

### 软件依赖(按阶段)
| 阶段 | 依赖 | 约束 |
|---|---|---|
| Mac开发 | HuggingFace Transformers, PyTorch(MPS) | 确认所选模型对MPS算子支持完整 |
| Mac交叉验证 | mlx-lm | 仅α数量级sanity check |
| 云端验证 | vLLM | CUDA≥11.8 |
| 云端量化对比 | LLM Compressor | CUDA环境,不支持Mac |

### 模型/数据假设
- draft/target模型对暂定Qwen2.5系列,具体尺寸待P1.0结果确定
- 假设模型对共享一致tokenizer/vocab(P1.0显式验证,不假设)
- 量化校准：C4 + 至少一个分布明显不同的第二数据集(P2.2)
- Agent评测：改编自AgentBench OS子环境的15-20个任务(附录B)，其中3-5个held-out(9.6风险1)

---

## 11. 路线图与工作量(对标参考文档 §6 格式)

| 阶段 | 上线内容 | 工作量 | 环境/预算 | 完成判据 |
|---|---|---|---|---|
| 0 | P1.0前置gate | S(0.5天) | Mac,$0 | vocab断言通过;α≥0.4;mlx-lm交叉验证一致 |
| 1 | P1.1-P1.4投机解码核心+验证 | M(7天) | Mac,$0 | 贪心100%一致;采样α与理论公式吻合;bonus token测试通过;验证器故障注入测试通过(9.6风险3);γ扫描曲线产出 |
| 2 | P2.0-P2.3 AWQ校准+正确性 | M(4天) | Mac,$0 | perplexity涨幅数字产出;跨分布矩阵产出;校准集消融产出 |
| 3 | P3.0-P3.1 AgentBench-OS | S(3天) | Mac,$0 | 15-20任务跑通(含3-5个held-out);结构化vs自由文本α对比数字产出 |
| 4 | P5.0-P5.1+P5.3自适应控制器(Mac部分) | M(4天) | Mac,$0 | GammaTune复现论文量级提速(3次跑均值±标准差);波动场景测试产出;熔断器重探测机制实现 |
| 5 | P4.0-P4.1 batch交叉点(Mac部分) | S(2天) | Mac,$0 | 交叉点初步曲线产出;显存占用数据产出 |
| 6 | P2.4+P4.2+P5.2+P5.4 云端规模化验证 | M(3天) | 云端,$30-50 | LLM Compressor对比数字;vLLM对比数字;量化-γ偏移结论(对标2→8而非41.2%);HF基线对比数字;held-out集跑一次最终确认 |
| 7 | 产出 | S(2天) | $0 | GitHub repo+README定稿;简历bullet定稿 |

**成本注记**：Mac阶段全程$0；云端阶段预算硬顶$50(护栏指标)，超支即停止规模化验证用已有结果收尾。**总时长**：约25-26天。

---

## 12. 验证/测试计划

- **贪心模式**：逐token严格比对，目标模型直接推理 vs 投机解码输出必须完全一致(或仅可忽略浮点误差)。
- **采样模式**：(a)统计层面验证实测α与理论公式$E[\min(p,q)]$吻合;(b)下游任务指标parity(差距<2点)。
- **量化正确性**：perplexity涨幅、逐层量化误差分布。
- **自适应控制器**：稳定场景+故意设计的波动场景双重测试。
- **验证器自身**：故障注入测试(9.6风险3)——手动破坏正确性，确认验证器能抓到。
- **引用溯源(v8新增)**：任何写进最终报告的文献数字，附上"论文名+章节/表号+是否为该论文自己的结果"三项标注(9.6风险5)。
- **交叉验证**：mlx-lm跑同一模型对做α数量级对拍；参考[romsto/Speculative-Decoding](https://github.com/romsto/Speculative-Decoding)等开源实现做输出合理性sanity check。

---

## 13. 端到端走读（计划中的产出格式，非已实测数字）

> 对标参考文档 §8"一天的旅程"。参考文档的故事里有一次交换差点"胎死腹中"——用叙事本身去检验系统在不顺利情况下是否还立得住,不只是演示顺利路径。下面的走读不只有一条顺利执行的任务,还包含一次"控制器一度做出错误决策、系统靠自己的补救机制而不是运气修正过来"的场景。

一个 agent 收到任务："读取 `repo/utils.py`，把其中的 `parse_config` 函数重构成支持嵌套 key 的版本，跑通已有单测。"

**第一幕：顺利的主路径**

1. **草稿模型先开口**。3B 级别的草稿模型看到任务描述 + 文件内容，自回归生成前 γ=5 个候选 token（比如先吐出 `def parse_config(`）。因为这是代码补全场景（经验依据来自 IBM/PyTorch 生产复盘：代码场景可以用更多 draft token），P5.0 的 GammaTune 控制器此时观察到最近几轮的高接受率，已经把 γ 从默认值扩到 7。

2. **目标模型单次前向验证**。7B 目标模型对这 7 个候选 token 做一次前向，逐位比较：前 5 个匹配，第 6 个不匹配——接受 5 个，从目标模型的调整分布里重采样第 6 个，丢弃第 7 个候选，本轮产出 6 个 token。P1.3 的统计验证器把这一步的接受数记录进 α 的滚动窗口。

3. **量化在背后悄悄生效**。目标模型本身是 P2 阶段自己校准出来的 4-bit AWQ 版本——如果这一步是在 Mac 上跑，验证的是"输出和 FP16 目标模型比 perplexity 涨幅是否可控"；如果是云端阶段用 LLM Compressor 的 GPTQ 版本做对照，验证的是"真实加速比"。

**第二幕：批量压力下控制器暂时判断失误——以及它怎么发现自己错了**

4. **batch 从 1 涨到 16。** 同时有 16 个类似的重构任务在跑。P4 的 batch 扫描告诉我们：从某个 batch 值开始，草稿模型的开销不再被目标模型省下来的前向次数覆盖，P5.3 的熔断器观察到吞吐下降信号，把 γ 降到 1（等效关闭投机解码）。

5. **就在这个时刻，其中4个任务的性质悄悄变了**——它们不再是"重构一个函数"这种代码模式很规律的任务，而是转成了"读一段自然语言 changelog，判断要不要触发一次版本号升级"这种更接近自由文本推理的任务。如果熔断器是 DSD 那种"关了就不会再看"的设计（坑11），它会一直误以为"投机解码在这批任务上不划算"，即使这4个任务其实和"高batch"完全无关，是任务类型本身变了才该重新评估。**这正是9.6风险4想防的那种陷阱**：一个"符合预期"的判断（高batch→该关）恰好掩盖了另一个真正原因（任务类型变了）。

6. **P5.3设计的周期性重探测机制在第50步插入一次γ=3的试探**——探测到这4个任务的α其实回升到了0.6以上（不是因为batch降了，是因为它们混进了原本就该受益的代码模式任务），控制器没有整体重新开启投机解码，而是——按照P5.3的设计意图——把粒度下放到per-task级别记录这个探测结果，为后续同类任务重新评估γ提供依据。这一步在实现前是一个**设计待验证的假设**，不是承诺的行为：P5.1的波动场景测试就是专门用来检验控制器在"部分任务性质变化、部分没变"这种混合场景下是否真的按预期反应，还是像GammaTune论文承认的那样在非稳态场景下打折扣（坑9）。

**第三幕：落到评测**

7. **最终评测**。这个重构任务本身来自 P3 改编的 AgentBench-OS 任务集里的一条，最终是否 FINISHED（单测通过）决定了这条 trace 在下游任务指标里怎么计分。第二幕的那次"熔断器差点判断失误"不会出现在最终的加速比数字里——它会被记录进P5.3的实验日志(附录C)，作为"控制器设计是否达到预期"这一独立问题的证据，和"这个任务本身有没有做完"分开评估，不混为一谈。

这条走读连接了支柱1(接受率)、支柱2(量化正确性)、支柱4(batch行为)、支柱5(自适应控制，包括它可能犯错的地方)、支柱3(任务结果)——第二幕的存在是刻意的：一个只会讲成功故事的系统设计,大概率是因为设计者没有认真想过失败模式,而不是因为系统真的不会失败。

---

## 14. 我们能讲的故事

一句话定位候选（面试/简历场景下用）：

1. "投机解码和量化不是两个孤立的优化技巧——它们是同一个系统原理的两个例证，我测出了这个原理失效的具体边界在哪。"
2. "调库打开投机解码只要一行 flag；我把这一行 flag 背后的正确性证明、失效条件、自适应控制全部手写实现并验证了一遍。"
3. "我知道生产界的最优解是训练专用 speculative head——我诚实地选择了预算约束下可行性更高的方案，并且把这个取舍写进了报告里。"

北极星论点：这个项目要证明的不是"我能让模型跑得更快"，而是"我理解这些加速技巧在什么条件下会失效，以及如何设计一个知道自己什么时候该失效的系统"——这是从"实现了论文里的技巧"到"理解了系统边界"的差距，也是新grad项目和有2-3年经验的工程师项目之间最常见的差距。

---

## 15. 产出物

- GitHub repo(代码+README，含论文引用、prior art引用、诚实边界说明——包括平台兼容性踩坑过程本身)
- 简历bullet(见附录D)
- 本系列计划文档(v1-v8)作为项目思考过程的书面记录，可选择性放入repo的`docs/`目录

---

## 16. 悬而未决的问题

1. 具体draft/target模型尺寸对——暂定Qwen2.5系列，等P1.0的α-gate结果才能定。
2. P5.3熔断器是否和P4的batch扫描代码共享同一套实现——倾向共享，等P4实现完看接口是否好复用。
3. 云端预算在阶段5和阶段6之间的具体分配——等实际云端GPU租用单价和阶段4本地测试结果出来后细化。
4. AgentBench任务集改编的具体任务数量(15个还是20个，扣除held-out后实际可调参的任务数会更少)——先按20个设计，如果太重再削减。

---

## 17. 决策记录

v1-v6：均由用户在迭代过程中通过继续深化调研要求的方式确认，尚无正式的"批准进入实现阶段"信号。
v7：读完参考文档全部28页后补齐颗粒度缺口，内容决策不变，等待用户确认后再讨论是否进入阶段0。
v8(本版本)：用户要求"回到原始材料"精读5篇自适应控制论文原文并授权自主执行；发现SpecKV"41.2%"误归因并更正为真实数字(2→8/4倍)，重新措辞BanditSpec不复现的理由(不是"太复杂"而是"不建模batch/切换开销")，补充若干可精确溯源的新引用；架构/技术决策不变，仍等待用户确认后再讨论是否进入阶段0。

---

## 附录A：核心公式、算法伪代码与数值算例

### A.1 Rejection Sampling（Leviathan et al. 2023）

```
for i in 1..γ:
    draft_token[i] ~ p_DM(x | context, draft_token[1..i-1])
target_probs = TargetModel.forward(context, draft_token[1..γ])  # 单次前向
n_accepted = 0
for i in 1..γ:
    r ~ Uniform(0,1)
    if r < min(1, p_TM(draft_token[i]) / p_DM(draft_token[i])):
        accept draft_token[i]; n_accepted += 1
    else:
        break
if n_accepted < γ:
    x_new ~ norm(max(0, p_TM(x) - p_DM(x)))  # 从调整分布重采样
else:
    x_new ~ p_TM(x | context, draft_token[1..γ])  # bonus token，必须来自目标模型
```

**数值算例**（γ=3，词表简化成3个候选token {A, B, C}，模拟一次verification）：

草稿模型对第1个位置的分布：`p_DM = {A:0.7, B:0.2, C:0.1}`，采样得到 `A`。
目标模型对同一位置的分布：`p_TM = {A:0.4, B:0.3, C:0.3}`。

- 接受概率 = `min(1, p_TM(A)/p_DM(A))` = `min(1, 0.4/0.7)` = `0.571`。
- 假设采样到的 `r=0.8 > 0.571` → **拒绝**，本轮到此为止，不再看后续候选。
- 从调整分布重采样：`p'_TM = norm(max(0, p_TM - p_DM))` = `norm({A:max(0,0.4-0.7)=0, B:max(0,0.3-0.2)=0.1, C:max(0,0.3-0.1)=0.2})` = `{A:0, B:0.33, C:0.67}`。
- 最终这一步吐出的 token 从 `{B:0.33, C:0.67}` 里采样，**绝不会是A**——这符合直觉：目标模型没那么偏爱A（0.4 vs 草稿模型的0.7），所以拒绝后重采样把A的概率清零，公平地把概率让给目标模型相对更青睐、又被草稿模型低估的B和C。

这个算例也直接演示了坑2要防的错误：如果重采样时不小心又从 `p_DM` 采样（而不是从上面算出的调整分布 `p'_TM` 采样），会系统性地偏向草稿模型的偏好（A），累积下来的输出分布就不再等于目标模型单独推理的分布——这是"看起来能跑、但正确性保证已经被破坏"的那类bug，P1.2的贪心验证器要能抓到,9.6风险3要求故意注入这类bug测试验证器本身。

### A.2 GammaTune 成本模型与Algorithm 1（Kim et al. 2025）

$$\text{cost} = \frac{N}{\alpha\gamma+1} \times (c+\gamma) \times T_{draft}, \quad c = T_{target}/T_{draft}$$

```
if A == γ:  # 上一轮全部接受，可能还有余量
    γ ← A + δ
else:
    γ̄ ← clip(γ_min, γ_max, (1-η)·γ̄ + η·A)
    γ ← ceil(γ̄)
```

**三轮数值算例**（η=0.3, δ=2, γ_min=1, γ_max=10, 初始 γ̄=3, γ=3）：

| 轮次 | 实际接受数 A | 触发分支 | 计算 | 新 γ̄ | 新 γ |
|---|---|---|---|---|---|
| 1 | 3（全部接受，A==γ） | 扩窗 | γ ← 3+2 | （不更新γ̄） | 5 |
| 2 | 2（部分接受，A<γ） | EMA | γ̄ ← clip(1,10, 0.7×3 + 0.3×2) = clip(1,10,2.7) | 2.7 | ceil(2.7)=3 |
| 3 | 3（全部接受，A==γ） | 扩窗 | γ ← 3+2 | （不更新γ̄） | 5 |

可以看到：只要连续命中"全部接受"，γ 会持续扩窗式增长（3→5→7→...直到γ_max）；一旦某一轮没有全部接受，立刻回落到EMA的保守估计，不会让γ在一次不理想的输出后继续野蛛式增长——这是"扩窗要快，收缩要稳"的设计意图在数字上的体现，也是P5.1波动场景测试要验证的核心行为。原论文还提出了一个GammaTune+变体，额外加入基于草稿模型top-1 logit概率的提前停止阈值τ，在4组模型对上平均提速从GammaTune的1.15±0.05x略微提升到1.16±0.03x——Specter不实现这个变体(不在P5.0范围内)，只作为"同一套EMA思路还能怎么扩展"的背景参考。

### A.3 AWQ 逐通道缩放——直觉性数值算例

AWQ 的核心洞察：并非所有权重通道都同等重要，激活值幅度大的输入通道对应的权重通道更"显著"，量化时应该被优先保护。一个简化的两通道玩具例子：

假设某一层有两个输入通道，激活值统计幅度分别为 `|X1|=10, |X2|=1`（通道1的激活值普遍大10倍，说明这个通道承载的信号更重要）；原始权重 `W1=0.5, W2=0.5`（两个通道权重原本一样大）。

- 如果直接对 `W1, W2` 统一做4-bit量化而不做任何缩放，两个通道的量化误差量级相近——但通道1的误差会被激活值放大10倍(`误差×|X1|`)，对最终输出的影响远大于通道2。
- AWQ 的做法：引入缩放因子 `s`(通常与激活值幅度正相关，比如 `s=|X1|^α` 的形式，α 是一个校准出来的超参数)，量化前把 `W1` 放大为 `W1×s`、把对应的激活值缩小为 `X1/s`，使数学上等价(`(W1×s)×(X1/s)=W1×X1`不变)，但量化误差被压缩到激活值更"迟钝"（即通道1本身激活值大、相对量化误差的百分比影响反而变小）的那一侧。
- 净效果：通道1(重要通道)的有效量化精度被"借用"了缩放因子提高，通道2(不重要通道)相应地牺牲一点精度——这就是"逐通道缩放"要在P2.1里通过校准集统计量算出来的那组 `s` 值,不是拍脑袋决定的。

这个算例只是帮助建立直觉，P2.1实现时的具体缩放因子搜索方式（网格搜索最小化量化后输出误差）以AWQ原论文Section 3的公式为准，不是这里简化的启发式。

---

## 附录B：AgentBench-OS 子集任务清单（草案，实现时细化）

改编自 AgentBench 的 OS/bash 子环境，聚焦"生成结构化输出/工具调用"这一投机解码效果最好的场景，草案任务类别（每类2-4个具体任务，共15-20个，其中3-5个在设计阶段就标记为held-out，仅在超参数定稿后跑一次，见9.6风险1）：
1. 文件操作类（读取/修改/重命名指定文件）
2. 代码重构类（如 §13 走读示例：重构函数签名并跑通单测）
3. 命令行工具调用类（用 grep/find/awk 完成指定查询并输出结构化结果）
4. 多步骤依赖类（前一步的输出是后一步的输入，测试长上下文下的接受率变化）

每个任务的完成判据（FINISHED）：产出的文件/命令输出通过预先写好的校验脚本，不依赖人工判断。

---

## 附录C：实验日志 / Telemetry Schema + 诊断分析代码

> 对标参考文档里引用自己产品日志(`tx_demand_search_log`等)和附录A"只读查询"的做法。Specter没有生产数据库，日志落地为本地文件（jsonl/parquet），下面的"查询"是pandas风格的伪代码,不是真实SQL。

### C1. 日志字段定义

**每个投机解码步骤应记录**：`step_id, draft_tokens[], target_logits_hash, n_accepted, gamma_used, latency_ms, model_pair_id, quantization_level`

**每个γ调整决策应记录**（P5.0-P5.3）：`timestamp, prev_gamma, new_gamma, trigger_reason(EMA更新/扩窗/熔断/重探测), acceptance_history_window`

**每个batch扫描数据点应记录**（P4）：`batch_size, throughput_tokens_per_sec, draft_model_memory_mb, kv_cache_memory_mb, speedup_ratio_vs_no_spec`

**每个量化校准实验应记录**（P2.2/P2.3）：`calibration_dataset, eval_dataset, n_calibration_samples, perplexity, group_size`

### C2. 诊断分析代码（只读，实现阶段直接复用）

```python
# 滚动窗口接受率 α（对应P1.3、P5.0的EMA输入）
df_steps["alpha_rolling"] = (
    df_steps["n_accepted"] / df_steps["gamma_used"]
).rolling(window=50).mean()

# 加速比按 batch_size 分布（对应P4，定位交叉点）
speedup_by_batch = (
    df_batch.groupby("batch_size")["speedup_ratio_vs_no_spec"]
    .agg(["mean", "std", "count"])
)
# 交叉点 = speedup_ratio 均值首次跌破 1.0 对应的 batch_size

# γ调整触发原因分布（对应P5.3，检查熔断/重探测是否按预期触发）
trigger_counts = df_gamma_decisions["trigger_reason"].value_counts(normalize=True)

# 量化程度对最优γ的偏移（对应P5.2，坑10核心实验，对标SpecKV自己的2→8而非误引的41.2%）
optimal_gamma_by_quant = (
    df_steps.groupby("quantization_level")
    .apply(lambda g: g.loc[g["latency_ms"].idxmin(), "gamma_used"])
)
shift_ratio = optimal_gamma_by_quant.max() / optimal_gamma_by_quant.min()

# 跨分布校准 perplexity 矩阵（对应P2.2，坑5核心实验）
ppl_matrix = df_calibration.pivot_table(
    index="calibration_dataset", columns="eval_dataset", values="perplexity"
)

# 验证器故障注入测试结果（对应9.6风险3）：注入已知bug后必须全部FAIL
assert df_fault_injection_tests["detected"].all(), "验证器漏检——先修验证器再信任其他结果"
```

---

## 附录D：简历 bullet 草稿

- 手写实现投机解码采样算法，贪心模式下逐 token 精确验证正确性，采样模式下通过统计检验和任务指标 parity 确认分布等价；诊断并规避 tokenizer 不一致、bonus token 误采样等已知实现陷阱
- 自研 AWQ 风格激活感知量化校准 pipeline，通过跨分布校准实验验证其相对 GPTQ 的抗过拟合优势
- 实测投机解码与量化在不同 batch size 下的吞吐增益曲线，定位出两者从"内存带宽优化"转为"计算瓶颈拖累"的交叉点，并验证量化程度对最优投机解码步长的系统性影响
- 实现训练-free 的自适应投机解码步长控制器（基于 EMA 的 GammaTune 算法），设计 batch-aware 熔断与周期性重探测机制，对标 Hugging Face Transformers 内置动态投机解码基线
- 基于 AgentBench 方法论改编 agent 工具调用评测集，验证结构化输出场景下投机解码接受率显著高于自由文本生成

---

## 附录E：分类文献与信息来源（v8：更正SpecKV/BanditSpec条目并补充精确引用）

### 一、投机解码核心与SOTA方法
- Leviathan et al. 2023 (arXiv 2211.17192, ICML'23)，*Fast Inference from Transformers via Speculative Decoding* —— 奠基论文之一，rejection sampling 正确性证明的来源，附录A.1算例直接基于其算法描述。
- Chen et al. 2023 (arXiv 2302.01318, DeepMind)，*Accelerating LLM Decoding with Speculative Sampling* —— 另一篇奠基论文，70B Chinchilla 上 2-2.5x 加速的实证参考。
- Li et al. 2024 (arXiv 2401.15077)，*EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* —— SOTA方案，特征级drafting而非独立小模型，Specter不复现但作为"生产界最优解"在§8/§14里对照引用。
- Li et al. 2024 (arXiv 2406.16858)，*EAGLE-2: Faster Inference with Dynamic Draft Trees* —— 树形草稿+动态调整，接受率更高，SGLang生产数据（batch=24掉速/batch=64仍有效）来自这条线。
- Cai et al. 2024 (arXiv 2401.10774)，*Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads* —— 多头并行解码对比基线，需要专门训练，是§8"投机解码架构"决策点的拒绝对象之一。
- 综述 2025 (arXiv 2502.19732)，*Speculative Decoding and Beyond: An In-Depth Survey* —— 全景综述，含SpecInfer/Sequoia等树形方法背景。

### 二、量化方法
- Lin et al. 2023 (arXiv 2306.00978)，*AWQ: Activation-aware Weight Quantization* —— 支柱2的核心方法来源，跨分布perplexity数字(+0.5-0.6)、附录A.3算例均基于此文。
- Frantar et al. 2023 (arXiv 2210.17323)，*GPTQ: Accurate Post-Training Quantization* —— 对比基线，Hessian-based，跨分布perplexity涨幅(+2.3-4.9)数字来源，本身不重新实现（§3非目标1）。
- Xiao et al. 2023 (arXiv 2211.10438)，*SmoothQuant: Accurate and Efficient PTQ for LLMs* —— W8A8方案，仅作背景对比，说明W4A16（AWQ路线）更适合本项目的低batch/延迟优先场景。

### 三、服务层架构
- Kwon et al. 2023 (arXiv 2309.06180, SOSP'23)，*Efficient Memory Management for LLM Serving with PagedAttention* —— vLLM底层架构论文，决定"造轮子 vs 用轮子"边界：Specter不重新实现KV cache管理，vLLM云端阶段作为现成对比基线引入。

### 四、Agent评测方法论
- Liu et al. 2023 (arXiv 2308.03688, ICLR'24)，*AgentBench: Evaluating LLMs as Agents* —— 支柱3任务设计方法论来源，只改编其OS/bash子环境（§3非目标2）。
- Qin et al. 2023 (arXiv 2307.16789)，*ToolLLM: Facilitating LLMs to Master 16000+ APIs* —— 工具调用任务集设计参考。

### 五、自适应投机步长控制（2024-2026前沿，v8：五篇均已精读原文，非摘要转述）
- Kim et al. 2025 (arXiv 2504.00030, Texas A&M)，*Token-Driven GammaTune* —— P5.0核心算法来源，Algorithm 1（EMA更新+扩窗）直接实现，成本模型 cost=N/(αγ+1)×(c+γ)×T_draft 中 c=T_target/T_draft 典型取值4-10；4组模型对(Vicuna-13B/160M、Vicuna-7B/68M、Llama-3.1-70B-Instruct[int8量化]/8B-Instruct、Llama-3.1-8B/3.2-1B)上平均提速1.15±0.05x，单张80GB H100实测；论文提出的GammaTune+变体(加入基于草稿模型logit的提前停止)提速略升至1.16±0.03x(附录A.2已引，Specter不实现该变体)；Limitations章节明确承认低方差场景收益有限、对抗性场景会退化（坑9）。
- SpecKV 2026 (arXiv 2605.02888)，*Adaptive Speculative Decoding with Compression-Aware Gamma Selection* —— **v8勘误**：此前(v6/v7)引用"最优γ随量化程度偏移最高达41.2%"，经读原文确认为误归因——41.2%出自SpecKV转述另一篇论文(Smurfs, arXiv 2405.05955)关于γ随模型/batch size/数据集变化的一般性背景论点，不是SpecKV自己的压缩实验结果。SpecKV自己的真实结果(其Table 1)：最优γ随压缩程度(BitsAndBytes的FP16/INT8/NF4)从FP16的2偏移到INT8的8(4倍/300%)；headline结果是其MLP-16控制器让每步期望token数比固定γ=4的基线提升56.0%，控制器开销仅0.34ms(单步耗时<0.5%)；草稿置信度/熵与接受率的相关性约0.55-0.58量级，跨压缩程度保持一致，其中min_draft_confidence(重要性30.0%)和max_draft_entropy(24.1%)是最具预测力的特征；实验平台Llama 3.2 1B/3B(共享词表128,256)，单张RTX 3090。坑10、P5.2、指标体系表已更正为2→8这一真实数字。
- Agrawal et al. 2024 (arXiv 2410.18351, Qualcomm AI Research)，*AdaEDL: Early Draft Stopping via Entropy-based Lower Bound* —— 熵基提前停止判据 $1-\sqrt{\gamma H_{DM}(x)}<\lambda$ 来源，注意此处γ是论文自定义的"熵因子"超参数(全部实验固定为0.2)，**和投机解码通常语境下的"步长γ"是两个不同的量，命名冲突**，Specter自己实现GammaTune/SpecKV时用的γ是步长含义，引用AdaEDL时需注意区分。DL=3/7/16下接受token数标准差递增(1.2/1.92/2.35)的数据确认来源为其Figure 1/附录Fig 7c，Dolly-15k创作数据集，目标模型Llama2-7B，草稿模型是对齐微调过的115M模型。独立佐证坑4的数据点(v8新增，已写入9.2)：未微调的TinyLlama-1B给Llama2-7B做草稿、固定投机长度=7时，静态投机解码反而比自回归基线慢16%,而换成自适应提前停止后变成快43%。只作文献对比不重新实现（§3非目标3）。
- BanditSpec 2025 (arXiv 2505.15141)，*Adaptive Speculative Decoding via Bandit Algorithms* —— UCB(UCBSpec)/EXP3(EXP3Spec)双算法框架，理论贡献是"stopping time regret"这个新量的regret bound证明，但论文自己在附录B.2明确说UCBSpec算法本身"是最简单的一类UCB算法之一"（**v8更正**：坑12此前"实现成本过高"的措辞不准确，复杂的是证明过程不是算法本身）；附录B.4明确将自己定位为与SpecDec++(Huang et al. 2024，训练一个接受率预测头)"正交"的方案——训练-free vs 训练-based。其K-armed框架不把batch size作为上下文特征(自己在"未来工作"章节列为Contextual Bandits方向)，也完全没有建模从γ=0切换回γ>0的KV cache重建成本，这两点是Nightjar批评的依据(坑11)，也是Specter选择GammaTune而非BanditSpec的真实理由。
- Nightjar 2025 (arXiv 2512.22420)，*Dynamic Adaptive Speculative Decoding for LLM Serving* —— 高负载30.25%吞吐倒退的数字来源(坑7，P4.2对标对象，原文Section 3.1: "a performance degradation of up to 30.25% compared to Vanilla Decoding")；批评DSD"重激活难题"和BanditSpec"忽视切换开销"，直接决定P5.3熔断器必须含周期性重探测机制的设计(坑11)；自己实测的KV cache切换开销(Table 3, RTX 4090 + DeepSeek-R1-Distill-Qwen-7B)在17.87ms(输入长度128/batch32)到102.03ms(输入长度512/batch64)区间，随输入长度和batch size增长；headline结果是6个7B/13B基准设置上平均比无投机解码提速27.29%、比标准投机解码提速8.32%(最高14.76%)、比BanditSpec/DSD分别提速22.89%/19.76%，平均端到端延迟比标准投机解码降低15.16%(最高20.18%)。

### 六、生产实践 / 工程博客（"理论之后发生了什么"）
- [A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)（PyTorch官方博客，2024，IBM Research生产环境实测）—— 全项目最重要的一手资料：Llama2-13B/Llama3-8B/Granite-7B 2x提速，Granite-20B代码模型3x提速；代码模型用更多draft token更划算（P1.1/P3设计依据）；batch>64开始吞吐下降（支柱4存在理由）；训练speculative head比经典两模型方案效果更好但需要多卡训练资源（§8架构决策拒绝理由的一手证据）；精确对比应用贪心而非采样（P1.2设计依据）；先测接受率再测吞吐的方法论（P1.0 gate设计依据）。
- [vLLM Blog: Speculative Decoding](https://vllm.ai/blog/tags/speculative-decoding) —— 原生支持EAGLE/EAGLE-3/Medusa/n-gram草稿，一个flag开启，佐证"调库无技术含量"论点（§2.2）。
- [SGLang 文档](https://docs.sglang.ai/advanced_features/speculative_decoding.html) —— EAGLE在batch=24掉速、EAGLE-3在batch=64仍有效的具体数字来源（支柱4）。
- [llama.cpp docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) —— 经典两模型方案+n-gram自投机的现状参考，EAGLE-3支持仍在讨论（Issue #15902）。
- [AutoAWQ GitHub](https://github.com/casper-hansen/AutoAWQ) —— 官方文档承认高batch场景下W4A16因反量化开销可能变慢（坑8/支柱4呼应），且明确CUDA-only（风险B来源之一）。
- [Medium: A practical guide to INT4 quantization for SLMs](https://medium.com/data-science-at-microsoft/a-practical-guide-to-int4-quantization-for-slms-gptq-vs-awq-olive-and-real-world-results-2f63d6963d1d) —— GPTQ vs AWQ真实场景对比，W4A16延迟优先/W8A8吞吐优先的场景划分依据。
- LLM Compressor（vLLM团队）—— 统一GPTQ/AWQ/FP8量化API，行业从分裂工具链（AutoGPTQ+AutoAWQ）走向统一工具的现状，决定P2.4对比基线工具选择（§8）。
- [vLLM Issue #1441](https://github.com/vllm-project/vllm/issues/1441)、[vllm-metal GitHub](https://github.com/vllm-project/vllm-metal) —— vLLM在Mac上不可用的直接证据（风险A）。
- [AutoGPTQ Issue #223](https://github.com/PanQiWei/AutoGPTQ/issues/223) —— AutoGPTQ需要专门MPS kernel（未实现）的维护者原话（风险B）。

### 七、参考开源实现（对拍用，不直接抄）
- [romsto/Speculative-Decoding](https://github.com/romsto/Speculative-Decoding) —— 干净实现Leviathan et al. 2023算法，含经典自回归/beam search/投机解码三种对照，用于P1.2输出合理性sanity check。
- [foundation-model-stack/fms-extras](https://github.com/foundation-model-stack/fms-extras) —— IBM生产代码开源部分，展示PagedAttention kernel如何支持多head KV cache管理，帮助理解投机解码+KV cache管理结合的工程难点。
- [Pramodith Dissects 博客+notebook](https://pramodith.github.io/posts/speculative-decoding/) —— SmolLM2-360M/1.7B手写贪心投机解码，模型选择思路（同系列不同尺寸）与Qwen2.5系列思路一致，α值/加速比量级sanity check参考。
