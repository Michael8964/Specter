# Specter 项目计划 v2 —— 基于论文精读的修订版

修订依据：完整读完 12 篇论文中最关键的 6 篇（Leviathan 2211.17192、Chen et al. 2302.01318、GPTQ 2210.17323、vLLM/PagedAttention 2309.06180、AgentBench 2308.03688、EAGLE 2401.15077、Medusa 2401.10774）的正文和实验部分，而不只是摘要。相比 v1，本版本修正了三个可能导致项目"做出来但站不住脚"的风险点，并把所有目标数字换成有论文出处的、诚实的量级。

---

## 一、核心定位不变，但验证标准更严格

**Specter：面向 Agent 场景的本地推理加速引擎，附带正确性证明**

三大支柱不变（手写投机解码、AWQ 风格量化、agent workload 评测），但每一支柱的"怎么算做对了"都要按论文原文的标准重新定义。

---

## 二、支柱 1：投机解码 —— 三处关键修正

### 修正 1：draft/target 配对必须先做"存活性检查"，否则可能全盘白做
- EAGLE 论文原文：用 7B 模型给 13B 模型做 draft，因为 7B 本身开销太大，**比不做投机解码还慢**（"rendering it less efficient than vanilla autoregressive decoding"）。
- Leviathan 论文的经验法则："choosing $M_q$ to be around two orders of magnitude smaller than $M_p$ usually performed best"。
- **执行动作**：在正式开发投机解码算法之前，先做一个**半天的存活性验证**——用候选 draft/target 组合跑一次最简单的 baseline，测 draft 模型单独的推理延迟 vs. 目标提速阈值，确认这个组合有正加速空间，再投入接下来一周的实现工作。候选组合建议：Qwen2.5-0.5B/1.5B 做 draft，Qwen2.5-7B 做 target（同系列 tokenizer 一致，尺寸比约 5-14x，在两篇论文的"最优区间"内）。

### 修正 2：正确性验证的标准要改——不是 KL=0，而是统计等价 + 任务指标 parity
- Chen et al. 原文明确说明输出**不会 bit-exact 一致**（"since pseudo-random seeds are processed differently between ArS and SpS... we cannot expect identical outputs"），他们验证"分布不变"用的是两种方法：
  1. 固定设置下大样本统计验证（论文里是在 10K 生成样本上估计 α/接受率的期望）；
  2. **下游任务指标 parity**——HumanEval pass@1、XSum ROUGE-2 在有/无投机解码下几乎相同（差距在 0.1-1.9 个点内，见 Table 1: HumanEval 45.1% vs 47.0%，XSum ROUGE-2 0.112 vs 0.114）。
- **执行动作**：验证分两层——(a) 数学证明层：跑通 Leviathan 论文 Theorem 3.5/Corollary 3.6 的 acceptance rate 公式，实测 α 值与理论公式（$\alpha = E[\min(p,q)]$）做数值比对；(b) 实证层：在同一批 prompt 上分别跑"纯 target 采样"和"投机解码"，比较任务级指标（选一个 code 生成或摘要任务），差距应该在 1-2 个点以内，而不是追求 KL 散度归零这种不现实的标准。

### 修正 3：γ（每轮草稿数）要扫描，不是拍一个固定值
- Chen et al. Figure 1 显示：随着 K 增大，接受率持续下降（因为后面的 token 依赖前面被接受），总耗时先降后趋于平缓甚至回升——XSum 任务在 K=3 时延迟最优，K 再大反而变差。
- **执行动作**：新增一个必做实验——扫描 γ ∈ {1,3,5,7,10}，画出"加速比 vs γ"和"接受率 vs γ"两条曲线，找到自己模型对上的最优 γ，而不是用文献里的默认值。这本身就是一张有说服力的图。

### 加分对比：为什么不直接抄 Medusa/EAGLE
- Medusa 用的"typical acceptance"方案**不保证**分布严格不变（论文原文："it cannot further enhance the acceleration rate. Alternatively we introduce a typical acceptance scheme... using temperature as a threshold"——这是用近似换速度）。
- EAGLE 是目前唯一在贪心和非贪心设置下都有严格证明的高阶方法（"the preservation of the output distribution by EAGLE is theoretically guaranteed for both the greedy and non-greedy settings"），但需要训练一个额外的特征预测层（约 1B 参数，60-70k 条 ShareGPT 对话，4×A100 跑 1-2 天）——这个训练成本超出我们的预算和硬件（24GB Mac / 短时云端租用）。
- **写法**：主实现用经典两模型方案（严格证明、可行性高），报告里专门写一节"为什么不用 EAGLE"，把上面两段论文原文引用进去，展示你懂技术谱系里的取舍，而不是只会调库。

---

## 三、支柱 2：量化 —— 维持原判断，补充可行性证据

精读 GPTQ 后确认：它需要计算 Hessian 矩阵 $H = 2XX^\top$、做 Cholesky 分解、逐 block（B=128）做 lazy batch update（见论文 Algorithm 1），并且有数值稳定性问题需要专门处理（dampening + Cholesky reformulation）。这是实打实的数值算法工程，一周内独立正确实现风险很高。

**维持 v1 结论**：自己实现 AWQ 风格的激活感知量化（只需要统计各通道激活幅值 + 做等价缩放搜索，没有二阶矩阵运算），GPTQ 数字直接用 AutoGPTQ 库跑出来做对比基线。

**新增一个从 GPTQ 论文里学到的重要工程事实**：GPTQ 论文自己承认——"our method currently does not provide speedups for the actual multiplications, due to lack of hardware support for mixed-precision operands (FP16×INT4)"。也就是说，4-bit 权重存储本身不会自动带来推理提速，**速度提升来自专门写的 dequant-on-the-fly 融合 kernel**，减少的是内存带宽而不是算力。
- **执行动作**：如果只是把权重存成 4-bit 但用通用库做 matmul（现场反量化成 fp16 再乘），很可能测不出延迟提升，甚至因为反量化开销变慢。必须用现成的高效反量化 kernel（bitsandbytes / AutoAWQ 自带的 CUDA/Metal kernel），不要自己从零写 kernel（那是另一个项目的工作量）。报告里要明确写清楚："量化的加速收益来自专用 kernel 的内存带宽优化，我们复用了 X 的 kernel 实现，自己实现的部分是校准算法本身"——诚实说明边界，反而显得更专业。

---

## 四、支柱 3：Agent workload 评测 —— 收缩范围，对标权威基准

精读 AgentBench 后发现它的 8 个环境里，Database 和 Knowledge Graph（3B facts 规模）这类环境本身就需要搭建复杂的基础设施（真实 SQL 数据库、Freebase 知识图谱服务），两三周内不现实。

**执行动作**：只挑 **Operating System (OS)** 这一个子环境改编——它是"在 Ubuntu Docker 里执行 bash 命令完成任务"，天然契合我们"本地 coding/dev agent"的定位，用 Docker 起一个沙箱、复用 AgentBench 公开的 OS 任务集或按其任务格式（人类问题 → 可执行 shell 操作 → success rate 打分）自己设计 15-20 个同类任务。评测方法论明确写"参考/改编自 AgentBench (Liu et al., ICLR'24) 的 OS 环境设计"，可信度比完全自创高。

---

## 五、架构（不变，现在有精确数字支撑）

```
[Agent Harness / OS 任务集]  ──bash 工具调用──▶  [FastAPI 服务层]
                                                      │
                                     ┌────────────────┴────────────────┐
                               [Draft 模型]                      [Target 模型]
                          (Qwen2.5-0.5B/1.5B, 4-bit)      (Qwen2.5-7B, AWQ 量化)
                                     └────手写投机解码验证器──────┘
                                                      │
                                        [vLLM PagedAttention KV Cache]
                                                      │
                                   [AWQ 校准 + 逐层误差分析模块（自研）]
                                                      │
                                     [Benchmark：γ 扫描 + AgentBench-OS 子集]
```

vLLM 论文给出的精确理由：13B 模型在 A100 40GB 上，参数占 65% 显存，KV cache 占 30%+；现有系统（Orca 各变体）KV cache 有效利用率只有 20.4%-38.2%，vLLM 靠 PagedAttention 做到 96.3%，吞吐提升 2-4x。这套调度器工程量是团队级的，**继续维持"不重造"的决定**，我们的代码只加在投机解码验证层和量化校准层，插在 vLLM 之上。

---

## 六、修订后的分阶段计划（18-22 天不变，内容有调整）

### 阶段 0：立项 + 存活性检查（第 1-2 天，Mac，$0）
- 新增：draft/target 配对存活性检查（见支柱 1 修正 1），避免选错模型对导致后面全部白做
- 建立评测脚本骨架

### 阶段 1：投机解码手写实现 + 双重验证（第 3-9 天，Mac，$0，+1 天缓冲）
- 实现 Algorithm 1（Leviathan）/ Algorithm 2（Chen）的 rejection sampling
- 数学层验证：实测 α vs 理论公式 $\alpha=E[\min(p,q)]$ 的吻合度
- 实证层验证：下游任务指标 parity（HumanEval 或摘要任务，目标差距 <2 个点）
- **新增必做**：γ ∈ {1,3,5,7,10} 扫描曲线

### 阶段 2：AWQ 量化 + 诚实的 kernel 说明（第 10-14 天，Mac，$0）
- 自己实现激活感知的逐通道缩放校准
- 用 AutoGPTQ/AutoAWQ 现成 kernel 做加速对比，报告中明确标注"哪部分是自己写的算法，哪部分复用现成 kernel"
- 产出：逐层量化误差图 + perplexity 对比表

### 阶段 3：AgentBench-OS 子集评测（第 15-17 天，Mac，$0）
- 用 Docker 沙箱改编 15-20 个 OS 类任务
- 跑通端到端 agent 任务成功率 + 延迟基线

### 阶段 4：规模化验证（第 18-20 天，云端，$20-40）
- 租 Vast.ai 4090/A100，target 模型换成 13B-34B，重跑全部实验
- 补做一次前面在 Mac 上没法测的大模型 draft/target 配对存活性检查

### 阶段 5：产出（第 21-22 天，$0）
- GitHub repo + README（含论文引用、诚实的边界说明）
- 简历 bullet 定稿

---

## 七、修订后的目标数字（全部有论文出处，避免被问住）

| 指标 | 目标 | 出处依据 |
|---|---|---|
| 投机解码加速比（经典两模型方案） | 2-3x | Leviathan 2.6-3.4x；Chen 1.92-2.46x |
| 接受率 α | 0.6-0.85（视 draft/target 相似度） | Leviathan Table 3 实测区间 |
| AWQ 4-bit 量化 perplexity 涨幅 | <5%，理想 <1 point | 对标 GPTQ 论文 4-bit 结果（OPT-175B 仅涨 0.03） |
| 量化显存降低 | ~65-75% | 4-bit vs fp16 理论压缩比 |
| AgentBench-OS 子集任务成功率 | 报告实测值，不设预期（这是探索性实验） | AgentBench 原论文里连 GPT-4 也做不到满分 |

不再写"EAGLE 级别 3-6.5x"这种数字，除非真的实现 EAGLE（预算内不做）。

---

## 八、结论

v1 的三大支柱和总体路线保持不变，v2 的价值在于把"可能被面试官一问就露馅"的三个地方（draft/target 配对风险、正确性验证标准、量化提速的真实来源）用论文原文钉死，同时把 agent 评测范围收缩到两三周内真正能做完、又能对标权威基准的子集。18-22 天预算不变，阶段 1 多留 1 天缓冲应对存活性检查可能的返工。
