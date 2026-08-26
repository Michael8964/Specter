# Specter — 参考论文索引

全部从 arXiv 抓取，来源均为 `https://arxiv.org/pdf/<id>`。

| 文件 | arXiv ID | 论文 | 与项目的关系 |
|---|---|---|---|
| leviathan2023_speculative_decoding.pdf | 2211.17192 | Fast Inference from Transformers via Speculative Decoding (Google, ICML'23) | 投机解码奠基论文之一，rejection sampling 正确性证明的来源 |
| chen2023_speculative_sampling_deepmind.pdf | 2302.01318 | Accelerating LLM Decoding with Speculative Sampling (DeepMind) | 另一篇奠基论文，70B Chinchilla 上 2-2.5x 加速的实证参考 |
| li2024_eagle.pdf | 2401.15077 | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | SOTA 投机解码（特征级 drafting，非独立小模型） |
| li2024_eagle2.pdf | 2406.16858 | EAGLE-2: Faster Inference with Dynamic Draft Trees | 树形草稿 + 动态调整，接受率更高 |
| cai2024_medusa.pdf | 2401.10774 | Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads | 多头并行解码方案，对比基线 |
| survey2025_speculative_decoding.pdf | 2502.19732 | Speculative Decoding and Beyond: An In-Depth Survey | 全景综述，含 SpecInfer/Sequoia 等树形方法 |
| lin2023_awq.pdf | 2306.00978 | AWQ: Activation-aware Weight Quantization | 量化方案一：基于激活值统计的逐通道缩放 |
| frantar2023_gptq.pdf | 2210.17323 | GPTQ: Accurate Post-Training Quantization | 量化方案二：基于二阶 Hessian 信息的重构 |
| xiao2023_smoothquant.pdf | 2211.10438 | SmoothQuant: Accurate and Efficient PTQ for LLMs | W8A8（权重+激活）量化，用于对比权重量化方案 |
| kwon2023_vllm_pagedattention.pdf | 2309.06180 | Efficient Memory Management for LLM Serving with PagedAttention (vLLM, SOSP'23) | 服务层 KV cache 管理的架构基础，决定"造轮子 vs 用轮子"的边界 |
| liu2023_agentbench.pdf | 2308.03688 | AgentBench: Evaluating LLMs as Agents (ICLR'24) | Agent workload 评测方法论参考 |
| qin2023_toolllm.pdf | 2307.16789 | ToolLLM: Facilitating LLMs to Master 16000+ APIs | 工具调用任务集设计参考 |

下载脚本：`download.sh`（可重跑以更新到最新版本）。

## 第二批：自适应投机解码控制（2024-2026 前沿研究）

| 文件 | arXiv ID | 论文 | 与项目的关系 |
|---|---|---|---|
| kim2025_gammatune.pdf | 2504.00030 | Token-Driven GammaTune: Adaptive Calibration for Enhanced Speculative Decoding (Texas A&M, 2025) | **核心参考**——训练free的自适应 γ 控制算法，Algorithm 1 可直接实现（指数移动平均 + 自适应扩窗），SpecBench 上平均 15-16% 提速 |
| speckv2026_adaptive_gamma.pdf | 2605.02888 | SpecKV: Adaptive Speculative Decoding with Compression-Aware Gamma Selection | 用 draft 模型自身的置信度/熵信号做 γ 选择的轻量控制器，作为对比方案 |
| adaedl2024_entropy_early_stop.pdf | 2410.18351 | AdaEDL: Early Draft Stopping via Entropy-based Lower Bound | 基于熵的提前停止草稿准则，另一种轻量自适应思路 |
| banditspec2025_bandit_gamma.pdf | 2505.15141 | BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms | 用 UCB 老虎机算法做在线 γ 调优，理论更严谨的对比方案 |
| nightjar2025_dynamic_serving.pdf | 2512.22420 | Nightjar: Dynamic Adaptive Speculative Decoding for LLM Serving | 明确提出"开销大于收益时应自动禁用投机解码"，验证 circuit breaker 设计思路 |

另：UC Berkeley 硕士论文 *TurboSpec*（EECS-2025-224）是这个方向里最接近"生产级闭环控制系统"的工作——用 offline profiling + online feedback 动态调整投机解码参数，正式提出 "goodput" 作为统一指标；未下载全文（校内链接跳转失效），仅供参考方向。

**关键背景事实**：动态 γ 调整已经是 Hugging Face Transformers 的默认行为（从 4.45.0 版本起，Mamou et al. 2024 的方法），不是纯学术玩具——这意味着我们做的不是好高骛远的空想，而是在复现/对比一个已经进了主流生产库的技术，评测时可以直接把 HF Transformers 自带的动态投机解码作为一个真实基线。

**⚠️ 引用错误（2026-08-27 发现，需要修正生成脚本）**：`kim2025_gammatune.pdf` 这个文件名和计划正文里"Kim et al. 2025"的署名是错的。读取 PDF 第一页确认真实作者是 **Aayush Gautam, Susav Shrestha, Narasimha Reddy**（Texas A&M ECE），全文没有姓 Kim 的作者。这条错误目前贯穿 P5.0、附录A.2、附录E 三处，需要统一改成 "Gautam et al. 2025"。

## 第三批：2026-08-27 重新评估补充下载（batch/量化交叉点 + circuit breaker 重叠核查）

| 文件 | arXiv ID | 论文 | 与项目的关系 |
|---|---|---|---|
| qspec2024_complementary_quant.pdf | 2410.11305 | QSpec: Speculative Decoding with Complementary Quantization Schemes | Figure 7 已经画出 AWQ vs FP16 在 batch=8/16/32 下的加速比收窄曲线——和支柱4想画的图基本是同一张，且早两年发表，之前的文献综述漏检了这篇 |
| synergy2023_batching_crossover.pdf | 2310.18813 | The Synergy of Speculative Decoding and Batching in Serving LLMs | 最早系统提出"batch=1时投机解码降63%延迟、batch=32时应缩短γ"的内存带宽↔计算瓶颈切换机制，支柱4的理论基础其实这篇已经讲清楚了 |
| specdec_meets_quant2025.pdf | 2505.22179 | Speculative Decoding Meets Quantization | 另一篇测投机解码和量化方案（W8A8/W4A16/W4A8-QQQ）兼容性的论文，Llama-3-70B+EAGLE-2，支柱2/AWQ这条线上的又一个先例 |
| learning_to_draft2026_rl.pdf | 2603.01639 | Learning To Draft: Adaptive Speculative Decoding with Reinforcement Learning | 已经把 GammaTune 当 baseline 在比——说明简单 EMA 启发式在2026年中已经不是这个方向的前沿，前沿在往 RL-based adaptive drafting 走 |
| lossless_not_free2026_empirical.pdf | 2607.17283 | Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware | 从内存带宽/计算瓶颈切换的机制层面解释了为什么 AWQ（带宽优化）在投机解码里会和 batch size 产生这种交叉效应——给支柱4的理论解释提供更底层的依据 |

下载脚本：`download3.sh`。详细的重叠分析和结论见 [`../notes/相关工作重新评估_2026-08-27.md`](../notes/相关工作重新评估_2026-08-27.md)。
