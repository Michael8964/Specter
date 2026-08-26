# Prior Art / 生产实践经验调研

目的：在写一行代码之前，先搞清楚别人已经把这条路走到哪了、踩过什么坑，避免重复造轮子或重复踩坑。这份笔记补充 papers/ 里的学术论文——这里全是工程博客、官方文档、生产复盘，代表"理论之后发生了什么"。

---

## 一、生产成熟度现状（2025-2026）

投机解码在 2025 年已经从"论文里的技巧"变成主流推理框架的标配：
- **vLLM**：原生支持 EAGLE / EAGLE-3 / Medusa / n-gram 草稿，一个 `--speculative-config` flag 就能开
- **SGLang**：`--speculative-algorithm EAGLE3`，文档直接给自动调参建议
- **TensorRT-LLM**：NVIDIA 官方在 H200 上演示 3.6x 吞吐提升
- **llama.cpp**：支持经典两模型方案 + n-gram 自投机，EAGLE-3 支持还在社区讨论中（Issue #15902），目前落后于 vLLM/SGLang/TRT-LLM

**这对我们项目的意义**：EAGLE3 现在"部署"只是改一个 flag，这恰好说明"调库"本身已经没有技术含量了——真正有区分度的是我们计划做的事：手写实现 + 数学/统计正确性验证 + 系统性的边界实验（什么时候有效、什么时候失效），这是调 flag 得不到的东西。

来源：[vLLM Blog](https://vllm.ai/blog/tags/speculative-decoding)、[SGLang 文档](https://docs.sglang.ai/advanced_features/speculative_decoding.html)、[llama.cpp docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

---

## 二、IBM/PyTorch 官方生产复盘（最重要的一手资料）

来源：[A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)（PyTorch 官方博客，2024，IBM Research 内部生产环境实测，服务数千日活用户）

**实测数字**（比论文里的实验室数字更可信，因为是真实生产流量）：
- Llama2-13B、Llama3-8B、IBM Granite-7B：**2x** 提速
- IBM Granite-20B 代码模型：**3x** 提速

**关键工程经验**：
1. **代码模型用更多 draft token 更划算**：语言任务用 3-4 个 speculative head 效果最好，代码模型能用到 6-8 个（代码的模式更可预测，接受率更高，值得多"赌"几个 token）。这和 llama.cpp 社区的独立观察一致："生成可预测文本（代码/JSON/工具调用）时投机解码接近于免费加速"。**→ 直接支撑我们选 agent 工具调用场景做评测的选择：这不只是贴题，还是能让效果最好看的场景。**
2. **batch size > 64 时开始出现吞吐量下降**——这是三个独立来源（IBM/PyTorch、SGLang 团队关于 EAGLE 在 batch=24 掉速/EAGLE-3 在 batch=64 仍有效、AutoAWQ 官方文档关于高 batch 量化反而变慢）**共同指向的同一个系统原理**：投机解码和量化本质上都是"用空闲算力换内存带宽"的优化，只在内存带宽瓶颈（低 batch/单用户延迟场景）下有效，一旦进入计算瓶颈（高 batch/高吞吐场景）就会失效甚至反向。这个"batch size 交叉点"现象值得做成我们项目里的**核心实验和图表**，而不是顺带一提的观察——它把量化和投机解码两条线在系统层面统一了起来，是全项目里含金量最高的一张图。
3. **他们对比过"独立小模型做 draft" vs "给原模型加 speculative head"两种架构，发现加 head 的方案在效果和延迟上都更好**——但加 head 需要专门训练（他们用 PyTorch FSDP 训了两阶段）。**对我们的启示**：我们选经典两模型方案是因为没有训练预算和多卡资源，这是一个基于资源约束的合理取舍，但报告里要诚实写清楚"生产界的最优解其实是训练专用 speculative head，我们受限于预算选择了可行性更高但效果稍弱的两模型方案"——这种诚实的边界说明比不懂装懂更加分。
4. **数值精度**：不同设备/框架间不保证 bit-exact 输出，"精确对比时应该用贪心解码而不是采样"。**→ 验证方法确定为：贪心模式做严格的逐 token 对照（应该完全一致或只有可忽略的浮点误差），采样模式做统计层面/任务指标层面的验证（这在 v2 计划里已经定了，这里是生产界的独立佐证）。**
5. **方法论：先测接受率/接受长度，再测吞吐**——一位实践者提到测到接受长度 τ=1.81 就能预判这次部署会掉速，不用等跑完整套 benchmark。**→ 把"测 α/接受率"作为我们 Phase 1 的第一个检查点（gate），而不是等实现完了才发现选错了模型对。**

---

## 三、量化的生产经验补充

- **AutoAWQ 官方文档**自己承认：W4A16（权重 4-bit，激活仍 fp16）量化模型在**高 batch（计算瓶颈）场景下不会提速，甚至因为反量化开销变慢**——量化只在低 batch/内存瓶颈场景（比如本地单用户 agent）有效。这和上面第 2 点的"batch size 交叉点"是同一个原理的另一个例证。
- W4A16 适合"延迟优先"场景（我们的本地 agent 场景），W8A8（如 SmoothQuant）适合"吞吐优先"的多租户服务场景——进一步确认我们选 AWQ（W4A16）风格量化是符合项目定位的正确选择，不是随便选的。
- 行业已经在从分裂的 AutoGPTQ/AutoAWQ 工具链走向统一的 **LLM Compressor**（vLLM 团队做的，统一 GPTQ/AWQ/FP8 quantization API，原生对接 vLLM）。**→ 我们做"现成库对比基线"时用 LLM Compressor 而不是分开装 AutoGPTQ/AutoAWQ，更贴近当前生产标准，报告里提一句也显得紧跟前沿。**

来源：[AutoAWQ GitHub](https://github.com/casper-hansen/AutoAWQ)、[Medium: A practical guide to INT4 quantization for SLMs](https://medium.com/data-science-at-microsoft/a-practical-guide-to-int4-quantization-for-slms-gptq-vs-awq-olive-and-real-world-results-2f63d6963d1d)

---

## 四、可参考代码结构的教学级开源实现

不直接抄，但用来对照自己的实现是否漏了边界情况：
- [romsto/Speculative-Decoding](https://github.com/romsto/Speculative-Decoding)：干净地实现了 Leviathan et al. 2023 算法，包含经典自回归、beam search、投机解码三种生成策略的对照实现，适合用来对拍（sanity check）自己的实现输出是否合理
- [foundation-model-stack/fms-extras](https://github.com/foundation-model-stack/fms-extras)：IBM 生产代码的开源部分，展示了他们怎么改 vLLM 的 PagedAttention kernel 来支持多 head 的 KV cache 管理而不重复存储——即使我们不实现 head 方案，这个仓库也能帮助理解"投机解码 + KV cache 管理"结合时的工程难点
- [Pramodith Dissects 博客+notebook](https://pramodith.github.io/posts/speculative-decoding/)：用 SmolLM2-360M 做 draft、SmolLM2-1.7B 做 target 手写实现贪心投机解码，模型选择思路（同系列不同尺寸）和我们计划里 Qwen2.5 系列的思路一致，可以直接参考它的 α 值/加速比量级做 sanity check

---

## 五、这份调研对 v2 计划的具体修订（→ v3）

1. **新增核心实验**："batch size 交叉点"实验——分别对投机解码和量化在 batch ∈ {1, 4, 8, 16, 32, 64} 下测速度提升，找到从"有效"变"失效/反向"的临界点。这一张图能同时服务两条支柱，把项目从"两个独立技巧的堆砌"提升为"一个系统原理的两个例证"，是含金量提升最大的一处修改。
2. **验证方法论最终定稿**：贪心解码做逐 token 严格比对（生产界共识：这是唯一能做 bit-level 比较的模式），采样模式做统计/任务指标级验证（v2 已定，这里补充生产界证据）。
3. **Phase 1 增加前置 gate**：实现投机解码算法主体之前，先测 draft/target 组合的 α（接受率），α 明显偏低（比如 <0.4）就先换模型对，不要等实现完整个 pipeline 才发现选错了模型——这是 v2 里"存活性检查"的具体操作化。
4. **量化对比基线工具换成 LLM Compressor**（而不是分开的 AutoGPTQ + AutoAWQ），更贴近当前生产标准。
5. **报告里新增一段诚实的边界说明**：生产界最优解（训练专用 speculative head/EAGLE）比我们选的经典两模型方案效果更好，我们的选择是"预算和时间约束下的合理取舍"，不是不知道更优方案的存在——这个坦诚的对比反而比"假装自己做的是 SOTA"更有说服力。
