#!/usr/bin/env python3
"""生成 Specter 两人开发计划 v2 PDF。

内容来源：git 分支 add-two-dev-task-plan 上的 notes/two_developer_plan_v2.md
（该分支已推送至 target/add-two-dev-task-plan，对应 PR #2；本地 main 分支
工作区不含此文件，内容以该分支为准，不得自行改写数字或结论）。
版式引擎：~/.claude/pdf-kit/layout.py（本机固定引擎，见同目录 RULES.md）
视觉/写作规则：timemartdocs 研究报告PDF模板/PDF生成规则.md

运行：
    ~/.claude/venv-pdf/bin/python generate_two_developer_plan_pdf.py
    pdfqa /Users/yyukin0/Documents/Specter/notes/two_developer_plan_v2.pdf
    pdfshots /Users/yyukin0/Documents/Specter/notes/two_developer_plan_v2.pdf
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude" / "pdf-kit"))

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from layout import (
    BODY, CONTENT_W, DISPLAY, LEAD, MICRO, MINT,
    PAGE_H, PAGE_W, PAPER, SLATE_2, SMALL,
    Page, register_fonts,
)

OUT = Path("/Users/yyukin0/Documents/Specter/notes/two_developer_plan_v2.pdf")

MASTHEAD = "SPECTER / TWO-DEVELOPER PLAN"
FOOTER = "SPECTER · TWO-DEV PLAN V2 · DRAFT"


def clean(t: str) -> str:
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", t)
    return t


# --------------------------------------------------------------------------
# 结构化小工具（与 generate_specter_v9_report.py 保持一致，便于风格统一）
# --------------------------------------------------------------------------

def chapter(p: Page, tag: str, title: str, lead: str | None = None) -> None:
    def _open() -> None:
        p.eyebrow(tag)
        p.h1(title)
        if lead:
            p.body(clean(lead))
    h, _ = p.measure(_open)
    p.section_break(title, h + 60)
    _open()


def sub(p: Page, title: str, body_text: str | None = None, size: float = BODY) -> None:
    p.h2(clean(title))
    if body_text:
        p.body(clean(body_text), size=size)


def item(p: Page, lead: str, body_text: str, size: float = BODY) -> None:
    p.body_bold(clean(lead), size=size)
    p.y -= 1
    p.body(clean(body_text), size=size)


def numbered(p: Page, entries: list[str]) -> None:
    for i, e in enumerate(entries, 1):
        p.bullet(clean(f"{i}. {e}"))


# --------------------------------------------------------------------------
# 封面
# --------------------------------------------------------------------------

def draw_cover(p: Page) -> None:
    c = p.c
    fg = PAPER
    muted = SLATE_2

    c.setFont("Body-Bold", MICRO)
    c.setFillColor(fg)
    y = PAGE_H - 64
    x = 42
    for ch in MASTHEAD:
        c.drawString(x, y, ch)
        x += pdfmetrics.stringWidth(ch, "Body-Bold", MICRO) + 1.1

    c.setFont("Body", MICRO)
    c.setFillColor(muted)
    c.drawRightString(PAGE_W - 42, y, "DRAFT · CHALLENGE REVIEW")

    c.setStrokeColor(MINT)
    c.setLineWidth(0.8)
    c.line(42, PAGE_H - 78, PAGE_W - 42, PAGE_H - 78)

    c.setFont("Body-Bold", MICRO)
    c.setFillColor(MINT)
    xx = 42
    for ch in "TWO-DEVELOPER TASK SPLIT · 双人开发计划":
        c.drawString(xx, PAGE_H - 210, ch)
        xx += pdfmetrics.stringWidth(ch, "Body-Bold", MICRO) + 1.0

    lines = p._wrap("Specter 两人开发计划 v2", "Display-Bold", DISPLAY, CONTENT_W)
    ty = PAGE_H - 260
    c.setFont("Display-Bold", DISPLAY)
    c.setFillColor(fg)
    for ln in lines:
        c.drawString(42, ty, ln)
        ty -= DISPLAY * 1.12

    lines2 = p._wrap(
        "把 project_plan_v9.md 单人 27-28 天路线图拆成两条并行轨道 —— "
        "developer A / developer B 的任务认领、并行窗口与里程碑判据",
        "Body", LEAD, CONTENT_W * 0.9,
    )
    ty -= 14
    c.setFont("Body", LEAD)
    c.setFillColor(muted)
    for ln in lines2:
        c.drawString(42, ty, ln)
        ty -= LEAD * 1.4

    meta_y = 150
    from reportlab.lib.colors import Color
    c.setStrokeColor(Color(1, 1, 1, alpha=0.18))
    c.setLineWidth(0.65)
    c.line(42, meta_y + 34, PAGE_W - 42, meta_y + 34)

    cols = [
        ("版本", "v2（草稿，待 challenge review）"),
        ("日期", "2026-08-27"),
        ("状态", "已开 PR #2（分支 add-two-dev-task-plan），未合并"),
        ("来源", "project_plan_v9.md §7/§11/§16 任务级依赖拆分"),
    ]
    col_w = CONTENT_W / 2
    for i, (k, v) in enumerate(cols):
        cx = 42 + (i % 2) * col_w
        cy = meta_y - (i // 2) * 52
        c.setFont("Body-Bold", MICRO)
        c.setFillColor(MINT)
        c.drawString(cx, cy, k.upper())
        c.setFont("Body", SMALL)
        c.setFillColor(fg)
        for ln in p._wrap(v, "Body", SMALL, col_w - 10):
            cy -= 13
            c.drawString(cx, cy, ln)

    c.setFont("Body", MICRO)
    c.setFillColor(muted)
    c.drawString(42, 40, FOOTER)
    c.drawRightString(PAGE_W - 42, 40, "01")


# --------------------------------------------------------------------------
# §0 v2 相比 v1 改了什么
# --------------------------------------------------------------------------

def sec_changes(p: Page) -> None:
    chapter(
        p, "§0 / WHAT CHANGED IN V2", "v2 相比 v1 改了什么",
        "针对用户两条 challenge 的具体修正——v1 把 P4/P5.2/P5.4 整体标成"
        "“联合任务”，掩盖了它们内部其实能再拆出并行子任务；v1 的里程碑只有一张模糊的"
        "ASCII 时间线。",
    )
    rows = [
        ("“没看到哪里能并行”",
         "v1 把 P4、P5.2、P5.4 直接标“联合任务”，掩盖了各自内部其实有能拆开独立"
         "执行的子任务",
         "§2/§3 拆到子任务粒度：P4 拆成“投机解码曲线”vs“量化曲线”两条独立曲线；"
         "P5.2 拆成 AWQ 臂 vs BnB 臂；P5.4 拆成 HF 基线 vs BanditSpec 基线——"
         "每一对都能两人同时做，只在最后合并结果"),
        ("“milestone 没写详细”",
         "v1 只有一张模糊的 ASCII 时间线",
         "§4 换成里程碑表：每个里程碑标 Day 数、交付物、验收标准、负责人、"
         "前置依赖"),
    ]
    p.table(
        [("你的 challenge", 100), ("v1 的问题", 195), ("v2 的修正", CONTENT_W - 100 - 195)],
        rows, allow_split=True,
    )


# --------------------------------------------------------------------------
# §1 现状评估
# --------------------------------------------------------------------------

def sec_status(p: Page) -> None:
    chapter(p, "§1 / STATUS", "现状评估（沿用 v1，未改动实质内容）")
    sub(p, "1.1 计划本身", (
        "project_plan_v9.md 已过架构可行性评审（v9），9 个关键决策点已落定为 "
        "ADR（contracts/decisions/）。按 §17，正式进入阶段0仍需用户明确授权——"
        "两人协作下变成“两人都要认可”。"
    ))
    p.h2("1.2 代码现状（会改变排期起点）")
    p.body(clean(
        "Michael8964/Specter 目前只有骨架 + 待合并的 PR #1（notes/papers 迁移）。"
        "但本地已有跑通的 P1.0 结果，还没进任何仓库："
    ))
    rows = [
        ("src/gate_p1_0.py（206行）", "HF Transformers 实现的 P1.0 gate"),
        ("src/gate_p1_0_mlx_crosscheck.py（124行）", "mlx-lm 交叉验证"),
        ("src/results/p1_0_gate_result.json",
         "draft=Qwen2.5-0.5B / target=Qwen2.5-3B，vocab_match=true，"
         "overall_alpha=0.7024（PASS）"),
        ("src/results/p1_0_mlx_crosscheck_result.json", "overall_alpha=0.5591，同量级"),
    ]
    p.table([("文件", 210), ("内容", CONTENT_W - 210)], rows, allow_split=True)
    p.y -= p.lead_in(BODY, gap=10)
    item(
        p, "阶段0实质已完成",
        "只差迁移+第二人复核。建议排期从“迁移+联合复核（0.5天）”开始，而不是"
        "从零开始的 P1.1。",
    )


# --------------------------------------------------------------------------
# §2 任务级依赖关系
# --------------------------------------------------------------------------

DEP_ROWS = [
    ("P1.1-P1.4", "draft/target模型", "否",
     "“标准算法：草稿模型自回归生成γ个候选…目标模型单次前向打分”——纯双模型"
     "循环，不涉及量化或评测框架"),
    ("P2.0-P2.3", "target模型 + 校准集", "否",
     "“在校准集上跑前向，收集每层激活值的逐通道统计量”——不涉及投机解码循环"),
    ("P2.4（Mac）", "target模型", "否",
     "mlx_lm.awq 是独立工具链，“额外可产出真实的Mac端AWQ速度/内存数字”"),
    ("P3.0", "无（设计阶段）", "否",
     "“15-20个bash/工具调用任务，改编自AgentBench OS子环境”——任务清单设计"
     "不需要跑引擎"),
    ("P3.1", "P1的可运行引擎", "是（需A）",
     "“同一模型对，工具调用/结构化输出场景vs自由文本对话场景，对比α差异”——"
     "α测量必须有能跑的投机解码"),
    ("P5.0-P5.1", "P1的可运行引擎", "是（需A自己产出，非B）",
     "“接受数A与当前γ相等时扩窗”——直接操作P1循环内部状态，不需要B的量化产出"),
    ("P5.3", "P1+P5.0的产出", "是（需A自己产出）",
     "“高batch时自动降级/禁用投机解码”——建立在P5.0的γ控制逻辑上"),
    ("P4 投机解码曲线", "P1的引擎", "是（需A）",
     "§7 P4.0“投机解码…各测一条吞吐提升曲线”——这一条曲线只需要P1，不需要"
     "量化模型"),
    ("P4 量化曲线", "P1的引擎 + B的AWQ模型", "是（需A的引擎+B自己的模型）",
     "同上，“量化…测一条吞吐提升曲线”——这一条曲线B可以独立跑，只是需要A的"
     "引擎已经存在（不需要A本人参与）"),
    ("P5.2 AWQ臂", "P5.0的控制器代码 + B的AWQ模型",
     "需要A已完成的控制器代码（不需要A本人同时在场）",
     "§7 P5.2“复用P2的量化模型…对同一draft/target组合分别扫描最优γ”"),
    ("P5.2 BnB臂", "P5.0的控制器代码 + BnB配置（HF内置）", "同上",
     "ADR-008：“几乎零成本（HF内置配置）”"),
    ("P5.4 HF双基线", "A已有的测试harness", "否（A独立可做）",
     "“HF Transformers num_assistant_tokens_schedule/assistant_confidence_"
     "threshold”——都是配置项，A的现有代码改几行配置"),
    ("P5.4 BanditSpec基线", "无（独立代码库）", "否（B独立可做）",
     "“BanditSpec公开代码…直接克隆运行”——完全独立的第三方代码库，不需要A的"
     "产出"),
]


def sec_dependency(p: Page) -> None:
    chapter(
        p, "§2 / TASK-LEVEL DEPENDENCIES", "任务级依赖关系",
        "之前 v1 只分析到“支柱”层级，容易把内部可拆分的任务误判成不可拆。"
        "这次逐个 P-子任务核对依赖：",
    )
    p.table(
        [("子任务", 95), ("实际输入", 110), ("是否需要另一方产出", 90),
         ("判断依据（引用§7原文）", CONTENT_W - 95 - 110 - 90)],
        DEP_ROWS, allow_split=True,
    )
    p.y -= p.lead_in(BODY, gap=10)
    item(
        p, "关键结论（回应你的第一条challenge）",
        "真正意义上“必须两人同时坐在一起改同一段代码”的任务其实不存在。所谓"
        "“联合任务”绝大部分是“两人各自独立产出一部分，最后合并对比”——这是"
        "可以并行的，只是需要一个共享的前置产出（比如“A把P1引擎跑通”）作为"
        "解锁条件，而不是全程绑在一起。唯一真正需要坐在一起讨论的是“合并/对比"
        "结果”那一步（找batch交叉点、对比γ偏移量级），这一步天然很短"
        "（通常<1天）。",
    )


# --------------------------------------------------------------------------
# §3 并行窗口一览表
# --------------------------------------------------------------------------

WINDOW_ROWS = [
    ("W0", "Day 0-0.5", "联合复核P1.0已有结果", "同左", "联合（非并行，但只需0.5天）", "见§1.2"),
    ("W1", "Day 0.5-6", "P1.1-P1.4（核心投机解码，7天中的前5.5天）", "P2.0-P2.3（全部5.5天）",
     "真并行", "互不依赖（§2）"),
    ("W2", "Day 6-7.5", "继续P1.1-P1.4（还剩1.5天）",
     "空档→P2.4-Mac交叉验证（1d）+P3.0任务设计启动（0.5d）",
     "B独立填充，不算联合", "P2.4/P3.0设计都不需要A"),
    ("W3", "Day 7.5-10", "P5.0-P5.1（3天，只需P1，不需要B）", "P3.0收尾（1.5d）+P3.1（1d）",
     "真并行", "A解锁条件是“自己的P1完成”；B解锁条件是“A的引擎存在”，两人此后"
     "各自独立推进"),
    ("W4", "Day 10-10.5", "继续P5.0-P5.1", "空档（0.5d）→提前写P4量化曲线的脚本",
     "B独立填充", "不需要A参与"),
    ("W5", "Day 10.5-12.5", "P5.3（2天，只需P5.0，不需要B）",
     "P4量化曲线（1d）+云端P2.4的本地dry-run脚本准备（1d，§10要求）",
     "真并行", "两人各自独立产出，互不阻塞"),
    ("W6", "Day 12.5-13.5", "P4投机解码曲线（1天，只需自己的P1引擎）",
     "（已在W5提前完成量化曲线，此时空档或帮A做数据整理）",
     "部分并行", "A是这个窗口唯一的阻塞方"),
    ("W7", "Day 13.5-14", "联合：合并两条batch曲线找交叉点，写P4.1记录", "同左",
     "联合（0.5天）", "唯一必须“坐在一起”的环节之一"),
    ("W8", "Day 14-14.5", "联合：云端启动会（预算记账方式、GPU实例数量决策——见§5）", "同左",
     "联合（0.5天）", "决策点，非执行"),
    ("W9", "Day 14.5-16.5（若2个GPU实例）", "P5.4 HF双基线（1d）+P4.2投机解码曲线（1d）",
     "P2.4云端GPTQ速度（1d）+P4.2量化曲线（1d）",
     "真并行（需2个GPU实例，见§5）", "若只租1个实例，这一列变成排队执行，见§5"),
    ("W10", "Day 16.5-18（若2个GPU实例）", "P5.2 AWQ臂（1d）",
     "P5.2 BnB臂（0.5d）+P5.4 BanditSpec基线（1d）",
     "真并行（同上前提）", ""),
    ("W11", "Day 18-19", "联合：合并P4.2曲线+P5.2两臂结果+held-out集最终跑一次", "同左",
     "联合（1天）", "计划要求held-out只跑一次（§9.6风险1），必须两人都在场"
     "确认这是“唯一一次”"),
    ("W12", "Day 19-21", "联合：README+简历bullet（阶段7）", "同左", "联合（2天）", ""),
]


def sec_windows(p: Page) -> None:
    chapter(p, "§3 / PARALLEL WINDOWS", "并行窗口一览表")
    p.table(
        [("窗口", 34), ("Day范围（估计）", 92), ("A 在做什么", 120),
         ("B 在做什么", 120), ("是否真并行", 62),
         ("解锁条件/备注", CONTENT_W - 34 - 92 - 120 - 120 - 62)],
        WINDOW_ROWS, allow_split=True,
    )
    p.y -= p.lead_in(BODY, gap=10)
    item(
        p, "并行窗口小计",
        "W1/W3/W5/W6/W9/W10 六个窗口是两人各自做不同事情、互不等待的真并行"
        "区间，覆盖了从 Day 0.5 到 Day 18 里的大部分时间；真正“必须两人一起做"
        "同一件事”的只有 W0、W7、W8、W11、W12，加起来约 4.5 天。",
    )
    item(
        p, "总工期估计",
        "约 19-21 天（云端若租2个GPU实例；若只租1个，见§5，云端阶段拉长约"
        "1.5-2天，总工期约21-23天）。这仍是排期估计，不是承诺值。",
    )


# --------------------------------------------------------------------------
# §4 里程碑表
# --------------------------------------------------------------------------

MILESTONE_ROWS = [
    ("M0", "Day 0.5", "P1.0结果迁移进Michael8964/Specter + 双人复核记录",
     "两人都读过gate_p1_0.py核心逻辑，无异议", "A+B联合", "无"),
    ("M1", "Day 6", "P2.0-P2.3完成",
     "perplexity涨幅数字产出；跨分布矩阵产出；校准集消融产出（§11阶段2）", "B", "M0"),
    ("M2", "Day 7.5", "P1.1-P1.4完成",
     "贪心100%一致；采样α与理论公式吻合；bonus token测试通过；验证器故障"
     "注入测试通过；γ扫描曲线产出（§11阶段1）", "A", "M0"),
    ("M3", "Day 10", "P3.0-P3.1完成",
     "15-20任务跑通（含3-5个held-out）；结构化vs自由文本α对比数字产出"
     "（§11阶段3）", "B", "M2"),
    ("M4", "Day 12.5", "P5.0-P5.1-P5.3完成",
     "GammaTune复现论文量级提速（3次跑均值±标准差）；波动场景测试产出；"
     "熔断器重探测机制实现（§11阶段4）", "A", "M2"),
    ("M5", "Day 14", "P4.0-P4.1完成（Mac部分双曲线已合并）",
     "交叉点初步曲线产出；显存占用数据产出；mlx_lm.awq真实Mac AWQ速度"
     "数字产出（§11阶段5）", "A+B（各自曲线+联合合并）", "M1+M2"),
    ("M6", "Day 14.5", "云端启动会纪要（预算记账方式+GPU实例数量决定）",
     "两人书面确认，见§5", "A+B联合", "M5"),
    ("M7", "Day 18", "P2.4云端补全+P4.2+P5.2（AWQ+BnB双臂）完成",
     "LLM Compressor对比数字；vLLM对比数字；AWQ量化-γ偏移结论+BnB同源"
     "对照结论（§11阶段6前半）", "A+B（各自执行）", "M6"),
    ("M8", "Day 19", "P5.4三基线对比+held-out最终确认",
     "HF双基线+BanditSpec代码对比数字；held-out集跑一次最终确认"
     "（§11阶段6后半）", "A+B联合（合并环节）", "M7"),
    ("M9", "Day 21", "GitHub repo定稿+README+简历bullet", "§11阶段7完成判据",
     "A+B联合", "M8"),
]


def sec_milestones(p: Page) -> None:
    chapter(p, "§4 / MILESTONES", "里程碑表")
    p.table(
        [("里程碑", 40), ("预计完成日", 62),
         ("交付物", 130), ("验收标准（对应§11完成判据）", 165),
         ("负责人", 62), ("前置依赖", CONTENT_W - 40 - 62 - 130 - 165 - 62)],
        MILESTONE_ROWS, allow_split=True,
    )


# --------------------------------------------------------------------------
# §5 GPU 实例数量约束
# --------------------------------------------------------------------------

def sec_gpu(p: Page) -> None:
    chapter(
        p, "§5 / GPU INSTANCE CONSTRAINT", "关键约束：云端阶段的并行程度取决于 GPU 实例数量",
        "这是 v1 没有点出的一个硬约束：Mac 阶段两人各用自己的 Mac，天然物理"
        "并行；但云端阶段如果只租一个 GPU 实例，A 和 B 没法同时用它跑各自的"
        "实验——W9/W10 里标的“真并行”会退化成排队执行，云端阶段总时长不会"
        "缩短，只是两人交替上手。",
    )
    rows = [
        ("租1个GPU实例（原计划默认，$30-50预算按1份算）",
         "云端阶段仍需排队，约4-5天，两人协作只省了“谁先谁后”的调度成本", "预算不变"),
        ("租2个GPU实例（W9/W10表格里假设的情况）",
         "云端阶段真并行，约3天完成",
         "预算翻倍风险——§5护栏指标的$50硬顶是按单实例场景定的，两个实例"
         "大概率突破$50"),
    ]
    p.table(
        [("方案", 190), ("效果", 175), ("代价", CONTENT_W - 190 - 175)],
        rows, allow_split=True,
    )
    p.y -= p.lead_in(BODY, gap=10)
    item(
        p, "待共同确认",
        "这是计划里完全没写过的新决策点，两人需要自己定——大概率结论应该是"
        "租1个实例、接受云端阶段排队执行，因为$50硬顶（护栏指标）大概率经"
        "不起双实例开销；如果要租2个，需要重新和用户确认预算上限是否还是"
        "$50。我不替你们做这个决定，但把两个选项的代价都摆出来了。",
    )


# --------------------------------------------------------------------------
# §6 任务认领表
# --------------------------------------------------------------------------

CLAIM_ROWS = [
    ("P1.0", "前置gate", "已完成，待联合复核", "无", "§7 P1.0"),
    ("P1.1-P1.4", "投机解码核心+验证", "A", "P1.0", "§7 P1.1-1.4"),
    ("P2.0-P2.3", "AWQ校准+正确性", "B", "无", "§7 P2.0-2.3"),
    ("P2.4（Mac）", "mlx_lm.awq交叉验证", "B（可提前）", "无", "§7 P2.4"),
    ("P3.0", "AgentBench任务设计", "B（可提前）", "无", "§7 P3.0"),
    ("P3.1", "接受率对比实验", "B", "P1.1-1.4", "§7 P3.1"),
    ("P5.0-P5.1", "GammaTune算法+波动测试", "A", "P1.1-1.4", "§7 P5.0-P5.1"),
    ("P5.3", "Batch-aware熔断器", "A", "P5.0", "§7 P5.3"),
    ("P4（投机解码曲线）", "batch扫描-非量化侧", "A", "P1.1-1.4", "§7 P4.0"),
    ("P4（量化曲线）", "batch扫描-量化侧", "B", "P1.1-1.4 + P2.1", "§7 P4.0"),
    ("P4.1", "显存占用记录", "A+B各记自己那条曲线", "同上", "§7 P4.1"),
    ("P4（合并）", "找交叉点，联合写记录", "联合", "P4两条曲线都完成", "§7 P4.0-4.1"),
    ("P2.4（云端）", "LLM Compressor GPTQ速度", "B", "P2.1", "§7 P2.4"),
    ("P4.2（投机解码曲线，云端）", "大模型batch验证-非量化侧", "A", "P4（合并）", "§7 P4.2"),
    ("P4.2（量化曲线，云端）", "大模型batch验证-量化侧", "B", "P4（合并）", "§7 P4.2"),
    ("P5.2（AWQ臂）", "量化-γ耦合，AWQ侧", "A（用自己的控制器）+B的AWQ模型",
     "P5.0 + P2.1", "§7 P5.2"),
    ("P5.2（BnB臂）", "量化-γ耦合，BnB对照", "B（HF配置）+A的控制器代码",
     "P5.0 + ADR-008", "§7 P5.2"),
    ("P5.4（HF双基线）", "HF内置两种基线对比", "A", "P5.0-5.3", "§7 P5.4"),
    ("P5.4（BanditSpec基线）", "克隆跑第三方代码", "B", "无（独立代码库）", "§7 P5.4"),
    ("held-out最终跑一次", "全部超参数定稿后跑一次", "联合（必须双人见证，§9.6风险1）",
     "所有超参数定稿", "§9.6风险1"),
    ("阶段7产出", "README+简历bullet", "联合", "全部完成", "§11阶段7"),
]


def sec_claims(p: Page) -> None:
    chapter(p, "§6 / TASK CLAIMS", "任务认领表（细到子任务，可直接勾选）")
    p.table(
        [("P-任务/子任务", 100), ("内容", 115), ("建议归属", 130),
         ("前置依赖", 90), ("计划出处", CONTENT_W - 100 - 115 - 130 - 90)],
        CLAIM_ROWS, allow_split=True,
    )


# --------------------------------------------------------------------------
# §7 协作纪律 / §8 开放问题 / §9 版本记录
# --------------------------------------------------------------------------

def sec_discipline(p: Page) -> None:
    chapter(p, "§7 / COLLABORATION DISCIPLINE", "协作纪律（沿用v1，未改动）")
    numbered(p, [
        "接口边界：B的AWQ模型要能被A的引擎直接加载，建议两条轨道各自进行到"
        "一半时（约Day3-4）先对一次接口形状。待共同确认：具体接口形式计划"
        "里没写。",
        "云端预算记账：见§5，现在多了“租几个实例”这个新决策，$50硬顶怎么"
        "在双实例场景下重新界定，待共同确认。",
        "Git工作流：延续直接在Michael8964/Specter开分支+PR、不fork的模式。"
        "待共同确认：分支命名约定（建议a/、b/前缀）。",
        "P1.0复核不能省：进入P1.1前，B至少读一遍gate_p1_0.py核心逻辑，"
        "确认没有假阳性风险（contracts/research-integrity.md风险3的精神）。",
    ])


def sec_open_questions(p: Page) -> None:
    chapter(p, "§8 / OPEN QUESTIONS", "开放问题")
    numbered(p, [
        "project_plan_v9.md §16 原有5条开放问题依然成立，现在需两人共同决定。",
        "云端GPU实例数量（§5，v2新增，最重要的一条新开放问题）。",
        "云端账号/计费主体由谁持有，如何同步进度。",
        "contracts/里的ADR/坑表是单人视角写的，两人协作后谁负责在跨轨道决策"
        "变化时同步更新（比如中途换draft/target模型对）。",
    ])


def sec_changelog(p: Page) -> None:
    chapter(p, "§9 / VERSION LOG", "版本记录")
    item(
        p, "v1",
        "基于project_plan_v9.md依赖图拆两轨道，发现本地已有P1.0结果未纳入"
        "排期，计入§1.2；但把P4/P5.2/P5.4整体标“联合任务”，掩盖了内部并行"
        "空间。",
    )
    item(
        p, "v2（本版本）",
        "针对“没看到哪里能并行”的challenge，把P4/P5.2/P5.4拆到子任务粒度，"
        "识别出6个真并行窗口（§3）；针对“milestone不详细”的challenge，新增"
        "里程碑表（§4）；新发现一个v1完全没提过的约束——云端阶段的并行程度"
        "取决于GPU实例数量，且直接牵动$50预算护栏（§5）。等待下一轮challenge"
        "review。",
    )


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main() -> None:
    register_fonts(cjk=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    p = Page(c, masthead=MASTHEAD, footer=FOOTER, cover_first=True)
    p.start("", dark=True)
    draw_cover(p)

    p.start("What Changed In V2", dark=False)
    sec_changes(p)
    sec_status(p)
    sec_dependency(p)
    sec_windows(p)
    sec_milestones(p)
    sec_gpu(p)
    sec_claims(p)
    sec_discipline(p)
    sec_open_questions(p)
    sec_changelog(p)

    c.showPage()
    c.save()
    print(f"wrote {OUT} ({p.page_no} pages)")


if __name__ == "__main__":
    main()
