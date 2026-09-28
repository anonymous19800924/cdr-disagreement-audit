# -*- coding: utf-8 -*-
"""
协议对称实验框架 —— 支持 5 个协议在同一 token 预算下对照
=========================================================
设计目标（解决论文的 Protocol Symmetry Gap，报告 §6.3 痛点 4）：
  所有协议共用同一套：预算、输出结构、轮数、并发、重试、缓存、评测口径。
  只允许「交互机制」不同。这样 Δ 才能归因到机制，而不是 prompt 工程。

协议:
  naive       无顾问，判官直答（下界）
  double      两顾问独立论证同时给判官（无交互）
  debate      对抗式，立场固定不可改（论文 baseline 的对称版）
  dr          论文原版 DR
  placebo     ★ 安慰剂对照：要求复述+格式对齐，但不识别 crux、不判定 winner
  dr_nocrux   ★ 消融：多轮改答案，但不要求识别 crux
  dr_noconf   ★ 消融：去掉 winner 标签，只给原始论证

用法:
  cp .env.example .env   # 填入 OPENROUTER_API_KEY
  python proto_run.py --protocol dr --dataset GPQA --pair gpt4o_claude --judge openai/gpt-4o-mini --limit 3 --dry-run
  python proto_run.py --protocol dr --dataset GPQA --pair gpt4o_claude --judge openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

_HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CDR_ROOT", _HERE.parent))
CODE = ROOT / "code"
DATA = CODE / "data"
PROMPTS = CODE / "prompts"
OUT = ROOT / "out"
CACHE = ROOT / "logs" / "cache"

# ---------------- 统一预算（所有协议共用，保证对称） ----------------
# ⚠️ 关键：智谱 GLM 系是「思维链模型」，reasoning_tokens 计入 completion，
#    且 content 与 reasoning_content 分离。实测 max_tokens=800 时真实任务
#    completion=259（含 reasoning 95），若沿用 800 会在长任务上把 content 饿死为空。
#    → 因此 provider 级预算必须分开：reasoning 模型给足余量，非 reasoning 模型保持原值。
BUDGET = dict(
    max_turns=5,            # 最多交互轮数
    max_tokens=800,         # 每次生成上限（沿用论文，保证可比性）
    temperature=1.0,
    top_p=1.0,
    words_limit=150,        # 单次论证词数上限
    retries=5,              # ★ 429 限流需要更多重试次数（原 3 次不够）
    timeout=1200,           # ★ 2026-09-26：升级 max_tokens 后 1 万+ token 生成需 10min+，
                            #   480s 反成瓶颈（I1 剩 5 题/G1 剩 3 题全是超时）；不进缓存键，零缓存影响
)

# 各 provider 的补充预算（不改变 BUDGET 的论文可比性，只是让推理模型能出答案）
PROVIDER_BUDGET = {
    # ★ 2026-09-24 修正：原 max_tokens=800 是「非推理模型」的默认值，但
    #   OpenRouter 轨实际承载 DS / QWEN / KIMI 这些**推理模型**——实测
    #   deepseek-v4.1-flash 单次 CoT 即可吃满 800，导致 finish=length 且
    #   content 为空（冒烟测试 3/4 题种子失败）。与 alibaba/zhipu 对齐预算。
    "openrouter": dict(max_tokens=8192, min_answer_tokens=1024, workers=4, backoff=5),
    # 推理模型：思维链吃 token，需放大上限。
    # ★ 并发经实测标定（2026-09-23，真实 GPQA+CoT 负载）：
    #   约束不是 QPS，而是「昂贵请求的在途并发数」—— 短提示词在 workers=16 下 0 报错，
    #   但真实 CoT 调用（单次 58~420s）在 workers=16 下 16 个只成功 7 个（rate=54,quota=15）。
    #   workers=2 的旧值过于保守：短调用实测吞吐 1.25 req/s，而 workers=4 可达 7.2 req/s（5.8x）。
    #   → 取 workers=4 作为「贵调用」的稳妥值；配合去重缓存，重复样本不产生请求。
    "zhipu":      dict(max_tokens=8192, min_answer_tokens=1024, workers=4, backoff=5),
    # ★ 阿里云百炼 MaaS（OpenAI 兼容）：可访问 261 个模型、跨 6+ 独立家族
    #   （Qwen / GLM / Kimi / DeepSeek / MiniMax / StepFun）——**这是打破「同源污染」的关键**
    #   实测（2026-09-23）：10 个模型可用，且 reasoning_content 与 content 同样分离
    "alibaba":    dict(max_tokens=8192, min_answer_tokens=1024, workers=4, backoff=5),
}

# ★ 思维链控制（实测：reasoning_len 可达 13750 字符，会吃光预算导致 content 为空）
#   {"thinking": {"type": "disabled"}} → reasoning=0，速度提升约 7 倍，行为确定
#   方法论决策：分角色处理，而不是一刀切
#     · 顾问 consultant：保留思维链 —— 论文研究的是「论证质量与说服力」，
#       推理能力是核心变量，关掉等于换了一个更弱的模型，会破坏实验有效性
#     · 判官 judge：关闭思维链 —— 论文设定判官「弱于顾问」，
#       关闭推理恰好更贴合「弱判官」设定，且消除空 content 导致的静默失败
#   可用 --think 统一覆盖：on=全部保留 / off=全部关闭 / auto=按角色（默认）
THINK_MODES = ("auto", "on", "off")
NO_THINK = {"thinking": {"type": "disabled"}}

PAIRS = {
    "gpt4o_claude": (("openai/gpt-4o", "gpt_4o.json"), ("anthropic/claude-sonnet-4", "claude.json")),
    "glm_kimi":     (("z-ai/glm-4.6", "glm.json"),      ("moonshotai/kimi-k2-thinking", "kimi.json")),
    # ★ 智谱同源配对：数据文件沿用 glm.json / kimi.json（论文数据本就是这两家模型产出的），
    #   运行时把两边都换成智谱可用模型 —— 这天然构成报告 §6.3 痛点2「同源污染」的对照
    "glm46_glm45":  (("glm-4.6", "glm.json"),           ("glm-4.5-air", "kimi.json")),
    "glm46_glmplus": (("glm-4.6", "glm.json"),          ("glm-4-plus", "kimi.json")),

    # ============ 阿里云 MaaS：异源配对（打破同源污染的关键）============
    # 全部经实测确认可用（2026-09-23，HTTP 200 + 正常生成 + CoT 分离）
    # ★ 论文原始配对的可复现版本：GLM × Kimi —— 用 kimi-k3 替代拿不到的 kimi-k2-thinking
    "glm_kimi3":    (("glm-4.6", "glm.json"),           ("kimi-k3", "kimi.json")),
    # ★ 三方异源：智谱 × DeepSeek（训练数据与对齐路线完全不同）
    "glm_ds":       (("glm-4.6", "glm.json"),           ("deepseek-v4-pro", "kimi.json")),
    # ★ 异源：Qwen × Kimi（阿里 vs 月之暗面）
    "qwen_kimi3":   (("qwen3.8-max", "glm.json"),       ("kimi-k3", "kimi.json")),
    # ★ 异源 + 代差对照：Qwen 旗舰 × DeepSeek 廉价
    "qwen_dsflash": (("qwen3.8-max", "glm.json"),       ("deepseek-v4.1-flash", "kimi.json")),
    # ★ 异源：GLM × MiniMax
    "glm_minimax":  (("glm-4.6", "glm.json"),           ("MiniMax-M2.5", "kimi.json")),
    # ★ 新代同源对照：GLM 4.6 × GLM 5.3（检验同门代差是否优于异源）
    "glm46_glm53":  (("glm-4.6", "glm.json"),           ("glm-5.3", "kimi.json")),
    # ★ 异源：DeepSeek × Qwen 中档
    "ds_qwen27b":   (("deepseek-v4-pro", "glm.json"),   ("qwen3.8-27b", "kimi.json")),

    # ============ flash 全家族（2026-09-24 用户要求控成本切换）============
    # 四个目标型号均已实测 HTTP 200 可生成：
    #   deepseek-v4.1-flash / qwen3.8-flash / kimi-k2.5  -> 阿里云 MaaS
    #   glm-5.3-flash                                    -> 智谱 coding 端点
    #   （★ glm-5.3-flash 在智谱 general 端点返回 429「余额不足」，
    #     只有 coding 端点可用；provider=zhipu 默认即 coding，无需额外设置。）
    # 各家族代表：GLM=glm-5.3-flash, QWEN=qwen3.8-flash,
    #             KIMI=kimi-k2.5,    DS=deepseek-v4.1-flash
    # DS 家族（含 DeepSeek 的全部配对）—— 家族矩阵中此前缺失的一支：
    "ds_glm":       (("deepseek-v4.1-flash", "glm.json"), ("glm-5.3-flash", "kimi.json")),
    "ds_qwen":      (("deepseek-v4.1-flash", "glm.json"), ("qwen3.8-flash", "kimi.json")),
    "ds_kimi":      (("deepseek-v4.1-flash", "glm.json"), ("kimi-k2.5", "kimi.json")),
    # flash 家族内的其余异源配对（备用）：
    "glm53f_qwenf": (("glm-5.3-flash", "glm.json"),       ("qwen3.8-flash", "kimi.json")),
    "glm53f_kimif": (("glm-5.3-flash", "glm.json"),       ("kimi-k2.5", "kimi.json")),

    # ============ directive D（2026-09-24）家族矩阵重划 ============
    # 用户两条 directive 的落地：
    #   ① 同一 family 内，两个 agent 必须用**同一个 base model**
    #      —— 同源靠 model id 成立，不靠 data file（旧的 glm46_glm45 用
    #         glm-4.6 × glm-4.5-air 两个不同模型冒充同源，必须废弃）
    #   ② 跨 family 配对走 MaaS 单端点
    #      —— 例外：GLM 在 MaaS 上不可用（ZHIPU/* 显示「未开通」，
    #         glm-5.3-flash 在 MaaS 目录里根本不存在）。用户裁定 GLM 走
    #         智谱 coding 端点的 glm-5.3-flash，故含 GLM 的对会跨端点；
    #         Client 已按模型自动路由（见 route_provider），无需人工干预。
    # 四族代表：GLM=glm-5.3-flash(智谱) / QWEN=qwen3.8-flash
    #           KIMI=kimi-k2.5 / DS=deepseek-v4.1-flash（后三者均在 MaaS）
    # ★ data file 槽位一律沿用 (glm.json, kimi.json)：题集与 swap 标志因此与
    #   已跑各臂**逐题一致**，新旧结果可直接对比。种子改由
    #   `--seed-source runtime` 现采样（见 seed_runtime），不再受数据文件影响。
    # ---- 同家族臂（两侧同 model id，directive ①）----
    "glm53f_homo":  (("glm-5.3-flash", "glm.json"),       ("glm-5.3-flash", "kimi.json")),
    "qwenf_homo":   (("qwen3.8-flash", "glm.json"),       ("qwen3.8-flash", "kimi.json")),
    "kimif_homo":   (("kimi-k2.5", "glm.json"),           ("kimi-k2.5", "kimi.json")),
    "dsf_homo":     (("deepseek-v4.1-flash", "glm.json"), ("deepseek-v4.1-flash", "kimi.json")),
    # ---- 跨家族臂（4 族全对，C(4,2)=6，directive ②）----
    "xf_ds_qwen":   (("deepseek-v4.1-flash", "glm.json"), ("qwen3.8-flash", "kimi.json")),
    "xf_ds_kimi":   (("deepseek-v4.1-flash", "glm.json"), ("kimi-k2.5", "kimi.json")),
    "xf_ds_glm":    (("deepseek-v4.1-flash", "glm.json"), ("glm-5.3-flash", "kimi.json")),
    "xf_qwen_kimi": (("qwen3.8-flash", "glm.json"),       ("kimi-k2.5", "kimi.json")),
    "xf_qwen_glm":  (("qwen3.8-flash", "glm.json"),       ("glm-5.3-flash", "kimi.json")),
    "xf_kimi_glm":  (("kimi-k2.5", "glm.json"),           ("glm-5.3-flash", "kimi.json")),
}


# ============ 模型 → 端点路由（2026-09-24）============
# 背景：flash 全家族横跨两个端点 —— glm-5.3-flash 只在智谱存在（阿里端返回
#       404 "Model not exist"），其余三家（DS / Qwen / Kimi）在阿里百炼。
#       但 Client 原是「单 provider」设计：--provider 决定整个 episode 走哪个
#       端点。于是 `--pair ds_glm --provider alibaba` 把 glm-5.3-flash 发到阿里
#       → 404 ModelUnavailable → DS 家族 6 个任务里 4 个（F1/F4/F5/F6，含两个
#       held-out）全部空转失败。这解释了「held-out 缺失」的真正来源。
# 规则（保守，刻意不改变既有行为）：
#   1) 模型在当前 --provider 上实测可用 → 保持不变
#      （不动缓存键 ⇒ 已跑过的顾问调用继续命中缓存，不会重跑烧钱）
#   2) 否则若该模型是某一端点的独占模型 → 自动路由到该端点
#   3) 两者都查不到 → 保持 --provider（维持旧行为，让 404 自己暴露问题）
PROVIDER_MODELS: Dict[str, set] = {
    # 智谱 open.bigmodel.cn（coding / general 两端配额独立）
    "zhipu": {
        "glm-4.6", "glm-4.5-air", "glm-4-plus", "glm-4-flash",
        "glm-5.3", "glm-5.3-flash",
    },
    # 阿里云百炼 MaaS（OpenAI 兼容，跨 6+ 独立家族）
    "alibaba": {
        "glm-4.6", "glm-4.5-air",                    # 阿里侧也提供 GLM 系（非独占）
        "deepseek-v4-pro", "deepseek-v4.1-flash",
        "qwen3.8-max", "qwen3.8-flash", "qwen3.8-27b", "qwen3.7-plus", "qwen-flash",
        "kimi-k3", "kimi-k2.5", "MiniMax-M2.5",
    },
    "openrouter": {
        "openai/gpt-4o", "openai/gpt-4o-mini",
        "anthropic/claude-sonnet-4",
        "z-ai/glm-4.6", "moonshotai/kimi-k2-thinking",
    },
}

# ================= OpenRouter 全家族映射（2026-09-24 增设）=================
# 起因：阿里百炼 MaaS **账号级封锁**（/models 200 但每个 chat 403
#   "Access to model denied ... eligible for using the model"），
#   整条阿里轨（DS / QWEN / KIMI / MiniMax + 判官 qwen-flash）停摆。
#   处置：改由 OpenRouter 承载，**pin 到 Alibaba 上游**（同厂商托管，
#   等价性最高；计费走 OpenRouter，绕开百炼账号）。
#
# 裸名 -> OpenRouter id。左列与 PAIRS 里的写法完全一致，故 PAIRS 无需改动。
# 刻意不用 '~...-latest' 与 ':batch' 变体（前者指向滚动别名、不可复现；
# 后者是异步批处理，违反在线协议）。
OPENROUTER_IDS: Dict[str, str] = {
    "deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "deepseek-v4-pro":     "deepseek/deepseek-v4-pro",
    "qwen3.8-flash":       "qwen/qwen3.8-flash",
    "qwen3.8-max":         "qwen/qwen3.8-max-0902",
    "kimi-k2.5":           "moonshotai/kimi-k2.5",
    "kimi-k3":             "moonshotai/kimi-k3",
    "glm-5.3-flash":       "z-ai/glm-5.3-flash",
    "glm-4.5-air":         "z-ai/glm-4.5-air",
    "glm-4.6":             "z-ai/glm-4.6",
    "MiniMax-M2.5":        "minimax/minimax-m2.5",
    # ⚠️ OpenRouter **没有** `qwen-flash` 这个 id（阿里侧专有）。
    #    取 3.7-flash：最廉价的 Qwen flash 档（实测 in=3e-8 out=1.3e-7），
    #    保持「廉价、无 CoT、快」的判官定位。**换判官必须全轨统一**，
    #    否则破坏跨臂协议对称（见 _supp_run.py 文件头）。
    "qwen-flash":          "qwen/qwen3.7-flash",
}

# ★ 上游钉选。OpenRouter 默认路由**不稳定**：实测同一 prompt 连发 4 次
#   落到 Novita x3 + Wafer x1；且各上游量化不一（fp4 / fp8 / unknown），
#   结果不可互换。⇒ 必须钉死，否则实验不可复现。
#   'Alibaba' 实测可 pin 且 3/3 稳定，且与阿里百炼同厂商托管。
#   ⚠️ 注意：**同厂商托管 ≠ 同权重配置**（OpenRouter 侧可能是不同量化/版本）。
#   因此换端点 = 引入一个未受控变量，整条臂必须**统一**走同一端点，
#   不可一半阿里端、一半 OpenRouter 端混用（那是端点混淆）。
#   设 OPENROUTER_PIN='' 可关闭 pin（不推荐）。
OPENROUTER_PIN = os.environ.get("OPENROUTER_PIN", "Alibaba").strip() or None


def route_provider(model: str, default: str) -> str:
    """按模型名解析实际应使用的 provider。规则见 PROVIDER_MODELS 上方注释。"""
    # ★ 显式指定 OpenRouter 轨时**优先保持**：否则 deepseek-v4.1-flash 等
    #   既在 alibaba 集合里（独占判定）就会被抢回被封锁的阿里端。
    if default == "openrouter" and model in OPENROUTER_IDS:
        return "openrouter"
    if model in PROVIDER_MODELS.get(default, ()):
        return default                        # 当前端点就有 → 保持原样
    hits = [p for p, ms in PROVIDER_MODELS.items() if model in ms]
    if len(hits) == 1:
        return hits[0]                        # 独占模型 → 必须换端点
    return default


# ================= 基础设施 =================
def load_env(path: Path) -> Dict[str, str]:
    cfg = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


class QuotaExhausted(RuntimeError):
    """智谱 5 小时滚动限额用尽（error code 1308）。重试无意义，必须等待或降级端点。"""


class RateLimited(RuntimeError):
    """真正的瞬时限流（QPS 超限）。指数退避可恢复。"""


class ModelUnavailable(RuntimeError):
    """模型不存在 / 未开通该产品。重试无意义，应立即放弃本模型（阿里云 404/400 实测）。"""


class AccessDenied(RuntimeError):
    """★ 账号级模型访问被拒（2026-09-24 阿里云百炼实测）。

    表现：HTTP 403 + {"code":"AccessDenied.Unpurchased",
                     "message":"Access to model denied. Please make sure you are
                                eligible for using the model."}
    特征：**同账号下所有模型一起被拒**（实测 15 个模型，含从未调用过的
          qwen-max / qwen-plus / qwen-turbo 全部 403），而 /models 列表接口
          仍返回 200 —— 说明 key 有效，是「未购买/未开通/免费额度用尽」的
          账号级授权问题，**换端点、重试、换模型都无效**。
    处置：立刻放弃，不要退避重试（旧行为会老老实实重试 5 次 × 80 题，
          白烧一小时才暴露问题 —— 这就是加 preflight 的原因）。
    """


class RateLimiter:
    """全局令牌桶 —— 跨线程共享的节流器。

    ★ 实测修正（2026-09-23）：早前认为「智谱限流严格、必须重间隔」是**误判**。
      真实原因是配额耗尽（code 1308）被当成限流，重试风暴自己把额度烧穿。
      实测：唯一化提示词的短调用在 workers=16、interval=0 下 32/32 全成功（16.3 req/s），
      完全不需要节流。因此默认间隔下调到 0.0（不节流），把并发交给 workers 控制；
      但仍保留该类，因为**真实 CoT 长调用**（58~420s）在途并发过高时会触发真 429，
      此时用 min_interval 做温和限速是有效的第二道闸。
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_ok - now)
            self._next_ok = max(now, self._next_ok) + self.min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


class Client:
    """支持 OpenRouter / 智谱 双 provider 的客户端，带重试 + 磁盘缓存 + 全局限流 + 端点降级。

    ★ 智谱适配的关键点（实测得出，勿改）：
      1. 端点 /api/coding/paas/v4（Coding Plan）与 /api/paas/v4（通用）**配额独立**
      2. 响应里 `content` 与 `reasoning_content` 是分开的两个字段 —— 只取 content
      3. reasoning_tokens 计入 completion_tokens → max_tokens 必须给足余量，
         否则 finish_reason='length' 且 content 为空字符串 → JSON 解析静默失败
      4. 缓存键必须含 provider 与端点，否则换端点会命中错误缓存
      5. ★ 必须区分两类 429：
         - code 1308「已达到 N 小时的使用上限」→ 配额耗尽，重试无意义（QuotaExhausted）
         - 其他 429 → 瞬时限流，指数退避可恢复（RateLimited）
         把配额耗尽误判为限流会导致：白白烧掉配额做无效重试
      6. ★ 思维链推理慢：timeout 需 240s（原 90s 会大量 ReadTimeout）
      7. 配额耗尽时可降级到 general 端点（实测 general 在 coding 耗尽时仍可用）
    """

    CODING_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
    GENERAL_URL = "https://open.bigmodel.cn/api/paas/v4"
    # ★ 阿里云百炼 MaaS（OpenAI 兼容）—— 可访问 Qwen/GLM/Kimi/DeepSeek/MiniMax 等 6+ 家族
    ALIBABA_URL = "https://llm-enerc2vosh8n2a1g.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: str, cache_dir: Path, dry_run=False, verbose=True,
                 provider="openrouter", base_url=None, min_interval=None, think="auto",
                 keys: Dict[str, str] = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.verbose = verbose
        self.provider = provider
        self.think = think          # auto / on / off
        # ★ 跨端点路由（2026-09-24）：每个 provider 各自的 key。
        #   单 provider 时 keys 只含一项，行为与改造前完全一致。
        self.keys: Dict[str, str] = {k: v for k, v in (keys or {}).items() if v}
        if api_key:
            self.keys.setdefault(provider, api_key)
        self.key = self.keys.get(provider) or api_key
        self.base_url = base_url or {
            "openrouter": "https://openrouter.ai/api/v1",
            "zhipu": self.CODING_URL,
            "alibaba": self.ALIBABA_URL,
        }.get(provider, self.CODING_URL)
        self.pbudget = PROVIDER_BUDGET.get(provider, PROVIDER_BUDGET["openrouter"])
        default_interval = 0.0          # ★ 实测无需节流（原 0.6 是误判产物）
        self.limiter = RateLimiter(min_interval if min_interval is not None else default_interval)
        self.calls = 0
        self.cache_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.reasoning_tokens = 0
        self.empty_contents = 0
        self.rate_limited = 0
        self.quota_exhausted = 0
        self.endpoint_switches = 0
        self.quota_reset_msg = ""
        # ★ (2026-09-24) 端点级「配额死亡」集合：(provider, url)。
        #   智谱 coding 的 5 小时滚动额度一旦耗尽，重试/换端点都无意义；
        #   标记后该端点的后续调用直接秒失败，避免 80 题各烧 5 次退避
        #   （实测退避风暴 1.5h 只跑完 3 题，还把额度本身烧光）。
        self._url_dead: set = set()
        self.routed: Dict[str, int] = {}      # 记录被路由到其他端点的模型调用次数
        self.end_urls: Dict[str, str] = {}    # 各 provider 实际生效的端点

    def _base_url_for(self, provider: str) -> str:
        return {
            "openrouter": "https://openrouter.ai/api/v1",
            "zhipu": self.CODING_URL,
            "alibaba": self.ALIBABA_URL,
        }.get(provider, self.CODING_URL)

    def _urls_for(self, provider: str, base_url: str):
        """返回候选端点列表。coding 优先，配额耗尽时降级到 general。"""
        if provider != "zhipu":
            return [base_url]
        urls = []
        for u in (base_url, self.GENERAL_URL, self.CODING_URL):
            if u not in urls:
                urls.append(u)
        return urls

    def _urls(self):
        """兼容旧调用：按主 provider 返回候选端点。"""
        return self._urls_for(self.provider, self.base_url)

    # ★ 经实测确认「接受 thinking:disabled 且不报错」的模型白名单（2026-09-23）
    #   不在名单内的模型一律不注入该字段 —— 因为 glm-5.3 会直接 400，
    #   宁可保留 CoT（多花 token）也不能让请求失败。
    NOTHINK_SAFE = {
        "glm-4.6", "glm-4.5-air", "glm-4-plus", "glm-4-flash",   # 智谱（实测可接受）
        "kimi-k3", "qwen3.8-max", "qwen3.8-27b", "qwen3.7-plus",
        "qwen-flash", "deepseek-v4-pro", "deepseek-v4.1-flash",   # 阿里（实测生效）
    }

    def _nothink_ok(self, model: str) -> bool:
        return model in self.NOTHINK_SAFE

    def _ck(self, model, messages, temp, max_tokens, url=None, prov=None,
            nonce: str = None) -> Path:
        # ★ 缓存键含 provider：换端点必须换缓存，否则命中错误缓存。
        #   注意：url 参数实际传的是 think 标志串，因此 base_url 不进键；
        #   这保证了「模型未路由时」的键与改造前逐字节一致（不触发重跑）。
        parts = [prov or self.provider, url or self.base_url, model, messages,
                 temp, max_tokens]
        # ★ nonce 只进缓存键、不进 payload（2026-09-24，用户 directive 1 的落地需要）：
        #   同源臂两侧是**同一个模型**，若 prompt 也相同则缓存键相同 —— 第二次调用
        #   会直接命中第一次的缓存，两侧种子逐字相同，「同源」就退化成「同一个人
        #   自言自语」，机制必然空转。传 nonce 使两次采样错开。
        #   不传 nonce 时 parts 与改造前逐字节一致 ⇒ 已跑各臂缓存继续命中，不重跑。
        if nonce:
            parts.append(f"#nonce={nonce}")
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest()[:24] + ".json")

    def _post(self, url, payload, key=None):
        """单次请求。把限流/配额/模型不存在细分为不同异常抛给调用方。

        ★ 阿里云 MaaS 的失败模式与智谱不同（实测 2026-09-23）：
          - HTTP 403 + "Free quota exhausted"  → 免费额度用尽，重试无意义（QuotaExhausted）
          - HTTP 404 + "Model not exist"       → 模型名不可用（ModelUnavailable，不可重试）
          - HTTP 400 + "product is not activated" → 未开通该产品（ModelUnavailable）
        智谱的失败模式：
          - HTTP 429 + code 1308               → 5 小时滚动配额耗尽（QuotaExhausted）
          - HTTP 429 其他                      → 瞬时限流，退避可恢复（RateLimited）
        """
        r = requests.post(
            url + "/chat/completions",
            headers={"Authorization": f"Bearer {key or self.key}",
                     "Content-Type": "application/json"},
            json=payload, timeout=BUDGET["timeout"])
        if r.status_code == 429:
            raw = r.text or ""
            # ★ 关键区分：1308 = 滚动窗口配额用尽；1113 = 余额不足/无可用资源包。
            #   两者都是**永久性**错误，退避重试纯属浪费（2026-09-24 实测教训：
            #   1113 被误判为瞬时限流 → 377 次退避风暴 → 1.5h 只跑完 3/80 题，
            #   且把 coding 端点的 5 小时额度本身也拖垮）。
            if ("1308" in raw or "1113" in raw
                    or "使用上限" in raw or "usage limit" in raw.lower()
                    or "余额不足" in raw or "请充值" in raw
                    or "insufficient" in raw.lower() or "no balance" in raw.lower()):
                msg = ""
                try:
                    msg = json.loads(raw).get("error", {}).get("message", "")
                except Exception:
                    msg = raw[:200]
                raise QuotaExhausted(msg or "quota exhausted")
            raise RateLimited(raw[:200])
        # ★ 阿里云：403 免费额度耗尽 / 404 模型不存在 / 400 未开通 —— 三者重试都无意义
        if r.status_code in (400, 403, 404):
            raw = r.text or ""
            low = raw.lower()
            try:
                emsg = json.loads(raw).get("error", {}).get("message", "") or raw[:200]
            except Exception:
                emsg = raw[:200]
            if ("quota" in low and "exhaust" in low) or "free quota" in low \
                    or "额度" in raw or "余额不足" in raw or "insufficient" in low:
                raise QuotaExhausted(emsg)
            # ★ 账号级封锁优先判定（2026-09-24）：MaaS 整站 403 AccessDenied.Unpurchased
            if "accessdenied" in low or "unpurchased" in low or "access denied" in low:
                raise AccessDenied(emsg)
            if "model not exist" in low or "not activated" in low or "not found" in low:
                raise ModelUnavailable(emsg)
        r.raise_for_status()
        return r.json()

    def chat(self, model: str, messages: List[Dict[str, str]],
             temperature: float = None, max_tokens: int = None,
             role: str = "consultant", nonce: str = None) -> str:
        """role: consultant / judge —— 决定是否关闭思维链（think=auto 时生效）
        nonce: 仅用于错开缓存键（同一模型取多个独立样本，见 _ck 注释）。"""
        # ★ 按模型解析实际端点（2026-09-24）：flash 家族横跨智谱/阿里两端，
        #   单 provider 客户端会把 glm-5.3-flash 发到阿里 → 404。
        prov = route_provider(model, self.provider)
        if prov != self.provider:
            self.routed[model] = self.routed.get(model, 0) + 1
        key = self.keys.get(prov) or self.key
        base_url = self._base_url_for(prov)
        pbudget = PROVIDER_BUDGET.get(prov, PROVIDER_BUDGET["openrouter"])
        if not key and not self.dry_run:
            raise RuntimeError(f"缺少 {prov} 的 API Key，无法调用模型 {model}"
                               f"（请在 code/.env 配置 {prov.upper()}_API_KEY）")

        temperature = BUDGET["temperature"] if temperature is None else temperature
        if max_tokens is None:
            max_tokens = pbudget["max_tokens"]
        floor = pbudget.get("min_answer_tokens")
        if floor:                       # 保证 content 不被思维链饿死
            max_tokens = max(max_tokens, floor)
        # ★ 决定本角色是否关闭思维链
        #   注意：NO_THINK({"thinking":{"type":"disabled"}}) 是**智谱专有**字段，
        #   跨 provider 行为不一致（2026-09-23 实测）：
        #     · kimi-k3 / qwen3.8-max / deepseek-v4-pro → 生效，CoT 归零
        #     · glm-4.6 / MiniMax-M2.5                  → 被忽略，CoT 仍开（无害）
        #     · glm-5.3                                 → **HTTP 400 参数不合法（有害！）**
        #   因此只在确认安全的模型上注入该字段，其余模型不注入（退化为保留 CoT）。
        disable_think = False
        if self.think == "off":
            disable_think = True
        elif self.think == "auto" and role == "judge":
            disable_think = True
        if disable_think and not self._nothink_ok(model):
            disable_think = False
        # ★ pin 进缓存键：同一 model 换上游（PIN 变了）结果不可互换，必须视作
        #   不同缓存项。**只在 OpenRouter 轨追加** —— 其它 provider 的 think 标签
        #   必须逐字节不变，否则已跑各臂的缓存会全部失效、整批重跑烧钱
        #   （见 _ck 的 url 参数注释）。注意 _ck 位置参数上限为 7，不可再加参数。
        _tag = f"think={disable_think}"
        if prov == "openrouter":
            _tag = f"think={disable_think}|pin={OPENROUTER_PIN or '-'}"
        ck = self._ck(model, messages, temperature, max_tokens, _tag, prov, nonce)
        if ck.exists():
            self.cache_hits += 1
            return json.loads(ck.read_text(encoding="utf-8"))["text"]
        if self.dry_run:
            return '{"answer":"A","reasoning":["dry-run step 1","step 2","step 3","step 4"],' \
                   '"conflicts":[],"confidence":"medium","selected_answer":"A","winner":"both",' \
                   '"exit_mode":"agree","reason":"dry-run"}'

        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if prov == "openrouter":
            # 裸名 -> OpenRouter id（PAIRS 里写的是裸名，此处翻译）
            payload["model"] = OPENROUTER_IDS.get(model, model)
            payload["top_p"] = BUDGET["top_p"]
            if OPENROUTER_PIN:
                # 钉死上游：默认路由不稳定 + 各上游量化不一致
                payload["provider"] = {"order": [OPENROUTER_PIN],
                                       "allow_fallbacks": False}
        elif disable_think:
            payload.update(NO_THINK)    # 智谱专用：关闭思维链

        last = None
        # ★ 外层遍历候选端点：coding 配额耗尽时自动降级到 general
        _cand = [u for u in self._urls_for(prov, base_url) if (prov, u) not in self._url_dead]
        if not _cand:
            # 该 provider 全部端点都已确认配额耗尽 → 秒失败，不再逐题重试
            raise QuotaExhausted(
                f"[{prov}] all endpoints exhausted/unavailable: "
                f"{self.quota_reset_msg or last}")
        for url in _cand:
            for attempt in range(1, BUDGET["retries"] + 1):
                try:
                    self.limiter.wait()          # 全局节流，防 QPS 限流
                    body = self._post(url, payload, key)
                    if "error" in body:
                        raise RuntimeError(str(body["error"])[:200])
                    msg = body["choices"][0]["message"]
                    txt = msg.get("content") or ""
                    # ★ CoT 字段名跨 provider 不一致：智谱/阿里用 reasoning_content，
                    #   OpenRouter 用 reasoning。只读前者会把「有 CoT」误报成
                    #   reasoning_len=0，把失败诊断带偏（2026-09-24 冒烟实测）。
                    _rc = (msg.get("reasoning_content") or msg.get("reasoning") or "")
                    # ★ 空 content 是智谱最常见的静默失败，必须重试而不是返回空串
                    if not txt.strip():
                        self.empty_contents += 1
                        raise RuntimeError(
                            f"empty content (finish={body['choices'][0].get('finish_reason')}, "
                            f"reasoning_len={len(_rc)})")
                    u = body.get("usage") or {}
                    self.tokens_in += u.get("prompt_tokens", 0)
                    self.tokens_out += u.get("completion_tokens", 0)
                    det = u.get("completion_tokens_details") or {}
                    self.reasoning_tokens += det.get("reasoning_tokens", 0)
                    self.calls += 1
                    self.end_urls[prov] = url    # 记住各 provider 的可用端点
                    if prov == self.provider:
                        self.base_url = url      # 主 provider 端点（用于报告）
                    ck.write_text(json.dumps({"text": txt, "model": model,
                                              "provider": prov, "url": url},
                                             ensure_ascii=False), encoding="utf-8")
                    return txt
                except QuotaExhausted as ex:
                    # ★ 配额耗尽：重试无意义，立刻换端点（不 sleep，不消耗重试次数）
                    last = ex
                    if prov == self.provider:
                        self.quota_exhausted += 1
                        self.quota_reset_msg = str(ex)
                    self._url_dead.add((prov, url))      # ★ 标记该端点已死
                    if self.verbose:
                        print(f"    [配额耗尽:{prov}] {str(ex)[:110]}")
                        print(f"    -> 端点 {url.rsplit('/', 3)[-2:]} 标记为不可用，切换下一个")
                    break                        # 跳出重试循环，进入下一个 url
                except ModelUnavailable as ex:
                    # ★ 模型不存在/未开通：换端点也没用，直接放弃本模型（避免 5 次无用重试）
                    last = ex
                    if self.verbose:
                        print(f"    [模型不可用:{prov}] {str(ex)[:110]} -> 放弃")
                    raise RuntimeError(f"model unavailable: {ex}")
                except AccessDenied as ex:
                    # ★ 账号级拒绝：整个账号下所有模型都不可用，重试/换端点纯属浪费
                    last = ex
                    if self.verbose:
                        print(f"    [账号级拒绝:{prov}] {str(ex)[:110]} -> 立即中止该端点")
                    raise RuntimeError(f"access denied (account-level): {ex}")
                except RateLimited as ex:
                    last = ex
                    self.rate_limited += 1
                    backoff = pbudget.get("backoff", 5)
                    wait = backoff * (2 ** attempt) + random.uniform(0, backoff)
                    if self.verbose:
                        print(f"    retry {attempt}/{BUDGET['retries']} [QPS限流:{prov}] "
                              f"等待{wait:.1f}s")
                    if attempt < BUDGET["retries"]:
                        time.sleep(wait)
                except Exception as ex:
                    last = ex
                    msg = str(ex)
                    # ★ 2026-09-25 修复：finish=length + 空 content = 思维链吃满
                    #   max_tokens 上限（G1 智谱 11/80 种子失败、OR OH1 停组皆此因），
                    #   原样重试必然再次截断。处置：检测到该失败签名后，把本线程
                    #   payload 的 max_tokens 升级一次（×3，≤65536）再重试。
                    #   缓存键仍按调用时的原始 max_tokens 计算（_ck 在升级前已算好），
                    #   完整回答（finish=stop）与上限取值无关 ⇒ 写回原缓存键无损可比性，
                    #   已缓存调用不受影响、零重跑。
                    if "empty content (finish=length" in msg and payload["max_tokens"] < 65536:
                        _old = payload["max_tokens"]
                        payload["max_tokens"] = min(_old * 3, 65536)
                        if self.verbose:
                            print(f"    [max_tokens升级:{prov}] {_old} -> {payload['max_tokens']}"
                                  f"（finish=length：思维链超上限，升级后重试）")
                    is_timeout = "timeout" in msg.lower() or "timed out" in msg.lower()
                    backoff = pbudget.get("backoff", 5)
                    wait = backoff * attempt
                    if self.verbose:
                        tag = "超时" if is_timeout else type(ex).__name__
                        print(f"    retry {attempt}/{BUDGET['retries']} [{tag}:{prov}] "
                              f"等待{wait:.1f}s: {msg[:90]}")
                    if attempt < BUDGET["retries"]:
                        time.sleep(wait)
        raise RuntimeError(f"chat failed after all attempts: {last}")


def jparse(txt: str) -> Dict[str, Any]:
    try:
        return json.loads(txt)
    except Exception:
        pass
    a, b = txt.find("{"), txt.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(txt[a:b + 1])
        except Exception:
            pass
    return {}


# ================= 数据 =================
def fmt_choices(raw: Any) -> str:
    if isinstance(raw, dict):
        return "\n".join(f"{k}. {v}" for k, v in raw.items())
    return str(raw or "")


def load_cases(dataset: str, pair: str, limit: Optional[int], seed: int = 42,
               offset: int = 0) -> List[Dict]:
    (m1, f1), (m2, f2) = PAIRS[pair]
    d1 = {e["index"]: e for e in json.loads((DATA / dataset / f1).read_text(encoding="utf-8"))}
    d2 = {e["index"]: e for e in json.loads((DATA / dataset / f2).read_text(encoding="utf-8"))}
    idx = sorted(set(d1) & set(d2))
    # ★ held-out 划分：cases 直接按 index 升序取前 limit 个，seed 只用于 rng、
    #   不参与切片，所以 [offset : offset+limit] 是确定性的、可复现的互斥划分。
    #   offset=0  -> 已见集（此前所有臂用的那批题）
    #   offset=80 -> 留出集（问题集与已见集零重叠）
    rng = random.Random(seed)
    if offset:
        idx = idx[offset:]
    cases = []
    for i in idx:
        a, b = d1[i], d2[i]
        cases.append(dict(
            index=i,
            question=(a.get("Question") or a.get("question") or "").strip(),
            choices=fmt_choices(a.get("choices")),
            correct=str(a.get("correct_choice") or a.get("Correct Answer") or "").strip().upper(),
            # 位置随机化（固定种子），控制位置偏差
            swap=rng.random() < 0.5,
            c1_model=m1, c1_answer=str(a.get("model_answer", "")).strip().upper(),
            c1_reasoning=a.get("model_reasoning") or [], c1_correct=bool(a.get("model_correct")),
            c2_model=m2, c2_answer=str(b.get("model_answer", "")).strip().upper(),
            c2_reasoning=b.get("model_reasoning") or [], c2_correct=bool(b.get("model_correct")),
        ))
    if limit:
        cases = cases[:limit]
    return cases


# ================= 运行时种子（directive ① 的落地） =================
# 用户裁定：「同源不靠 data file，靠同 model id」。因此同家族臂两侧的
# 初始答案/推理不再从 glm.json / kimi.json 读取，而是**用顾问自己的模型
# 现采样两次**（同模型 + 同一问题 + 两个 nonce ⇒ 两次独立生成）。
# 这样「同源」成立与否完全由 model id 决定，与任何数据文件无关。
ITEM_LABEL_RE = re.compile(r"^\s*[\(\[]?\s*([A-Za-z])\s*[\.\)\]:,、]")


def _labels_of(choices_str: str) -> List[str]:
    """从格式化后的选项文本里抽出标签集合，如 'A. foo\\nB. bar' -> ['A','B']。"""
    labs = []
    for line in (choices_str or "").splitlines():
        m = ITEM_LABEL_RE.match(line)
        if m:
            labs.append(m.group(1).upper())
    return labs


def norm_label(raw: Any, choices_str: str) -> str:
    """把模型输出规整成选项标签（模型可能回 'A'、'A.'、'A) 4.9 m/s^2'、'(A)'）。"""
    labs = _labels_of(choices_str)
    s = str(raw or "").strip()
    if not s:
        return ""
    up = s.upper()
    if up in labs:
        return up
    m = ITEM_LABEL_RE.match(s)
    if m and m.group(1).upper() in labs:
        return m.group(1).upper()
    for tok in re.findall(r"\b([A-Za-z])\b", s):
        if tok.upper() in labs:
            return tok.upper()
    return up[:1]


def gen_seed(cl: "Client", model: str, case: Dict, tpl: str,
             nonce: str) -> Tuple[str, List[str]]:
    """用 `model` 现生成一个顾问的「初始答案 + 推理」。"""
    prompt = (tpl.replace("{question_text}", "{question}")
                 .replace("{choices_text}", "{choices}"))
    prompt = prompt.format(question=case["question"], choices=case["choices"])
    txt = cl.chat(model, [{"role": "user", "content": prompt}],
                  role="consultant", nonce=nonce)
    j = jparse(txt)
    ans = norm_label(j.get("answer", ""), case["choices"])
    rs = j.get("reasoning") or []
    if isinstance(rs, str):
        rs = [rs]
    rs = [str(x) for x in rs if str(x).strip()]
    if not rs:                       # JSON 解析失败：退化为把原文当推理，不让 episode 崩
        rs = [txt.strip()[:400]]
    return ans, rs


def seed_runtime(cl: "Client", cases: List[Dict], tpl: str,
                 workers: int = 4) -> Tuple[int, List[int]]:
    """把 cases 的种子整体换成「同一模型运行时双采样」。返回 (失败题数, 失败 index 列表)。

    nonce 形如 seedA-<index> / seedB-<index>：既错开同模型两侧的缓存键，
    又保持确定性（重跑同一条命令会命中缓存，不重复烧钱）。

    ★ 关键不变量（2026-09-24 修复）：失败题的种子字段必须被**清空**，
      绝不能让它沿用数据文件里的旧种子 —— 否则产物会变成
      「15 题 runtime 种子 + 65 题 data 种子」的静默混合体，而 JSON 里毫无标记。
    """
    by_idx = {c["index"]: c for c in cases}

    def one(c: Dict):
        i = c["index"]
        a1, r1 = gen_seed(cl, c["c1_model"], c, tpl, f"seedA-{i}")
        a2, r2 = gen_seed(cl, c["c2_model"], c, tpl, f"seedB-{i}")
        return i, a1, r1, a2, r2

    n_fail = 0
    failed: List[int] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(one, c): c["index"] for c in cases}
        for f in as_completed(futs):
            idx = futs[f]
            try:
                i, a1, r1, a2, r2 = f.result()
            except Exception as e:
                n_fail += 1
                failed.append(idx)
                c = by_idx[idx]
                # ★ 清空而非保留：让「种子缺失」在下游显式暴露，而不是伪装成有效数据
                c["c1_answer"] = c["c2_answer"] = ""
                c["c1_reasoning"] = c["c2_reasoning"] = []
                c["c1_correct"] = c["c2_correct"] = False
                c["seed_failed"] = True
                print(f"  [seed] [{idx}] ERROR {type(e).__name__}: {str(e)[:110]}")
                continue
            c = by_idx[i]
            c["c1_answer"], c["c1_reasoning"] = a1, r1
            c["c2_answer"], c["c2_reasoning"] = a2, r2
            c["c1_correct"] = (a1 == c["correct"])
            c["c2_correct"] = (a2 == c["correct"])
            c.pop("seed_failed", None)
    return n_fail, sorted(failed)


# ================= 各协议的 prompt（对称设计） =================
_WORD = f"Keep each argument under {BUDGET['words_limit']} words."

CONSULTANT_HEAD = (
    "You are a PhD-level domain expert. A question and its options are below.\n\n"
    f"Question: {{question}}\n\nOptions:\n{{choices}}\n\n"
    "Your previously committed answer and reasoning are earlier in this conversation; "
    "another consultant has provided a competing position.\n\n"
    "Other consultant's latest position:\nAnswer: {other_answer}\nReasoning:\n{other_reasoning}\n\n"
)

# 各协议的任务描述（这是唯一被允许的差异 —— 交互机制本身）
TASK_SPEC = {
    # 论文原版 DR：识别冲突 + 判定 winner + 可改答案
    "dr": (
        "Your task:\n"
        "1. Identify every concrete conflict (numerical discrepancy, conceptual disagreement, "
        "unstated assumption). For each, state the issue, compare both sides, and conclude who "
        "currently has the stronger argument (\"self\", \"other\", or \"tie\").\n"
        "2. Decide whether to adjust your answer. If you keep it, restate a clean reasoning trace.\n"
        "3. Be explicit about any evidence you adopt from the other consultant.\n"
        f"{_WORD}\n"
    ),
    # ★ 安慰剂：等长、等结构、同样的 JSON 要求，但机制内容被替换为无信息的格式化任务
    "placebo": (
        "Your task:\n"
        "1. Restate both reasoning traces in parallel, aligned step by step.\n"
        "2. Normalize the terminology, units and notation used by each side.\n"
        "3. Rephrase both positions in a common format.\n"
        f"{_WORD}\n"
    ),
    # ☆ 消融：允许多轮改答案，但不识别 crux、不判定 winner
    "dr_nocrux": (
        "Your task:\n"
        "1. Read the other consultant's reasoning.\n"
        "2. Decide whether to adjust your answer. If you keep it, restate a clean reasoning trace.\n"
        "3. Be explicit about anything you adopt from the other consultant.\n"
        f"{_WORD}\n"
    ),
    # ☆ 消融：识别冲突但不输出 winner 标签
    "dr_noconf": (
        "Your task:\n"
        "1. Identify every concrete conflict and state the issue with both sides' evidence. "
        "Do NOT label a winner.\n"
        "2. Decide whether to adjust your answer. If you keep it, restate a clean reasoning trace.\n"
        f"{_WORD}\n"
    ),
    # 对抗式：立场固定，不可改
    "debate": (
        "Your task:\n"
        "1. You MUST continue to defend your original answer. You may not switch answers.\n"
        "2. Rebut the other consultant's reasoning point by point.\n"
        "3. Supply your strongest supporting evidence.\n"
        f"{_WORD}\n"
    ),
}

JSON_TAIL_DR = (
    '\nRespond in strict JSON with these keys only:\n'
    '- "answer": exactly one label from the provided choices.\n'
    '- "reasoning": an ordered list of 4-6 concise steps.\n'
    '- "conflicts": a list of objects each with "issue","winner"("self"/"other"/"tie"),"explanation".\n'
    '- "confidence": "low","medium",or "high".\n'
    'Do not include any text outside the JSON object.'
)
JSON_TAIL_FLAT = (
    '\nRespond in strict JSON with these keys only:\n'
    '- "answer": exactly one label from the provided choices.\n'
    '- "reasoning": an ordered list of 4-6 concise steps.\n'
    '- "confidence": "low","medium",or "high".\n'
    'Do not include any text outside the JSON object.'
)

JUDGE_BODY = (
    "**Consultant 1 Latest Position**\nAnswer: {a1}\nReasoning Trace:\n{r1}\n\n"
    "**Consultant 2 Latest Position**\nAnswer: {a2}\nReasoning Trace:\n{r2}\n\n"
    "**Case Status**\n- Consultant turns used: {turns}/{maxturns}\n- Agreement status: {agree}\n\n"
    "**Question**\n{question}\n\n**Answer Options**\n{choices}\n\n"
    "{conflict_block}"
    "**Guidance for Evaluation**\n"
    "Rely on logical coherence and self-consistency rather than subject-matter recall:\n"
    "1. Does each reasoning trace progress logically without leaps?\n"
    "2. Do definitions, units and intermediate conclusions align with the final answer?\n"
    "3. Prefer derivations grounded in first principles over unsupported assertions.\n"
    "4. If the consultants disagree, determine which line of reasoning is stronger.\n\n"
    'When Agreement status == "agree", confirm the shared answer is sound. If you trust it, set '
    '"exit_mode":"agree","winner":"both". If you distrust it, override with the label you believe '
    'correct and set "winner":"neither".\n'
    'When Agreement status == "disagree", pick the stronger reasoning.\n\n'
    "Always respond with valid JSON:\n"
    "{{\n    \"reason\": \"<short justification>\",\n    \"exit_mode\": \"<agree | disagree>\",\n"
    "    \"winner\": \"<both | consultant_1 | consultant_2 | neither>\",\n"
    "    \"selected_answer\": \"<exactly one of the provided choice labels>\"\n}}\n"
)


@dataclass
class State:
    label: str
    model: str
    answer: str
    reasoning: List[str]
    history: List[Dict[str, str]] = field(default_factory=list)
    conflicts: List[Dict] = field(default_factory=list)
    confidences: List[str] = field(default_factory=list)
    turns: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)   # ★ 逐轮留痕，供审稿人核对 crux 机制


def init_messages(case: Dict, prompt_tpl: str, state: State, side: str):
    """把预存答案作为 assistant 消息注入 —— 与论文 run.py:283 一致的接口"""
    # 论文模板用的是 {question_text}/{choices_text}，这里做占位符统一
    q = (prompt_tpl.replace("{question_text}", "{question}")
                  .replace("{choices_text}", "{choices}"))
    q = q.format(question=case["question"], choices=case["choices"])
    payload = json.dumps({"answer": state.answer, "reasoning": [str(s) for s in state.reasoning]},
                         ensure_ascii=False)
    state.history = [{"role": "user", "content": q}, {"role": "assistant", "content": payload}]


def consultant_turn(cl: Client, proto: str, case: Dict, me: State, other: State) -> Dict:
    spec = TASK_SPEC[proto]
    tail = JSON_TAIL_FLAT if proto in ("placebo", "dr_nocrux") else JSON_TAIL_DR
    prompt = CONSULTANT_HEAD.format(
        question=case["question"], choices=case["choices"],
        other_answer=other.answer, other_reasoning="\n".join(f"{i+1}. {s}" for i, s in enumerate(other.reasoning)),
    ) + spec + tail
    msgs = list(me.history) + [{"role": "user", "content": prompt}]
    raw = cl.chat(me.model, msgs, role="consultant")
    p = jparse(raw)
    me.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": raw}]
    me.turns += 1
    new_ans = str(p.get("answer") or "").strip().upper() or me.answer
    if proto == "debate":           # 对抗式：立场锁定
        new_ans = me.answer
    me.answer = new_ans
    r = p.get("reasoning")
    if isinstance(r, list) and r:
        me.reasoning = [str(x) for x in r]
    c = p.get("conflicts")
    if isinstance(c, list):
        me.conflicts = c
    if p.get("confidence"):
        me.confidences.append(str(p["confidence"]))
    # ★ 逐轮留痕：记录本轮输入/输出，使 crux 识别过程可被独立审计
    me.trace.append(dict(
        turn=me.turns,
        in_answer=other.answer,
        out_answer=new_ans,
        changed=(str(p.get("answer") or "").strip().upper() != me.answer) if proto != "debate" else False,
        n_conflicts=len(me.conflicts),
        conflicts=me.conflicts[:3],
        confidence=p.get("confidence"),
        raw_len=len(raw),
    ))
    return p


def run_protocol(cl: Client, proto: str, case: Dict, prompt_tpl: str,
                 early_stop: bool = True) -> Dict:
    # 位置随机化
    if case["swap"]:
        L = [("consultant_1", case["c2_model"], case["c2_answer"], case["c2_reasoning"]),
             ("consultant_2", case["c1_model"], case["c1_answer"], case["c1_reasoning"])]
    else:
        L = [("consultant_1", case["c1_model"], case["c1_answer"], case["c1_reasoning"]),
             ("consultant_2", case["c2_model"], case["c2_answer"], case["c2_reasoning"])]
    s1 = State(L[0][0], L[0][1], L[0][2], list(L[0][3]))
    s2 = State(L[1][0], L[1][1], L[1][2], list(L[1][3]))
    init_messages(case, prompt_tpl, s1, "1"); init_messages(case, prompt_tpl, s2, "2")

    turns = 0
    stop_reason = "not_started"
    if proto == "naive":
        # 无顾问，判官直答
        stop_reason = "no_consultants"
    elif proto == "double":
        # 无交互：两顾问各自独立生成 5 轮论证（与 DR 的 token 预算对齐），轮换展示顺序
        for _ in range(BUDGET["max_turns"]):
            consultant_turn(cl, "dr_nocrux", case, s1, State("t", s1.model, "", []))
            consultant_turn(cl, "dr_nocrux", case, s2, State("t", s2.model, "", []))
            turns += 1
        stop_reason = "budget_exhausted"
    elif proto == "debate":
        # 对抗式：立场锁定，固定跑满 max_turns（必然 DC 收场）
        for _ in range(BUDGET["max_turns"]):
            snap = (s1.answer, list(s1.reasoning)), (s2.answer, list(s2.reasoning))
            consultant_turn(cl, proto, case, s1, State("t", s2.model, snap[1][0], snap[1][1]))
            consultant_turn(cl, proto, case, s2, State("t", s1.model, snap[0][0], snap[0][1]))
            turns += 1
        stop_reason = "budget_exhausted"
    else:
        # dr / placebo / dr_nocrux / dr_noconf：协商式，达成共识即提前终止
        while turns < BUDGET["max_turns"]:
            snap = (s1.answer, list(s1.reasoning)), (s2.answer, list(s2.reasoning))
            consultant_turn(cl, proto, case, s1, State("t", s2.model, snap[1][0], snap[1][1]))
            consultant_turn(cl, proto, case, s2, State("t", s1.model, snap[0][0], snap[0][1]))
            turns += 1
            if s1.answer == s2.answer:
                stop_reason = "consensus_reached"
                if early_stop:
                    break
        else:
            stop_reason = "budget_exhausted"
        if turns >= BUDGET["max_turns"] and stop_reason == "not_started":
            stop_reason = "budget_exhausted"

    agree = s1.answer == s2.answer
    # 判官
    if proto == "naive":
        jp = (f"**Question**\n{case['question']}\n\n**Answer Options**\n{case['choices']}\n\n"
              "Answer the question directly. Respond only with JSON:\n"
              '{"reason":"<short>","selected_answer":"<one label>"}')
    else:
        cblock = ""
        if proto not in ("placebo", "dr_noconf") and proto != "double":
            def cf(s):
                if not s.conflicts: return "No explicit conflicts recorded."
                return "\n".join(f"- {c.get('issue','?')} -> {c.get('winner','tie')}" for c in s.conflicts[:3])
            cblock = (f"Recent conflict summaries:\nConsultant 1:\n{cf(s1)}\n\nConsultant 2:\n{cf(s2)}\n\n")
        jp = JUDGE_BODY.format(
            a1=s1.answer, r1="\n".join(f"{i+1}. {x}" for i, x in enumerate(s1.reasoning)),
            a2=s2.answer, r2="\n".join(f"{i+1}. {x}" for i, x in enumerate(s2.reasoning)),
            turns=turns, maxturns=BUDGET["max_turns"], agree="agree" if agree else "disagree",
            question=case["question"], choices=case["choices"], conflict_block=cblock)
    jraw = cl.chat(case["judge_model"], [{"role": "user", "content": jp}], role="judge")
    jp_ = jparse(jraw)
    selected = str(jp_.get("selected_answer") or "").strip().upper()
    # ★ crux 是否被真正识别（论文核心机制的可审计量）
    crux_1 = [c for c in (s1.conflicts or []) if str(c.get("winner", "tie")).lower() in ("self", "other")]
    crux_2 = [c for c in (s2.conflicts or []) if str(c.get("winner", "tie")).lower() in ("self", "other")]
    return dict(
        index=case["index"], protocol=proto, dataset=case["dataset"], pair=case["pair"],
        judge=case["judge_model"], correct=case["correct"],
        c1_answer=s1.answer, c2_answer=s2.answer, consensus=agree,
        judge_answer=selected, judge_correct=(selected == case["correct"]),
        judge_winner=jp_.get("winner"), judge_exit=jp_.get("exit_mode"),
        judge_reason=jp_.get("reason"),
        # 行为标注（对齐论文 Def.3/4 的分类学）
        action_1=classify(case, s1), action_2=classify(case, s2),
        exit_mode=exit_mode(case, s1, s2),
        turns=turns, swap=case["swap"], stop_reason=stop_reason,
        # ★ 机制可审计性
        n_conflicts_1=len(s1.conflicts or []), n_conflicts_2=len(s2.conflicts or []),
        n_crux_1=len(crux_1), n_crux_2=len(crux_2),
        crux_1=crux_1[:3], crux_2=crux_2[:3],
        n_ans_changes_1=sum(1 for t in s1.trace if t["changed"]),
        n_ans_changes_2=sum(1 for t in s2.trace if t["changed"]),
        trace_1=s1.trace, trace_2=s2.trace,
        conf_1=s1.confidences, conf_2=s2.confidences,
        c1_correct=case["c1_correct"], c2_correct=case["c2_correct"],
    )


def classify(case: Dict, s: State) -> str:
    """论文 Def.3 四类动作。用该顾问自己的初始正误作 C/W 基准。"""
    was_correct = case["c1_correct"] if s.label == "consultant_1" else case["c2_correct"]
    if case["swap"]:
        was_correct = case["c2_correct"] if s.label == "consultant_1" else case["c1_correct"]
    now_correct = (s.answer == case["correct"])
    if was_correct and now_correct: return "Persistence"
    if (not was_correct) and now_correct: return "Recovery"
    if was_correct and not now_correct: return "Overthinking"
    return "Stubbornness"


def exit_mode(case: Dict, s1: State, s2: State) -> str:
    a = s1.answer == case["correct"]; b = s2.answer == case["correct"]
    if a and b: return "TC"
    if (not a) and (not b): return "AT"
    return "DC"


def preflight(cl: "Client", models: List[str], dry_run: bool = False) -> bool:
    """开跑前给每个模型打一次极短探针，任一不可达就中止。

    动机（2026-09-24 的实惨教训）：阿里云百炼账号被整站拒绝时，
    harness 会照常跑满 80 题、每题重试 5 次、指数退避到 25s，
    白烧一小时才在日志里看到「全灭」。这个探针把发现时间压到 ~30 秒。
    """
    if dry_run:
        return True
    bad = []
    for m in models:
        try:
            cl.chat(m, [{"role": "user", "content": "Reply with the single word: ok"}],
                    role="consultant", nonce="preflight")
        except Exception as e:
            bad.append((m, f"{type(e).__name__}: {str(e)[:120]}"))
    if bad:
        print("  [preflight] ★★ 以下模型不可达，已中止（避免白跑 80 题）：")
        for m, msg in bad:
            print(f"      {m}  ->  {msg}")
        print("  [preflight] 排查提示：阿里云百炼若返回 403 AccessDenied.Unpurchased，"
              "是账号级「未购买/未开通/免费额度用尽」，需到百炼控制台确认并开通或充值。")
        return False
    print(f"  [preflight] OK，{len(models)} 个模型全部可达: {models}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True,
                    choices=["naive", "double", "debate", "dr", "placebo", "dr_nocrux", "dr_noconf"])
    ap.add_argument("--dataset", default="GPQA", choices=["GPQA", "SuperGPQA", "HLE"])
    ap.add_argument("--pair", default="glm46_glm45", choices=list(PAIRS))
    ap.add_argument("--judge", default=None,
                    help="判官模型；默认按 provider 自动选择（智谱=glm-4.5-air，阿里=qwen-flash）")
    ap.add_argument("--provider", default="zhipu", choices=["zhipu", "openrouter", "alibaba"])
    ap.add_argument("--key", default=None,
                    help="API Key；默认读 $ZHIPU_API_KEY / $ALIBABA_API_KEY / $OPENROUTER_API_KEY 或 code/.env")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0,
                    help="按 index 升序跳过前 N 个 case 以构造 held-out 划分；"
                         "0=已见集（默认），80=留出集（与已见集零重叠）")
    ap.add_argument("--seed-source", default="data", choices=["data", "runtime"],
                    help="顾问初始答案/推理的来源：data=读数据文件（复现论文原始数据，默认）；"
                         "runtime=用顾问自己的模型现采样两次（directive ① 的严格同源口径："
                         "同源只由 model id 决定，不依赖任何数据文件）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=None,
                    help="并发数；默认按 provider（智谱/阿里=4，OpenRouter=4）")
    ap.add_argument("--min-interval", type=float, default=None,
                    help="全局请求最小间隔秒数；默认 0（实测无需节流）")
    ap.add_argument("--think", default="auto", choices=list(THINK_MODES),
                    help="思维链控制：auto=顾问保留/判官关闭（默认）｜on=全保留｜off=全关闭")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="跳过开跑前的模型可达性探针（默认会探，任一模型不可达即中止）")
    ap.add_argument("--allow-partial-seeds", action="store_true",
                    help="runtime 种子部分失败时仍继续（产物标记 seed_complete=false 供审计）")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="禁用共识提前终止，强制跑满 max_turns（用于检验多轮机制是否真的有效）")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_env(CODE / ".env")

    def _env_key(name: str) -> str:
        return os.environ.get(name) or cfg.get(name) or ""

    # ★ 三个 provider 的 key 全部读进来：跨端点配对（如 ds_glm = 阿里 DS × 智谱 GLM）
    #   需要同时持有两端凭据，才能按模型路由。单端点配对行为不变。
    KEYS = {"zhipu": _env_key("ZHIPU_API_KEY"),
            "alibaba": _env_key("ALIBABA_API_KEY"),
            "openrouter": _env_key("OPENROUTER_API_KEY")}
    if args.key:
        KEYS[args.provider] = args.key
    key = KEYS.get(args.provider, "")

    if args.provider == "zhipu":
        if args.judge is None:
            args.judge = "glm-4.5-air"       # 廉价判官，复现论文「弱判官」设定
    elif args.provider == "alibaba":
        if args.judge is None:
            args.judge = "qwen-flash"        # 廉价判官（实测可用、无 CoT、0.3s）
    else:
        if args.judge is None:
            args.judge = "openai/gpt-4o-mini"

    # ★ 判官也要能跨端点（若判官是另一端点独占的模型）
    judge_prov = route_provider(args.judge, args.provider)
    if not KEYS.get(judge_prov) and not args.dry_run:
        print(f"[FATAL] 判官 {args.judge} 需要 {judge_prov} 的 key"
              f"（{judge_prov.upper()}_API_KEY），但 code/.env 里没有。")
        return 1
    if not key and not args.dry_run:
        print("[FATAL] 缺少 API Key。用 --key 或设 ZHIPU_API_KEY / ALIBABA_API_KEY / OPENROUTER_API_KEY 环境变量。")
        return 1
    cl = Client(key, CACHE, dry_run=args.dry_run, provider=args.provider,
                min_interval=args.min_interval, think=args.think, keys=KEYS)
    if args.workers is None:
        args.workers = cl.pbudget.get("workers", 4)

    cases = load_cases(args.dataset, args.pair, args.limit, args.seed, args.offset)
    for c in cases:
        c.update(dataset=args.dataset, pair=args.pair, judge_model=args.judge)
    tpl = (PROMPTS / "reasoning_generation.txt").read_text(encoding="utf-8")

    # ★ directive ①：种子来源可切换（data=论文原始数据 / runtime=同模型双采样）
    _n_seed_fail, _seed_failed = 0, []
    if args.seed_source == "runtime":
        print(f"  [seed] 来源=runtime：用各顾问自己的模型现采样两次"
              f"（N={len(cases)} 题 ⇒ {2 * len(cases)} 次调用；重跑命中缓存）")
        _n_seed_fail, _seed_failed = seed_runtime(cl, cases, tpl, args.workers)
        if _n_seed_fail:
            print(f"  [seed] ★ {_n_seed_fail}/{len(cases)} 题种子生成失败 "
                  f"（index 前 12 个：{_seed_failed[:12]}）")
            if not args.allow_partial_seeds:
                print("  [seed] ★★ 已中止：runtime 种子不完整 ⇒ 同源臂两侧立场缺失，"
                      "结论不可用。请修复配额/网络后重跑 —— "
                      "已成功的种子命中缓存，不会重复计费。")
                print("  [seed]    （确要带残缺种子继续，加 --allow-partial-seeds，"
                      "但产物必须标注 seed_complete=false）")
                return 3
            print("  [seed] ⚠ --allow-partial-seeds 已开启，继续运行，"
                  "但 seed_complete=false 会写入产物供审计")
    else:
        print(f"  [seed] 来源=data（glm.json / kimi.json）")

    print(f"[{args.protocol}] {args.dataset}/{args.pair} judge={args.judge} N={len(cases)} "
          f"offset={args.offset} split={'held-out' if args.offset else 'seen'} "
          f"provider={args.provider} workers={args.workers} think={args.think} dry_run={args.dry_run}")
    print(f"  预算: {BUDGET}  provider补充: {cl.pbudget}  限流间隔={cl.limiter.min_interval}s")
    # ★ 端点路由预览：一眼看出哪个模型走哪个端点（跨端点配对的主要排错依据）
    _route_preview = {m: route_provider(m, args.provider) for m, _ in PAIRS[args.pair]}
    _route_preview[args.judge] = judge_prov
    print("  端点路由: " + ", ".join(f"{m}→{p}" for m, p in _route_preview.items()))
    _mismatch = sorted({m for m, p in _route_preview.items() if p != args.provider})
    if _mismatch:
        print(f"  ★ 跨端点模型 {_mismatch}（改用 "
              f"{sorted({_route_preview[m] for m in _mismatch})} 的 key/端点）")
    # ★ 开跑前探针：模型不可达就立刻中止，别跑满 80 题才发现
    if not args.skip_preflight:
        if not preflight(cl, sorted(set(list(_route_preview))), args.dry_run):
            return 2
    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_protocol, cl, args.protocol, c, tpl, not args.no_early_stop): c["index"]
                for c in cases}
        for f in as_completed(futs):
            idx = futs[f]
            try:
                r = f.result(); results.append(r)
                print(f"  [{idx}] judge={r['judge_answer']} correct={r['judge_correct']} "
                      f"exit={r['exit_mode']} turns={r['turns']}")
            except Exception as ex2:
                print(f"  [{idx}] ERROR {type(ex2).__name__}: {str(ex2)[:140]}")
                results.append(dict(index=idx, protocol=args.protocol, status="error", error=str(ex2)))

    ok = [r for r in results if "judge_correct" in r]
    acc = sum(1 for r in ok if r["judge_correct"]) / len(ok) * 100 if ok else 0.0
    # ★ 留出集必须落进文件名，否则 offset=80 会覆盖 offset=0 的结果
    _ho = f"_ho{args.offset}" if getattr(args, "offset", 0) else ""
    # ★ 种子来源也必须落进文件名，否则 runtime 双采样会覆盖 data 版结果
    _sd = "" if args.seed_source == "data" else f"_{args.seed_source}seed"
    out = args.out or (OUT / f"{args.protocol}_{args.dataset}_{args.pair}_"
                              f"{args.judge.replace('/','_')}{_ho}{_sd}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(protocol=args.protocol, dataset=args.dataset, pair=args.pair, judge=args.judge,
                   provider=args.provider, budget=BUDGET, provider_budget=cl.pbudget,
                   routes=_route_preview, routed_calls=cl.routed, end_urls=cl.end_urls,
                   seed_source=args.seed_source, offset=args.offset,
                   seed_complete=(_n_seed_fail == 0),
                   n_seed_fail=_n_seed_fail, seed_failed_indices=_seed_failed,
                   n_ok=len(ok), n_err=len(results) - len(ok), accuracy=acc,
                   elapsed_s=round(time.time() - t0, 1), api_calls=cl.calls, cache_hits=cl.cache_hits,
                   tokens_in=cl.tokens_in, tokens_out=cl.tokens_out,
                   reasoning_tokens=cl.reasoning_tokens, empty_contents=cl.empty_contents,
                   rate_limited=cl.rate_limited, quota_exhausted=cl.quota_exhausted,
                   quota_reset_msg=cl.quota_reset_msg, end_url=cl.base_url,
                   think=args.think, workers=args.workers, results=results)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nACCURACY = {acc:.1f}%  ({sum(1 for r in ok if r['judge_correct'])}/{len(ok)})")
    print(f"api_calls={cl.calls} cache_hits={cl.cache_hits} tokens={cl.tokens_in}/{cl.tokens_out} "
          f"(reasoning={cl.reasoning_tokens}) empty={cl.empty_contents} "
          f"qps_limit={cl.rate_limited} quota_exhausted={cl.quota_exhausted} "
          f"elapsed={payload['elapsed_s']}s")
    print(f"最终端点: {cl.base_url}")
    if cl.end_urls:
        print(f"  各端点实际命中: {cl.end_urls}")
    if cl.routed:
        print(f"  路由调用计数: {cl.routed}")
    if cl.quota_reset_msg:
        print(f"⚠️ 配额提示: {cl.quota_reset_msg}")
    if ok:
        from collections import Counter
        print("exit modes:", dict(Counter(r["exit_mode"] for r in ok)))
        print("actions c1:", dict(Counter(r["action_1"] for r in ok)))
        print("stop reasons:", dict(Counter(r.get("stop_reason") for r in ok)))
        nc = [r.get("n_conflicts_1", 0) for r in ok]
        print(f"crux 统计: 冲突数均值={sum(nc)/len(nc):.2f} "
              f"样本中出现 crux 的占比={sum(1 for x in nc if x > 0)/len(nc)*100:.0f}%")
    print(f"saved -> {out}")


if __name__ == "__main__":
    # ★ 必须把 main() 的返回码传给 shell（2=模型不可达，3=runtime 种子不完整）——
    #   否则调度脚本无法感知中止，会把「空产物」当成成功继续排队。
    sys.exit(main())
