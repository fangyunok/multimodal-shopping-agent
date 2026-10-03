"""会话记忆与上下文预算。

多轮 Agent 与单轮 Agent 的差别不在模型，而在两件事：**状态怎么跨轮累积**、**上下文怎么不爆**。
本模块只做这两件事，不碰检索与工具：

- ``SessionMemory``：把每轮的用户输入、工具观察、回答沉淀成 (1) 结构化槽位（预算 / 类目 /
  已见商品 / 排除项 / 偏好）与 (2) 事件流。新值覆盖旧值，未提及的约束自动继承。
- ``ContextBudget`` + ``window()``：在 token 预算内把事件流压成可发给模型的 messages。
  超预算时先折叠最旧的轮次为摘要，再逐条丢弃，最后硬保底保留槽位摘要与最近一轮，
  因此上下文永远不会因为历史过长而把当前问题挤出去。

token 计数使用启发式估算（CJK 每字 1 token，其余每 4 字符 1 token）而不是真实 BPE：
预算逻辑因此零依赖、可在纯 CPU 环境复现，且估算偏保守（偏高），触发压缩是安全的。
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .planner import Plan

CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")

# 无论预算多紧，都必须给“最近一条事件”留出的额度，否则上下文里就没有当前问题了。
MIN_TAIL_TOKENS = 32

PRODUCT_ID_PATTERN = re.compile(r"\b([a-z][a-z0-9]*-\d{2,})\b", re.IGNORECASE)

EXCLUSION_PATTERNS = (
    re.compile(r"(?:不要|不用|别|排除|除了)\s*([a-z][a-z0-9]*-\d{2,})", re.IGNORECASE),
    re.compile(r"(?:不要|不用|别|排除)\s*(?:这个|那款|这款)"),
)


def estimate_tokens(text: str) -> int:
    """启发式 token 估算：CJK 字符按 1 token，其余按 4 字符 1 token。

    刻意不做真实分词——预算逻辑要保持零依赖、结果可复现；估算偏保守，宁可早压缩。
    """
    if not text:
        return 0
    cjk = len(CJK_PATTERN.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def render(value: Any) -> str:
    """把任意工具返回值渲染成稳定文本（排序键 + 非 ASCII 原样保留），便于估算与摘要。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(render(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _product_ids_of(payload: Any) -> list[str]:
    """结构化提取工具返回值里的商品 ID，避免依赖对象字符串表示。"""
    ids: list[str] = []

    def inner_product_id(item: Any) -> str | None:
        product = getattr(item, "product", None)
        if product is None and isinstance(item, dict):
            product = item.get("product")
        if product is not None:
            value = getattr(product, "id", None)
            if value is None and isinstance(product, dict):
                value = product.get("id")
            if value:
                return str(value)
        value = getattr(item, "product_id", None) or getattr(item, "id", None)
        if value is None and isinstance(item, dict):
            value = item.get("product_id") or item.get("id")
        return str(value) if value else None

    items = payload if isinstance(payload, list) else [payload]
    for item in items:
        product_id = inner_product_id(item)
        if product_id and product_id not in ids:
            ids.append(product_id)
    return ids


class SessionState(BaseModel):
    """跨轮累积的结构化槽位。新值覆盖旧值，未提及的约束继承。"""

    max_price: float | None = Field(default=None, description="用户最近一次明确表达的最高预算")
    category: str | None = Field(default=None, description="用户最近一次明确表达的类目")
    seen_product_ids: list[str] = Field(default_factory=list, description="本轮会话出现过的商品 ID")
    last_candidates: list[str] = Field(
        default_factory=list, description="最近一次检索返回的候选商品 ID（按相关性排序）"
    )
    excluded_product_ids: list[str] = Field(default_factory=list, description="用户明确排除的商品 ID")
    preferences: list[str] = Field(default_factory=list, description="用户逐轮补充的偏好短语")
    turn_count: int = Field(default=0, description="已处理的用户轮次")

    def digest(self) -> str:
        """渲染成一行槽位摘要；始终注入上下文，保证压缩后关键约束不丢。"""
        parts: list[str] = []
        if self.max_price is not None:
            parts.append(f"预算上限 ¥{self.max_price:g}")
        if self.category:
            parts.append(f"类目 {self.category}")
        if self.preferences:
            parts.append("偏好 " + "、".join(self.preferences[-5:]))
        if self.last_candidates:
            parts.append("最近候选 " + "、".join(self.last_candidates[:6]))
        elif self.seen_product_ids:
            parts.append("已提及商品 " + "、".join(self.seen_product_ids[-8:]))
        if self.excluded_product_ids:
            parts.append("已排除 " + "、".join(self.excluded_product_ids))
        if not parts:
            return "[会话状态] 尚无已确认约束。"
        return "[会话状态] " + "；".join(parts) + "。"


class Turn(BaseModel):
    """事件流中的一条记录。``observation`` 是工具返回，序列化时降级为 user 角色。"""

    role: Literal["user", "assistant", "observation"]
    content: str
    tool: str | None = None
    step: int = 0

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)


class ContextBudget(BaseModel):
    """上下文预算。``reserve_tokens`` 预留给模型回复，不参与历史占用。"""

    max_tokens: int = Field(default=2048, ge=64)
    reserve_tokens: int = Field(default=256, ge=0)
    keep_recent_turns: int = Field(default=6, ge=1)

    @model_validator(mode="after")
    def _validate_reserve(self) -> ContextBudget:
        if self.reserve_tokens >= self.max_tokens:
            raise ValueError(
                f"reserve_tokens({self.reserve_tokens}) 必须小于 max_tokens({self.max_tokens})，"
                "否则上下文没有任何可用额度"
            )
        return self

    @property
    def usable_tokens(self) -> int:
        return max(0, self.max_tokens - self.reserve_tokens)


class ContextWindow(BaseModel):
    """一次压缩后的结果与统计，供调用方记录与观测。"""

    messages: list[dict[str, str]]
    summary: str
    kept_turns: int
    dropped_turns: int
    original_tokens: int
    tokens: int
    max_tokens: int
    compressed: bool

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 1.0
        return round(self.tokens / self.original_tokens, 4)

    def stats(self) -> dict[str, Any]:
        return {
            "kept_turns": self.kept_turns,
            "dropped_turns": self.dropped_turns,
            "original_tokens": self.original_tokens,
            "tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "compression_ratio": self.compression_ratio,
            "compressed": self.compressed,
        }


class SessionMemory:
    """单会话的事件流 + 槽位。并发安全由调用方保证（MCP 层按 session_id 分片）。"""

    def __init__(
        self,
        session_id: str = "default",
        budget: ContextBudget | None = None,
        system_prompt: str = "你是商品导购助手，价格与库存只能来自工具返回结果。",
    ) -> None:
        self.session_id = session_id
        self.budget = budget or ContextBudget()
        self.system_prompt = system_prompt
        self.state = SessionState()
        self.turns: list[Turn] = []
        self._summaries: list[str] = []

    # ---------- 写入 ----------

    def begin_turn(self, query: str, plan: Plan | None = None) -> None:
        """记录用户输入，并把 Planner 解析出的槽位合并进会话状态。"""
        self.state.turn_count += 1
        self.turns.append(Turn(role="user", content=query))
        if plan is not None:
            self.merge_plan(plan)
        self._absorb_exclusions(query)
        self._absorb_product_ids(query)

    def merge_plan(self, plan: Plan) -> None:
        """新值覆盖旧值，未提及的约束继承——这是多轮指代消解的核心规则。"""
        if plan.max_price is not None:
            self.state.max_price = plan.max_price
        if plan.category is not None:
            self.state.category = plan.category

    def record_preference(self, text: str) -> None:
        if text and text not in self.state.preferences:
            self.state.preferences.append(text)

    def record_observation(self, tool: str, payload: Any, step: int = 0) -> None:
        """记录工具观察，并顺带把出现的商品 ID 沉入槽位。"""
        text = render(payload)
        self.turns.append(Turn(role="observation", content=f"[{tool}] {text}", tool=tool, step=step))
        product_ids = _product_ids_of(payload)
        if tool == "search_products" and product_ids:
            # 检索结果按相关性排序，单独留一份供后续“有货吗 / 对比”这类追问直接复用。
            self.state.last_candidates = product_ids
        self._absorb_product_ids(text)

    def record_answer(self, text: str) -> None:
        self.turns.append(Turn(role="assistant", content=text))

    # ---------- 读取 ----------

    def window(self) -> ContextWindow:
        """在预算内产出 messages：折叠旧轮次 → 收缩摘要 → 逐条丢弃 → 硬保底留住最近一条。

        压缩是**就地且持久**的：被折叠的轮次会从 ``turns`` 移出并进入 ``_summaries``，
        因此重复调用不会产生重复摘要，上下文长度在整个会话里单调有界——这正是 ReAct
        每步都重算窗口却不会越算越大的原因。
        """
        digest = self.state.digest()
        usable = self.budget.usable_tokens
        dropped = 0
        compressed = False
        original_tokens = self._tokens(digest)

        # 阶段一：把最旧的轮次折叠成摘要，直到只剩 keep_recent_turns 条。
        while len(self.turns) > self.budget.keep_recent_turns and self._tokens(digest) > usable:
            oldest = self.turns.pop(0)
            self._summaries.append(f"{oldest.role}：{oldest.content[:60]}")
            dropped += 1
            compressed = True

        # 阶段二：摘要本身也可能吃掉预算，从最久远的一条开始收缩。
        while len(self._summaries) > 1 and self._tokens(digest) > usable:
            self._summaries.pop(0)
            compressed = True

        # 阶段三：仍然超预算就继续丢事件，但硬保底留住最后一条。
        while len(self.turns) > 1 and self._tokens(digest) > usable:
            self.turns.pop(0)
            dropped += 1
            compressed = True

        # 阶段四：单条消息本身就可能超过剩余额度（一次检索的观察可能有几百 token）。
        # 预算必须是硬边界，所以这里截断内容，而不是靠"保留最后一条"绕过去。
        available = usable - estimate_tokens(self._head(digest))
        if available >= MIN_TAIL_TOKENS:
            if len(self.turns) > 1 and sum(turn.tokens for turn in self.turns) > available:
                # 尾部整体放不下：优先保住最近一条的完整信息，其余丢弃。
                dropped += len(self.turns) - 1
                self.turns = self.turns[-1:]
                compressed = True
            if self.turns and self.turns[-1].tokens > available:
                self.turns[-1] = self.turns[-1].model_copy(
                    update={"content": self._truncate(self.turns[-1].content, available)}
                )
                compressed = True

        summary_text = " ".join(self._summaries)
        head = self._head(digest)
        messages: list[dict[str, str]] = [{"role": "system", "content": head}]
        messages += [
            {"role": "user" if turn.role == "observation" else turn.role, "content": turn.content}
            for turn in self.turns
        ]
        tokens = sum(estimate_tokens(message["content"]) for message in messages)

        return ContextWindow(
            messages=messages,
            summary=summary_text,
            kept_turns=len(self.turns),
            dropped_turns=dropped,
            original_tokens=original_tokens,
            tokens=tokens,
            max_tokens=self.budget.max_tokens,
            compressed=compressed,
        )

    def _head(self, digest: str) -> str:
        """上下文头部：系统提示 + 槽位摘要 + 历史摘要。

        必须按**拼接后的实际字符串**估算 token（分隔符与标签也会占额度），
        否则分量相加会低估 head，硬边界就会被几 token 的误差突破。
        """
        text = self.system_prompt + "\n" + digest
        if self._summaries:
            text += "\n[历史摘要] " + " ".join(self._summaries)
        return text

    def _tokens(self, digest: str) -> int:
        """当前上下文的 token 估算，含头部与全部事件。"""
        return estimate_tokens(self._head(digest)) + sum(turn.tokens for turn in self.turns)

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        """按估算比例截断文本并加省略号，保证结果不超过 ``max_tokens``。"""
        if max_tokens <= 0 or not text:
            return ""
        if estimate_tokens(text) <= max_tokens:
            return text
        ratio = max_tokens / max(1, estimate_tokens(text))
        clipped = text[: max(1, int(len(text) * ratio * 0.9))]
        while clipped and estimate_tokens(clipped + "…") > max_tokens:
            clipped = clipped[: int(len(clipped) * 0.8)]
        return clipped + "…"

    def over_budget(self) -> bool:
        """槽位摘要 + 最近一条事件已经吃满预算——此时应主动终止而不是继续灌上下文。"""
        floor = estimate_tokens(self._head(self.state.digest()))
        if self.turns:
            floor += self.turns[-1].tokens
        return floor > self.budget.usable_tokens

    def reset(self) -> None:
        self.state = SessionState()
        self.turns.clear()
        self._summaries.clear()

    # ---------- 内部 ----------

    def _absorb_exclusions(self, query: str) -> None:
        for pattern in EXCLUSION_PATTERNS:
            for match in pattern.finditer(query):
                if match.groups() and match.group(1):
                    product_id = match.group(1)
                    if product_id not in self.state.excluded_product_ids:
                        self.state.excluded_product_ids.append(product_id)
                elif self.state.seen_product_ids:
                    # “不要这个”这类指代：排除最近一次提到的商品。
                    latest = self.state.seen_product_ids[-1]
                    if latest not in self.state.excluded_product_ids:
                        self.state.excluded_product_ids.append(latest)

    def _absorb_product_ids(self, text: str) -> None:
        for product_id in PRODUCT_ID_PATTERN.findall(text or ""):
            if product_id not in self.state.seen_product_ids:
                self.state.seen_product_ids.append(product_id)
