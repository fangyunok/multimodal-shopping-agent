"""ReAct 循环：显式的 Thought → Action → Observation 轨迹与可中断的执行控制。

与 ``agent.py`` 的一次性规划不同，本模块把一次用户请求展开成**多步**工具调用：每一步都把
上一步的观察回注到决策上下文，工具失败不中断循环而是作为 observation 参与下一步决策，
从而支持「检索 → 逐候选校验 → 降级 → 收口」这类真实流程。

三件事是本模块的工程重点，也都是面试里会被追问的点：

1. **可审计轨迹**：每步都记下 thought / action / arguments / observation / error / 耗时，
   最终答案必须能回溯到具体工具返回，而不是模型自由生成。
2. **失控防护**：``max_steps`` 限制步数，``(tool, arguments)`` 去重检测原地打转，
   单步异常被分类成结构化错误而不是抛出——循环只会被"收口 / 超步 / 检测到循环 / 预算耗尽"终止。
3. **预算感知**：每步后重算上下文窗口，压缩后仍超预算即带 ``budget_exhausted`` 提前收口，
   避免历史无限增长把当前问题挤出去。

``Reasoner`` 是决策器抽象：``RuleReasoner`` 是确定性基线（离线可测、无外部依赖），
LLM 决策器只要实现同样的 ``decide()`` 即可替换，循环控制逻辑无需改动。
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from .planner import RulePlanner
from .models import SearchRequest
from .session import ContextBudget, SessionMemory, SessionState, estimate_tokens
from .tools import ShoppingTools, ToolCall, UnknownToolError

StopReason = Literal["final_answer", "max_steps", "loop_detected", "budget_exhausted", "no_candidates"]


class Observation(BaseModel):
    """一次工具调用的结果，回注给下一步决策。"""

    step: int
    tool: str
    arguments: dict[str, Any]
    payload: Any | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Decision(BaseModel):
    """决策器的一步输出：要么给出一个动作，要么给出最终答案。"""

    thought: str = Field(description="为什么做这一步，用于审计与调试")
    action: str | None = Field(default=None, description="要调用的工具名；为 None 表示收口")
    arguments: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = Field(default=None, description="action 为 None 时的最终回答")


class DecisionContext(BaseModel):
    """交给决策器的只读视图：当前问题、累积槽位、已产生的观察、步数。"""

    query: str
    intent: str
    state: SessionState
    observations: list[Observation]
    step: int
    max_steps: int
    history: list[dict[str, str]] = Field(default_factory=list)


class ReActStep(BaseModel):
    index: int
    thought: str
    action: str | None
    arguments: dict[str, Any]
    observation: Any | None = None
    error: str | None = None
    latency_ms: float = 0.0


class ReActResult(BaseModel):
    answer: str
    intent: str
    steps: list[ReActStep]
    citations: list[str] = Field(default_factory=list)
    stop_reason: StopReason = "final_answer"
    tool_calls: int = 0
    context: dict[str, Any] = Field(default_factory=dict)

    def trace(self) -> list[dict[str, Any]]:
        """扁平化轨迹，与 ``agent.py`` 的 ``tool_trace`` 字段保持一致以便复用评测。"""
        return [
            {
                "step": step.index,
                "tool": step.action,
                "arguments": step.arguments,
                **({"error": step.error} if step.error else {"result": step.observation}),
            }
            for step in self.steps
        ]


class Reasoner(Protocol):
    def decide(self, context: DecisionContext) -> Decision: ...


def _hit_fields(hit: Any) -> dict[str, Any]:
    """把 SearchHit（或已序列化的 dict）统一成扁平的候选描述。"""
    if isinstance(hit, dict):
        return hit
    product = getattr(hit, "product", None)
    if product is None:
        return {"value": hit}
    return {
        "product_id": product.id,
        "title": product.title,
        "category": product.category,
        "price": product.price,
        "rating": product.rating,
        "stock": product.stock,
        "score": getattr(hit, "score", None),
        "reasons": list(getattr(hit, "reasons", []) or []),
    }


def _product_fields(item: Any) -> dict[str, Any]:
    """把 Product（或已序列化的 dict）统一成扁平的展示字段。"""
    if isinstance(item, dict):
        return item
    return {
        "product_id": item.id,
        "title": item.title,
        "category": item.category,
        "price": item.price,
        "rating": item.rating,
        "stock": item.stock,
    }


def extract_candidates(payload: Any, limit: int = 3) -> list[dict[str, Any]]:
    """把 search_products 的返回值统一成最多 ``limit`` 个候选描述。"""
    if not isinstance(payload, list):
        return []
    candidates = [_hit_fields(hit) for hit in payload]
    return [candidate for candidate in candidates if candidate.get("product_id")][:limit]


class RuleReasoner:
    """确定性 ReAct 决策器：把导购技能拆成可审计的多步动作。

    策略：检索 → 逐个校验候选库存 → 在可用候选里收口。不依赖任何外部模型，
    因此在 CPU 干净克隆上也能复现完整的多步轨迹；LLM 决策器可平替。
    """

    backend = "rule"

    def __init__(self, max_candidates: int = 3) -> None:
        self.max_candidates = max_candidates

    def decide(self, context: DecisionContext) -> Decision:
        searches = [item for item in context.observations if item.tool == "search_products"]
        inventories = [item for item in context.observations if item.tool == "check_inventory"]
        compares = [item for item in context.observations if item.tool == "compare_products"]

        # 多轮追问的第一层价值：库存与比价可以直接复用会话记住的候选，不必重新检索。
        if context.intent == "inventory":
            if inventories:
                return self._inventory_outcome(inventories[-1])
            target = context.state.last_candidates[0] if context.state.last_candidates else None
            if target:
                return Decision(
                    thought=f"用户追问库存，直接复用会话记住的候选 {target}，无需重新检索。",
                    action="check_inventory",
                    arguments={"product_id": target},
                )
            return Decision(
                thought="用户追问库存但会话里没有可用候选，需要先确认商品。",
                answer="请告诉我要查询库存的商品 ID，或先描述一下你在找什么。",
            )

        if context.intent == "compare":
            if compares:
                return self._compare_outcome(compares[-1])
            targets = context.state.last_candidates[:2]
            if len(targets) >= 2:
                return Decision(
                    thought=f"用户要求对比，使用会话记住的候选 {targets}，无需重新检索。",
                    action="compare_products",
                    arguments={"product_ids": targets},
                )
            return Decision(
                thought="比价至少需要两个商品，先向用户确认。",
                answer="请至少提供两个商品 ID 进行对比。",
            )

        if not searches:
            arguments: dict[str, Any] = {"query": context.query, "top_k": self.max_candidates}
            if context.state.excluded_product_ids:
                arguments["excluded_product_ids"] = list(context.state.excluded_product_ids)
            if context.state.max_price is not None:
                arguments["max_price"] = context.state.max_price
            if context.state.category is not None:
                arguments["category"] = context.state.category
            inherited = [
                name
                for name, value in (("预算", context.state.max_price), ("类目", context.state.category))
                if value is not None
            ]
            note = f"，继承会话已确认的{'与'.join(inherited)}约束" if inherited else ""
            return Decision(
                thought=f"先把用户描述解析成检索请求{note}。",
                action="search_products",
                arguments=arguments,
            )

        if not searches[-1].ok:
            return Decision(
                thought="检索工具报错，无法继续，直接向用户说明并给出可执行的下一步建议。",
                answer="检索服务暂时不可用，请稍后重试，或换一种描述方式。",
            )

        candidates = [
            candidate for candidate in self._candidates(searches[-1].payload)
            if candidate["product_id"] not in context.state.excluded_product_ids
        ]
        if not candidates:
            relaxation = "放宽预算" if context.state.max_price is not None else "换一种描述"
            return Decision(
                thought="检索没有返回任何候选，问题出在约束而非工具，收口并给出放宽路径。",
                answer=f"没有找到满足条件的商品，可以尝试{relaxation}，或告诉我更具体的品牌与型号。",
            )

        checked = {
            item.arguments.get("product_id")
            for item in inventories
        }
        pending = [candidate for candidate in candidates if candidate["product_id"] not in checked]
        if pending:
            target = pending[0]
            position = candidates.index(target) + 1
            remaining = len(pending) - 1
            return Decision(
                thought=(
                    f"共 {len(candidates)} 个候选，逐个校验库存；"
                    f"本轮检查第 {position} 个（{target['product_id']}），剩余 {remaining} 个待查。"
                ),
                action="check_inventory",
                arguments={"product_id": target["product_id"]},
            )

        available = self._available(candidates, inventories)
        if not available:
            if any(not item.ok for item in inventories):
                return Decision(thought="部分库存检查失败，不能把未知库存当成无货。",
                                answer="已找到相关商品，但暂时无法确认可用库存，请稍后重试。")
            return Decision(
                thought="候选都已校验且全部无货，收口并给出降级建议。",
                answer="检索到了相关商品，但目前都没有库存，建议放宽预算或换一个类目。",
            )

        return Decision(thought="候选库存校验完成，在可用商品里收口。",
                        answer=self._compose(candidates, available, inventories))


    # ---------- 内部 ----------

    @staticmethod
    def _inventory_outcome(observation: Observation) -> Decision:
        product_id = observation.arguments.get("product_id")
        if not observation.ok:
            return Decision(thought="库存工具报错，收口并说明。", answer="库存服务暂时不可用，请稍后重试。")
        payload = observation.payload if isinstance(observation.payload, dict) else {}
        if payload.get("available"):
            answer = f"商品 [{product_id}] 有货，剩余 {payload.get('stock')} 件。"
        else:
            answer = f"商品 [{product_id}] 暂时无货，可以让我换一个候选看看。"
        return Decision(thought=f"库存已确认，收口（{product_id}）。", answer=answer)

    @staticmethod
    def _compare_outcome(observation: Observation) -> Decision:
        if not observation.ok:
            return Decision(thought="对比工具不可用，不能引用未确认库存。",
                            answer="对比服务暂时不可用，当前无法确认商品库存，请稍后重试。")
        products = observation.payload if observation.ok else None
        if not isinstance(products, list) or len(products) < 2:
            return Decision(thought="有效商品不足两个，收口并提示。", answer="请至少提供两个有效商品 ID 进行对比。")
        fields = [_product_fields(item) for item in products]
        lines = [
            f"{item['title']} [{item['product_id']}]：¥{item['price']:g}，评分 {item['rating']:.1f}，库存 {item['stock']}"
            for item in fields
        ]
        best = max(fields, key=lambda item: (item["rating"], -item["price"]))
        return Decision(
            thought="对比完成，给出结构化差异与优先建议。",
            answer="；".join(lines) + f"。综合评分与价格，优先考虑 {best['title']}。",
        )

    def _candidates(self, payload: Any) -> list[dict[str, Any]]:
        return extract_candidates(payload, self.max_candidates)

    @staticmethod
    def _available(candidates: list[dict[str, Any]], inventories: list[Observation]) -> list[dict[str, Any]]:
        in_stock = {
            item.arguments.get("product_id")
            for item in inventories
            if item.ok and isinstance(item.payload, dict) and item.payload.get("available")
        }
        return [candidate for candidate in candidates if candidate["product_id"] in in_stock]

    @staticmethod
    def _compose(candidates: list[dict[str, Any]], available: list[dict[str, Any]],
                 inventories: list[Observation] | None = None) -> str:
        available_ids = {candidate["product_id"] for candidate in available}
        unavailable_ids = {item.arguments.get("product_id") for item in inventories or []
                           if item.ok and isinstance(item.payload, dict) and item.payload.get("stock") == 0}
        lines: list[str] = []
        for candidate in candidates:
            mark = ("有货" if candidate["product_id"] in available_ids else
                    "无货" if candidate["product_id"] in unavailable_ids else "库存未知")
            lines.append(
                f"{candidate['title']} [{candidate['product_id']}]：¥{candidate['price']:g}，"
                f"评分 {candidate['rating']:.1f}，{mark}"
            )
        best = max(available, key=lambda item: (item["rating"], -item["price"]))
        reason = "、".join(best.get("reasons") or []) or "综合评分与价格更优"
        return "；".join(lines) + f"。综合来看优先考虑 {best['title']} [{best['product_id']}]，因为{reason}。"


class ReActAgent:
    """带会话记忆与上下文预算的 ReAct 执行器。"""

    def __init__(
        self,
        tools: ShoppingTools,
        reasoner: Reasoner | None = None,
        max_steps: int = 6,
        budget: ContextBudget | None = None,
        session: SessionMemory | None = None,
    ) -> None:
        self.tools = tools
        self.reasoner = reasoner or RuleReasoner()
        self.max_steps = max_steps
        self.budget = budget or ContextBudget()
        self.session = session or SessionMemory(budget=self.budget)
        self.planner = RulePlanner([product.category for product in tools.retriever.products])

    def run(self, query: str, intent: str = "auto") -> ReActResult:
        plan = self.planner.plan(query, intent)
        self.session.begin_turn(query, plan)
        resolved_intent = intent if intent != "auto" else plan.intent

        steps: list[ReActStep] = []
        observations: list[Observation] = []
        seen: set[str] = set()
        stop_reason: StopReason = "max_steps"
        answer = ""
        window = self.session.window()

        for index in range(self.max_steps):
            if self.session.over_budget():
                stop_reason = "budget_exhausted"
                answer = "对话上下文已超出预算，请开启新一轮会话继续。"
                break

            context = DecisionContext(
                query=query,
                intent=resolved_intent,
                state=self.session.state.model_copy(deep=True),
                observations=list(observations),
                step=index,
                max_steps=self.max_steps,
                history=window.messages,
            )
            decision = self.reasoner.decide(context)

            if decision.action is None:
                answer = decision.answer or ""
                steps.append(
                    ReActStep(
                        index=index,
                        thought=decision.thought,
                        action=None,
                        arguments={},
                    )
                )
                stop_reason = "final_answer"
                break

            signature = self._signature(decision.action, decision.arguments)
            if signature in seen:
                stop_reason = "loop_detected"
                answer = answer or "检测到重复的工具调用，已停止循环。"
                break
            seen.add(signature)

            started = time.perf_counter()
            observation, error = self._execute(decision.action, decision.arguments, index)
            latency_ms = (time.perf_counter() - started) * 1000
            observations.append(observation)
            if observation.ok:
                self.session.record_observation(decision.action, observation.payload, index)
            steps.append(
                ReActStep(
                    index=index,
                    thought=decision.thought,
                    action=decision.action,
                    arguments=observation.arguments,
                    observation=observation.payload,
                    error=error,
                    latency_ms=round(latency_ms, 3),
                )
            )
            window = self.session.window()
        else:
            # 循环自然耗尽步数：用最后一次观察兜底收口，避免返回空答案。
            answer = answer or "已达到最大步数上限，请缩小问题范围后重试。"
            stop_reason = "max_steps"

        if answer:
            self.session.record_answer(answer)

        citations = [product_id for product_id in self._citations(observations)
                     if f"[{product_id}]" in answer and product_id not in self.session.state.excluded_product_ids]
        stats = window.stats()
        stats.update({"reasoner_backend": getattr(self.reasoner, "backend", "custom"),
                      "fallback_used": bool(getattr(self.reasoner, "fallback_used", False)),
                      "inventory_source": self.tools.inventory.source})
        return ReActResult(
            answer=answer,
            intent=resolved_intent,
            steps=steps,
            citations=citations,
            stop_reason=stop_reason,
            tool_calls=sum(1 for step in steps if step.action),
            context=stats,
        )

    # ---------- 内部 ----------

    @staticmethod
    def _signature(action: str, arguments: dict[str, Any]) -> str:
        canonical = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        return f"{action}:{canonical}"

    def _execute(self, action: str, arguments: dict[str, Any], index: int) -> tuple[Observation, str | None]:
        """执行工具并把异常翻译成结构化错误——失败是观察的一部分，不是循环的终点。"""
        try:
            arguments = self._constrain(action, arguments)
            payload = self.tools.execute(ToolCall(name=action, arguments=arguments))
        except ValidationError:
            return Observation(step=index, tool=action, arguments=arguments, error="invalid_arguments"), "invalid_arguments"
        except KeyError:
            return Observation(step=index, tool=action, arguments=arguments, error="not_found"), "not_found"
        except UnknownToolError:
            return Observation(step=index, tool=action, arguments=arguments, error="unknown_tool"), "unknown_tool"
        except PermissionError:
            return Observation(step=index, tool=action, arguments=arguments, error="constraint_violation"), "constraint_violation"
        except Exception as error:  # 兜底：任何工具异常都不得中断循环
            return Observation(step=index, tool=action, arguments=arguments, error="tool_failure"), "tool_failure"
        return Observation(step=index, tool=action, arguments=arguments, payload=payload), None

    def _constrain(self, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        state = self.session.state
        if action == "search_products":
            request = SearchRequest.model_validate(arguments)
            if request.image_path:
                raise PermissionError("模型不能读取本地文件")
            prices = [price for price in (request.max_price, state.max_price) if price is not None]
            return request.model_copy(update={
                "max_price": min(prices) if prices else None,
                "category": state.category or request.category,
                "top_k": min(request.top_k, 3),
                "excluded_product_ids": sorted(set(request.excluded_product_ids + state.excluded_product_ids)),
            }).model_dump()
        if action in {"check_inventory", "compare_products"}:
            # 参数类型仍由工具 Schema 校验；这里只校验模型不能放宽的业务边界。
            ids = [arguments.get("product_id")] if action == "check_inventory" else arguments.get("product_ids", [])
            if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
                return arguments
            for product_id in ids:
                product = self.tools.by_id.get(product_id)
                if product_id in state.excluded_product_ids or (product and (
                    (state.max_price is not None and product.price > state.max_price)
                    or (state.category and product.category != state.category)
                )):
                    raise PermissionError("工具目标违反会话约束")
        return arguments

    def _citations(self, observations: list[Observation]) -> list[str]:
        citations: list[str] = []
        for item in observations:
            if not item.ok:
                continue
            if item.tool == "search_products":
                candidates = extract_candidates(item.payload, limit=10)
            elif item.tool == "check_inventory" and isinstance(item.payload, dict):
                candidates = [item.payload]
            elif item.tool == "compare_products" and isinstance(item.payload, list):
                candidates = [_product_fields(product) for product in item.payload]
            else:
                candidates = []
            for candidate in candidates:
                product_id = candidate["product_id"]
                if product_id not in citations:
                    citations.append(product_id)
        return citations


def estimate_prompt_tokens(result: ReActResult) -> int:
    """整个轨迹的 token 估算，用于对比压缩收益。"""
    return sum(estimate_tokens(json.dumps(step.model_dump(), ensure_ascii=False, default=str)) for step in result.steps)

