"""营销内容工具：受商品事实约束的确定性文案生成、渠道改写与合规检查。

与购物工具同构的三件套：JSON Schema 工具定义、参数严格校验的 Dispatcher、
可复现的规则基线。默认路径零模型依赖——生成只消费商品目录与营销画像里的
既有事实，每条宣称都携带证据链（字段名 + 取值 + 商品 ID），供忠实度评测
回溯；LLM 生成后端在 content_generation 中按同一契约替换。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import Product
from .tools import ToolCall, ToolDefinition, UnknownToolError

Channel = Literal["xiaohongshu", "detail_page", "short_video"]

CHANNEL_LABELS: dict[str, str] = {
    "xiaohongshu": "小红书种草文",
    "detail_page": "电商详情页",
    "short_video": "短视频口播脚本",
}

# 渠道 × 语气的行动号召模板；改文案不改事实。
_CALL_TO_ACTION: dict[tuple[str, str], str] = {
    ("xiaohongshu", "friendly"): "喜欢的话点商品卡看看",
    ("xiaohongshu", "professional"): "点击商品卡查看详情",
    ("detail_page", "friendly"): "进入详情页了解全部卖点",
    ("detail_page", "professional"): "查看商品详情",
    ("short_video", "friendly"): "点击下方链接看同款",
    ("short_video", "professional"): "点击下方链接查看商品",
}

# 广告法绝对化用语与虚假承诺：语境无关，直接拦截（block）。
_BLOCK_TERMS: dict[str, str] = {
    "国家级": "absolute", "世界级": "absolute", "最高级": "absolute",
    "最佳": "absolute", "最优": "absolute", "最先进": "absolute", "最便宜": "absolute",
    "最优秀": "absolute", "最低价": "absolute", "全网最低": "absolute", "全网最": "absolute",
    "史上最": "absolute", "顶级": "absolute", "极品": "absolute", "王牌": "absolute",
    "绝对": "absolute", "第一品牌": "absolute", "销量第一": "absolute", "销量冠军": "absolute",
    "永久有效": "promise", "永久保修": "promise", "万能": "promise",
    "百分百有效": "promise", "100%有效": "promise", "no.1": "absolute", "top1": "absolute",
}

# 语境歧义词：如「第一次使用」不违规、「销量第一」违规，需人工复核（review）。
_REVIEW_TERMS: tuple[str, ...] = ("第一", "最", "唯一", "首个", "独家", "首选", "冠军")


class MarketingProfile(BaseModel):
    """营销素材数据契约：卖点与目标人群可离线标注，缺失时从商品属性推导。"""
    model_config = ConfigDict(extra="forbid")
    product_id: str = Field(min_length=1, max_length=128)
    selling_points: list[str] = Field(default_factory=list, max_length=5)
    target_audience: str = Field(default="", max_length=40)


class CopyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_ids: list[str] = Field(min_length=1, max_length=3)
    channel: Channel
    tone: Literal["friendly", "professional"] = "friendly"
    max_points: int = Field(default=3, ge=1, le=5)


class Evidence(BaseModel):
    """文案事实的证据链：字段名、取值与商品 ID，忠实度评测按此回溯。"""
    model_config = ConfigDict(extra="forbid")
    field: str
    value: str
    product_id: str


class ContentDraft(BaseModel):
    """结构化文案草稿：渠道字段固定，格式符合率可由 schema 直接判定。"""
    model_config = ConfigDict(extra="forbid")
    channel: Channel
    title: str = Field(min_length=1, max_length=40)
    body_points: list[str] = Field(min_length=1, max_length=5)
    call_to_action: str = Field(min_length=1, max_length=40)
    tags: list[str] = Field(default_factory=list, max_length=8)
    cited_product_ids: list[str] = Field(min_length=1, max_length=3)
    evidence: list[Evidence] = Field(default_factory=list)


class RewriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    draft: ContentDraft
    channel: Channel


class ComplianceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)


class ComplianceViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    term: str
    category: Literal["absolute", "promise"]
    severity: Literal["block", "review"]
    excerpt: str
    suggestion: str


class ComplianceReport(BaseModel):
    """block 级违规直接不通过；review 级需要人工复核但不拦截。"""
    model_config = ConfigDict(extra="forbid")
    passed: bool
    violations: list[ComplianceViolation]


def load_marketing_profiles(path: Path) -> dict[str, MarketingProfile]:
    """从 JSONL 加载营销画像；重复商品 ID 视为数据错误而非覆盖。"""
    profiles: dict[str, MarketingProfile] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                profile = MarketingProfile.model_validate(json.loads(line))
            except ValueError as error:
                raise ValueError(f"营销画像第 {line_number} 行格式无效: {error}") from None
            if profile.product_id in profiles:
                raise ValueError(f"营销画像第 {line_number} 行商品 ID 重复: {profile.product_id}")
            profiles[profile.product_id] = profile
    return profiles


class MarketingTools:
    """营销内容工具集：与 ShoppingTools 同一 Dispatcher 契约，可独立替换与评测。"""

    def __init__(self, products: list[Product], profiles: dict[str, MarketingProfile] | None = None):
        self.by_id = {product.id: product for product in products}
        self.profiles = profiles or {}

    def generate_copy(self, request: CopyRequest) -> ContentDraft:
        products: list[Product] = []
        for product_id in request.product_ids:
            if product_id not in self.by_id:
                raise KeyError(f"商品不存在: {product_id}")
            products.append(self.by_id[product_id])
        primary = products[0]
        points: list[str] = []
        evidence: list[Evidence] = []
        for product in products:
            for text, field, value in self._selling_points(product):
                if len(points) >= request.max_points:
                    break
                points.append(text)
                evidence.append(Evidence(field=field, value=value, product_id=product.id))
        # 价格宣称必须有证据；没有卖点时价格点也保证草稿非空。
        for product in products:
            if len(points) >= request.max_points:
                break
            price_text = f"参考价 ¥{product.price:g}"
            points.append(price_text)
            evidence.append(Evidence(field="price", value=f"{product.price:g}", product_id=product.id))
        evidence.insert(0, Evidence(field="title", value=primary.title, product_id=primary.id))
        # 标题证据必须与草稿实际宣称一致：短视频标题是钩子问句而非商品标题。
        profile = self.profiles.get(primary.id)
        if request.channel == "short_video":
            audience = profile.target_audience if profile else ""
            title = f"还在纠结{audience or primary.category}怎么选？"
            title_evidence = (Evidence(field="marketing.target_audience", value=audience, product_id=primary.id)
                              if audience else
                              Evidence(field="category", value=primary.category, product_id=primary.id))
        else:
            title_text = primary.title or primary.category
            title = title_text[:20] if request.channel == "xiaohongshu" else title_text[:40]
            title_evidence = Evidence(field="title", value=title_text, product_id=primary.id)
        evidence.insert(0, title_evidence)
        return ContentDraft(
            channel=request.channel,
            title=title,
            body_points=points,
            call_to_action=_CALL_TO_ACTION[(request.channel, request.tone)],
            tags=self._tags(primary) if request.channel == "xiaohongshu" else [],
            cited_product_ids=[product.id for product in products],
            evidence=evidence,
        )

    def rewrite_channel(self, request: RewriteRequest) -> ContentDraft:
        """跨渠道改写：保留引用与证据链，只重排渠道相关字段（标题钩子与行动号召）。"""
        draft, channel = request.draft, request.channel
        title = draft.title[:40]
        points = draft.body_points
        if channel == "short_video":
            title = f"{draft.title[:20]}，值得入手吗？"
            points = [point[:16] for point in draft.body_points]
        return ContentDraft(
            channel=channel,
            title=title or draft.cited_product_ids[0],
            body_points=points,
            call_to_action=_CALL_TO_ACTION[(channel, "friendly")],
            tags=draft.tags if channel == "xiaohongshu" else [],
            cited_product_ids=draft.cited_product_ids,
            evidence=draft.evidence,
        )

    def check_compliance(self, request: ComplianceRequest) -> ComplianceReport:
        """词表扫描：block 级按最长匹配去重，review 级命中 block 区间时不再重复报告。"""
        text = request.text
        lowered = text.lower()
        violations: list[ComplianceViolation] = []
        claimed: list[tuple[int, int]] = []
        # 命中区间按词长降序登记，短词落在长词区间内视为同一处违规。
        for term in sorted(_BLOCK_TERMS, key=len, reverse=True):
            haystack = lowered if term.isascii() else text
            start = haystack.find(term)
            while start != -1:
                end = start + len(term)
                if not any(start < bound and end > begin for begin, bound in claimed):
                    claimed.append((start, end))
                    violations.append(ComplianceViolation(
                        term=term, category=_BLOCK_TERMS[term], severity="block",
                        excerpt=text[max(0, start - 6):end + 6],
                        suggestion="改为可验证的具体表述，如参数、认证或数据来源",
                    ))
                start = haystack.find(term, end)
        for term in _REVIEW_TERMS:
            start = text.find(term)
            while start != -1:
                end = start + len(term)
                if not any(start < bound and end > begin for begin, bound in claimed):
                    claimed.append((start, end))
                    violations.append(ComplianceViolation(
                        term=term, category="absolute", severity="review",
                        excerpt=text[max(0, start - 6):end + 6],
                        suggestion="语境歧义词，结合上下文确认是否为绝对化用语",
                    ))
                start = text.find(term, end)
        return ComplianceReport(passed=not any(v.severity == "block" for v in violations),
                                violations=violations)

    def _selling_points(self, product: Product) -> list[tuple[str, str, str]]:
        """返回 (文案点, 证据字段, 证据取值)；画像优先，其次商品属性，最后品类兜底。"""
        profile = self.profiles.get(product.id)
        if profile and profile.selling_points:
            return [(point, "marketing.selling_points", point) for point in profile.selling_points]
        if product.attributes:
            return [(f"{key}：{value}", f"attributes.{key}", value)
                    for key, value in product.attributes.items()]
        return [(f"品类：{product.category}", "category", product.category)]

    def _tags(self, primary: Product) -> list[str]:
        tags = [primary.category] + list(primary.attributes.values())
        return tags[:6]

    def execute(self, call: ToolCall):
        if call.name == "generate_copy":
            return self.generate_copy(CopyRequest.model_validate(call.arguments))
        if call.name == "rewrite_channel":
            return self.rewrite_channel(RewriteRequest.model_validate(call.arguments))
        if call.name == "check_compliance":
            return self.check_compliance(ComplianceRequest.model_validate(call.arguments))
        raise UnknownToolError(f"未知工具: {call.name}")

    @staticmethod
    def definitions() -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="generate_copy",
                description="根据商品事实生成指定渠道的营销文案草稿；卖点与价格只能来自"
                            "商品目录或营销画像，返回结构化草稿与逐条证据链。",
                parameters=CopyRequest.model_json_schema(),
            ),
            ToolDefinition(
                name="rewrite_channel",
                description="把已有文案草稿改写到其他渠道，保留引用商品与证据链，只重排"
                            "渠道相关字段。",
                parameters=RewriteRequest.model_json_schema(),
            ),
            ToolDefinition(
                name="check_compliance",
                description="按广告法绝对化用语与虚假承诺词表检查文案；block 级违规直接"
                            "不通过，review 级语境歧义词需人工复核。",
                parameters=ComplianceRequest.model_json_schema(),
            ),
        ]
