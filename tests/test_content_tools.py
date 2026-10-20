import json

import pytest
from pydantic import ValidationError

from shopping_agent.content_tools import (
    ComplianceRequest,
    CopyRequest,
    MarketingProfile,
    MarketingTools,
    RewriteRequest,
    load_marketing_profiles,
)
from shopping_agent.models import Product
from shopping_agent.tools import ToolCall, UnknownToolError


def make_products() -> list[Product]:
    return [
        Product(id="shoe-1", title="轻便通勤运动鞋", category="运动鞋", description="",
                price=499.0, attributes={"材质": "网面", "适用场景": "通勤"}, rating=4.8, stock=5),
        Product(id="shoe-2", title="缓震跑步鞋", category="跑步鞋", description="",
                price=699.0, attributes={"中底": "缓震胶"}, stock=3),
    ]


def test_definitions_are_llm_function_calling_ready() -> None:
    definitions = MarketingTools.definitions()
    assert len({definition.name for definition in definitions}) == len(definitions)
    for definition in definitions:
        assert definition.description
        assert definition.parameters["type"] == "object"
        assert "properties" in definition.parameters


def test_generate_copy_is_grounded_in_product_facts() -> None:
    draft = MarketingTools(make_products()).generate_copy(
        CopyRequest(product_ids=["shoe-1"], channel="detail_page"))
    assert draft.cited_product_ids == ["shoe-1"]
    # 价格宣称必须携带证据，且取值与商品事实一致。
    price = next(evidence for evidence in draft.evidence if evidence.field == "price")
    assert price.value == "499" and price.product_id == "shoe-1"
    assert any("499" in point for point in draft.body_points)
    # 每条证据都能回溯到具体商品字段，不允许无出处的宣称。
    for evidence in draft.evidence:
        assert evidence.product_id in draft.cited_product_ids


def test_generate_copy_renders_channel_specific_structure() -> None:
    tools = MarketingTools(make_products())
    for channel in ("xiaohongshu", "detail_page", "short_video"):
        draft = tools.generate_copy(CopyRequest(product_ids=["shoe-1"], channel=channel))
        assert draft.channel == channel
        assert draft.body_points
        assert draft.call_to_action
        if channel == "xiaohongshu":
            assert draft.tags and draft.tags[0] == "运动鞋"
        if channel == "short_video":
            assert draft.title.endswith("？")


def test_marketing_profile_overrides_attribute_derived_points() -> None:
    profiles = {"shoe-1": MarketingProfile(product_id="shoe-1",
                                           selling_points=["一步上脚不磨脚"], target_audience="上班族")}
    tools = MarketingTools(make_products(), profiles)
    draft = tools.generate_copy(CopyRequest(product_ids=["shoe-1"], channel="short_video"))
    assert "一步上脚不磨脚" in draft.body_points
    assert "上班族" in draft.title
    assert any(evidence.field == "marketing.target_audience" for evidence in draft.evidence)


def test_generate_copy_rejects_unknown_product() -> None:
    with pytest.raises(KeyError, match="商品不存在"):
        MarketingTools(make_products()).generate_copy(
            CopyRequest(product_ids=["shoe-404"], channel="detail_page"))


def test_generate_copy_is_deterministic() -> None:
    tools = MarketingTools(make_products())
    request = CopyRequest(product_ids=["shoe-1", "shoe-2"], channel="detail_page")
    assert tools.generate_copy(request) == tools.generate_copy(request)


def test_rewrite_channel_preserves_citations_and_evidence() -> None:
    tools = MarketingTools(make_products())
    draft = tools.generate_copy(CopyRequest(product_ids=["shoe-1"], channel="xiaohongshu"))
    rewritten = tools.rewrite_channel(RewriteRequest(draft=draft, channel="short_video"))
    assert rewritten.channel == "short_video"
    assert rewritten.cited_product_ids == draft.cited_product_ids
    assert rewritten.evidence == draft.evidence
    assert rewritten.title != draft.title
    assert rewritten.tags == []


def test_check_compliance_blocks_absolute_terms() -> None:
    report = MarketingTools(make_products()).check_compliance(
        ComplianceRequest(text="全网最低价，销量第一"))
    assert report.passed is False
    terms = {violation.term for violation in report.violations}
    assert {"全网最低", "销量第一"} <= terms
    assert all(violation.severity == "block" for violation in report.violations)


def test_check_compliance_deduplicates_overlapping_terms() -> None:
    report = MarketingTools(make_products()).check_compliance(
        ComplianceRequest(text="全网最低价"))
    # 「全网最」「最低价」与「全网最低」重叠，只报最长匹配一次。
    assert [violation.term for violation in report.violations] == ["全网最低"]


def test_check_compliance_flags_ambiguous_terms_for_review() -> None:
    report = MarketingTools(make_products()).check_compliance(
        ComplianceRequest(text="第一次用就爱上了"))
    assert report.passed is True
    assert all(violation.severity == "review" for violation in report.violations)
    assert any(violation.term == "第一" for violation in report.violations)


def test_check_compliance_matches_ascii_terms_case_insensitively() -> None:
    report = MarketingTools(make_products()).check_compliance(
        ComplianceRequest(text="我们的产品是 NO.1"))
    assert report.passed is False
    assert any(violation.term == "no.1" for violation in report.violations)


def test_check_compliance_passes_clean_text() -> None:
    report = MarketingTools(make_products()).check_compliance(
        ComplianceRequest(text="网面材质，参考价 499 元，适合通勤场景。"))
    assert report.passed is True
    assert report.violations == []


def test_dispatcher_validates_and_executes_call() -> None:
    result = MarketingTools(make_products()).execute(
        ToolCall(name="check_compliance", arguments={"text": "参考价 499 元"}))
    assert result.passed is True


def test_dispatcher_rejects_unknown_tool() -> None:
    with pytest.raises(UnknownToolError, match="未知工具"):
        MarketingTools(make_products()).execute(ToolCall(name="generate_poster", arguments={}))


def test_dispatcher_rejects_invalid_or_extra_arguments() -> None:
    with pytest.raises(ValidationError):
        MarketingTools(make_products()).execute(
            ToolCall(name="check_compliance", arguments={"text": "好的", "admin": True}))


def test_load_marketing_profiles_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "marketing.jsonl"
    path.write_text(json.dumps({"product_id": "shoe-1", "selling_points": ["轻到无感"],
                                "target_audience": "上班族"}, ensure_ascii=False) + "\n"
                    + json.dumps({"product_id": "shoe-2"}, ensure_ascii=False) + "\n", encoding="utf-8")
    profiles = load_marketing_profiles(path)
    assert set(profiles) == {"shoe-1", "shoe-2"}
    assert profiles["shoe-1"].selling_points == ["轻到无感"]


def test_load_marketing_profiles_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "marketing.jsonl"
    line = json.dumps({"product_id": "shoe-1"}, ensure_ascii=False) + "\n"
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_marketing_profiles(path)
