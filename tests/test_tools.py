import pytest
from pydantic import ValidationError

from shopping_agent.models import Product
from shopping_agent.retrieval import HybridRetriever
from shopping_agent.tools import ShoppingTools, ToolCall, UnknownToolError


def test_tool_definitions_are_llm_function_calling_ready() -> None:
    definitions = ShoppingTools.definitions()
    assert len({definition.name for definition in definitions}) == len(definitions)
    for definition in definitions:
        assert definition.description
        assert definition.parameters["type"] == "object"
        assert "properties" in definition.parameters


def make_tools(tmp_path) -> ShoppingTools:
    products = [Product(id="one", title="鞋", category="运动鞋", description="", price=100, stock=3)]
    return ShoppingTools(HybridRetriever(products, tmp_path / "products.jsonl"))


def test_dispatcher_validates_and_executes_call(tmp_path) -> None:
    result = make_tools(tmp_path).execute(ToolCall(name="check_inventory", arguments={"product_id": "one"}))
    assert result == {"product_id": "one", "stock": 3, "available": True}


def test_dispatcher_rejects_unknown_tool(tmp_path) -> None:
    with pytest.raises(UnknownToolError, match="未知工具"):
        make_tools(tmp_path).execute(ToolCall(name="delete_products", arguments={}))


def test_dispatcher_rejects_invalid_or_extra_arguments(tmp_path) -> None:
    with pytest.raises(ValidationError):
        make_tools(tmp_path).execute(ToolCall(name="check_inventory", arguments={"product_id": "one", "admin": True}))
