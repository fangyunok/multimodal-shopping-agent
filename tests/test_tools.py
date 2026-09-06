from shopping_agent.tools import ShoppingTools


def test_tool_definitions_are_llm_function_calling_ready() -> None:
    definitions = ShoppingTools.definitions()
    assert len({definition.name for definition in definitions}) == len(definitions)
    for definition in definitions:
        assert definition.description
        assert definition.parameters["type"] == "object"
        assert "properties" in definition.parameters

