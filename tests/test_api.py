from io import BytesIO
import asyncio
import threading

import httpx

from fastapi.testclient import TestClient
from PIL import Image

from shopping_agent.app import app


client = TestClient(app)


def test_demo_page() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "多模态购物 Agent" in response.text


def test_health() -> None:
    assert client.get("/health").json() == {
        "status": "ok", "retriever_backend": "baseline", "planner_backend": "rule"
    }


def test_tool_registry_exposes_json_schemas() -> None:
    response = client.get("/tools")
    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()}
    assert set(tools) == {"search_products", "compare_products", "check_inventory"}
    assert "max_price" in tools["search_products"]["parameters"]["properties"]


def test_agent_endpoint() -> None:
    response = client.post("/agent", json={"query": "白色通勤鞋", "max_price": 500})
    assert response.status_code == 200
    assert response.json()["hits"][0]["product"]["id"] == "shoe-001"


def test_json_endpoint_rejects_local_image_path() -> None:
    response = client.post("/agent", json={"query": "鞋", "image_path": "C:/private/image.png"})
    assert response.status_code == 400
    assert client.post("/search", json={"image_path": "C:/private/image.png"}).status_code == 400


def test_slow_image_request_does_not_block_health(monkeypatch) -> None:
    from shopping_agent import app as app_module
    from shopping_agent.models import AgentResponse

    entered, release = threading.Event(), threading.Event()

    def slow_run(self, request):
        entered.set()
        assert release.wait(3), "测试应在模型返回前完成健康检查"
        return AgentResponse(intent="search", answer="ok", tool_trace=[])

    monkeypatch.setattr(app_module.ShoppingAgent, "run", slow_run)
    stream = BytesIO()
    Image.new("RGB", (8, 8), "white").save(stream, format="PNG")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api:
            upload = asyncio.create_task(api.post(
                "/agent/image", files={"image": ("q.png", stream.getvalue(), "image/png")},
            ))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                response = await asyncio.wait_for(api.get("/healthz"), timeout=1)
                assert response.status_code == 200
                assert not upload.done(), "模型仍在执行时健康检查应已经返回"
            finally:
                release.set()
                await upload

    asyncio.run(scenario())


def test_image_upload_endpoint() -> None:
    stream = BytesIO()
    Image.new("RGB", (32, 32), "white").save(stream, format="PNG")
    response = client.post(
        "/agent/image",
        data={"query": "通勤运动鞋", "max_price": "500"},
        files={"image": ("query.png", stream.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["tool_trace"][0]["tool"] == "search_products"


def test_image_upload_rejects_wrong_media_type() -> None:
    response = client.post(
        "/agent/image",
        files={"image": ("query.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415

