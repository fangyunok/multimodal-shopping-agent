from io import BytesIO

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
