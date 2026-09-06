from fastapi.testclient import TestClient

from shopping_agent.app import app


client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_agent_endpoint() -> None:
    response = client.post("/agent", json={"query": "白色通勤鞋", "max_price": 500})
    assert response.status_code == 200
    assert response.json()["hits"][0]["product"]["id"] == "shoe-001"

