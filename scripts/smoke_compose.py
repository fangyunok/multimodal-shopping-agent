"""验证已启动的隔离 Compose 栈：跨副本会话、探针、故障摘流及恢复。

此脚本会临时停止并重新启动该 Compose 项目的 Redis，仅用于测试栈。
用 COMPOSE_PROJECT_NAME 指向隔离项目；CI 自动创建并清理该栈。
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid


CONTAINER_REQUEST = """
import http.client, json, sys, urllib.request, urllib.error
data = None if sys.argv[2] == 'null' else sys.argv[2].encode()
request = urllib.request.Request('http://127.0.0.1:8000' + sys.argv[1], data=data,
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + sys.argv[3]})
try:
    response = urllib.request.urlopen(request, timeout=8)
except urllib.error.HTTPError as error:
    response = error
except (OSError, http.client.HTTPException):
    print(json.dumps({'status': 0, 'body': {}}))
    sys.exit(0)
print(json.dumps({'status': response.status, 'body': json.loads(response.read())}))
"""


def container_request(container: str, path: str, payload=None, token: str = "") -> dict:
    output = subprocess.check_output(
        ["docker", "exec", container, "python", "-c", CONTAINER_REQUEST, path, json.dumps(payload), token],
        text=True, timeout=15,
    )
    return json.loads(output)


def gateway_status(path: str = "/readyz") -> int:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8010" + path, timeout=8) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (OSError, http.client.HTTPException):
        return 0


def wait_until(check, message: str) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.5)
    raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=3)
    args = parser.parse_args()
    if not os.getenv("COMPOSE_PROJECT_NAME", "").startswith("stability-"):
        parser.error("请使用 COMPOSE_PROJECT_NAME=stability-开头的隔离测试项目")
    containers = subprocess.check_output(
        ["docker", "compose", "ps", "-q", "shopping-agent"], text=True,
    ).split()
    assert len(containers) == args.replicas and len(containers) >= 2
    wait_until(lambda: gateway_status() == 200, "网关未就绪")
    for container in containers:
        wait_until(lambda: container_request(container, "/readyz")["status"] == 200, "副本未就绪")
    tokens = list(json.loads(os.getenv("API_KEYS_JSON") or "{}").values())
    secured = os.getenv("AUTH_MODE", "demo") == "api_key"
    if secured:
        assert len(tokens) >= 2, "鉴权测试需要两个独立调用方"
        assert gateway_status("/tools") == 401
    token = tokens[0] if secured else ""
    session_id = "smoke-" + uuid.uuid4().hex
    first = container_request(containers[0], "/chat", {
        "session_id": session_id, "query": "预算 500 以内的运动鞋",
    }, token=token)
    assert first["status"] == 200, first
    second = container_request(containers[1], "/chat", {"session_id": session_id, "query": "有货吗"}, token=token)
    assert second["status"] == 200, second
    assert second["body"]["tool_calls"] == 1 and "有货" in second["body"]["answer"], second
    print("PASS: 多副本启动、默认镜像依赖、跨副本会话", flush=True)
    if secured:
        other = container_request(containers[-1], "/chat", {
            "session_id": session_id, "query": "有货吗",
        }, token=tokens[1])
        assert other["status"] == 200 and other["body"]["tool_calls"] == 0, other
        print("PASS: 未认证请求拒绝、不同调用方的同名会话隔离", flush=True)

    subprocess.run(["docker", "compose", "stop", "redis"], check=True, timeout=30)
    try:
        for container in containers:
            assert container_request(container, "/readyz")["status"] == 503
            assert container_request(container, "/healthz")["status"] == 200
        # 副本的 /healthz 仍返回 200，而网关返回 503，才能证明副本已被摘除。
        # 只测 /readyz 的 503 无法区分网关摘流与后端自己报告未就绪。
        wait_until(lambda: gateway_status("/healthz") == 503, "依赖故障后网关仍在发送流量")
        print("PASS: Redis 故障时返回 503、保留存活状态、网关摘流", flush=True)
    finally:
        subprocess.run(["docker", "compose", "start", "redis"], check=True, timeout=30)
    wait_until(lambda: gateway_status() == 200, "Redis 恢复后服务未恢复")
    print("PASS: 依赖恢复后重新接收流量", flush=True)


if __name__ == "__main__":
    main()
