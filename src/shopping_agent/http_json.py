"""有超时、响应上限且禁止重定向的 JSON 服务客户端。"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import http.client


class UpstreamError(RuntimeError):
    """对外只暴露稳定错误，不传播含凭据的 URL、响应或请求头。"""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class JSONServiceClient:
    def __init__(self, base_url: str, api_key: str = "", timeout_seconds: float = 10,
                 max_response_bytes: int = 262144) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("服务地址必须是无内嵌凭据、查询参数和片段的 HTTP(S) 地址")
        if not 0 < timeout_seconds <= 120 or max_response_bytes < 1:
            raise ValueError("服务超时或响应上限配置无效")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes

    def request(self, path: str, payload: dict | None = None) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            with urllib.request.build_opener(_NoRedirect).open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise UpstreamError("上游响应超过大小限制")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise UpstreamError("上游响应必须是 JSON 对象")
            return result
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise UpstreamError("上游请求失败或响应格式无效") from None
