"""HTTP 调用方身份与会话隔离；API 密钥只用于服务接入，不代替终端用户登录。"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re


def auth_config() -> tuple[str, dict[str, str]]:
    mode = os.getenv("AUTH_MODE", "demo").lower()
    if mode == "demo":
        return mode, {}
    if mode != "api_key":
        raise ValueError("AUTH_MODE 配置无效")
    try:
        keys = json.loads(os.getenv("API_KEYS_JSON", ""))
        if not isinstance(keys, dict) or not keys or len(keys) > 100:
            raise ValueError
        for principal, token in keys.items():
            if (not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", principal) or principal == "demo"
                    or not isinstance(token, str) or not token.isascii() or len(token) < 32):
                raise ValueError
        if len(set(keys.values())) != len(keys):
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("api_key 模式需要有效且互不重复的调用方密钥配置") from None
    return mode, keys


def identify(authorization: str, keys: dict[str, str]) -> str | None:
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token or not token.isascii():
        return None
    principal = None
    for name, expected in keys.items():
        if hmac.compare_digest(token, expected):
            principal = name
    return principal


def session_key(principal: str, session_id: str) -> str:
    if principal == "demo":
        return session_id
    value = json.dumps([principal, session_id], ensure_ascii=False).encode("utf-8")
    return "auth:v1:" + hashlib.sha256(value).hexdigest()
