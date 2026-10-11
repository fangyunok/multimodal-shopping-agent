export class ApiError extends Error {
  constructor(
    message: string,
    public status = 0,
  ) {
    super(message);
  }
}
export async function call<T>(
  path: string,
  token: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {}
    throw new ApiError(
      (
        {
          401: "访问令牌无效，请检查连接设置。",
          409: "会话已更新，请恢复会话后重试。",
          503: "服务暂时不可用，请稍后重试。",
        } as Record<number, string>
      )[response.status] ||
        (typeof detail === "string" ? detail : "请求失败，请检查输入后重试。"),
      response.status,
    );
  }
  return response.json();
}
export async function stream(
  path: string,
  token: string,
  body: unknown,
  signal: AbortSignal,
  onEvent: (event: string, data: any) => void,
) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok)
    throw new ApiError(
      response.status === 401
        ? "访问令牌无效，请检查连接设置。"
        : "服务暂时不可用，请稍后重试。",
      response.status,
    );
  if (!response.body) throw new Error("浏览器无法读取事件流。");
  const reader = response.body.getReader(),
    decoder = new TextDecoder();
  let buffer = "",
    finished = false;
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const lines = frame.split("\n");
        const event = lines
          .find((x) => x.startsWith("event:"))
          ?.slice(6)
          .trim();
        const data = lines
          .filter((x) => x.startsWith("data:"))
          .map((x) => x.slice(5))
          .join("\n");
        if (event && data) {
          if (event === "result") finished = true;
          if (event === "error") {
            const error = JSON.parse(data);
            throw new ApiError(error.detail, error.status);
          }
          onEvent(event, JSON.parse(data));
        }
      }
      if (done) break;
    }
    if (!finished)
      throw new Error("连接中断，结果尚未确认。请恢复会话检查结果后再重试。");
  } finally {
    reader.releaseLock();
  }
}
