# 购物前端使用与开发

前端采用 React + TypeScript + Vite，后端保持 FastAPI；Docker 构建时自动打包前端，运行时不需要 Node.js。前端通过同源接口访问服务，模型密钥只配置在服务端。

## 最快启动

在项目根目录执行：

```bash
git pull
docker compose up --build -d
```

打开 http://localhost:8010 。这是你电脑上的地址；要让其他人访问，需要另行部署服务器和配置 HTTPS。

## 没有 Docker

需要 Python 3.10+、Node.js 22+。首次安装及每次更新前端后运行：

```bash
python -m venv .venv
# Windows PowerShell：.venv\Scripts\Activate.ps1
# macOS / Linux：source .venv/bin/activate
python -m pip install -e .
npm ci --prefix frontend
npm run build --prefix frontend
python -m uvicorn shopping_agent.app:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 。前端源代码修改期间可另开终端运行 `npm run dev --prefix frontend`，访问终端显示的 Vite 地址；开发代理将请求转发至 8000 端口。

## 建议体验顺序

1. 点击“500 元以内的运动鞋”，查看商品卡片与已核验库存。
2. 勾选两件商品，点击“对比所选商品”，查看结构化比较表。
3. 点击指定商品的“查库存”，确认查询的是该商品。
4. 点击“排除”，继续推荐；左侧显示已排除商品。
5. 将预算改为 300，或清空预算/类目后发送；明确修改的控件优先于该轮文字解析，未修改的条件由对话继承或更新。
6. 上传 JPEG/PNG/WebP 图片（最大 5 MB）并配合文字。默认 5 件演示商品没有图片，因此会明确提示仅按文字检索；接入带商品图片的图文索引后才能验证图片相似度效果。
7. 刷新同一标签页，恢复最近 20 轮。文本历史与购物状态同一次 CAS 提交，独立于模型上下文压缩。Redis 会话过期后重新开始；默认内存模式重启即丢失。

页面不提供订单、支付或跨商家实时爬取。没有真实商品图片时展示类目示意，不冒充商品实拍。规则决策和演示库存均有明确标记。真实库存配置见 `docs/BUSINESS_INTEGRATION.md`（以及 README 对应说明）。

## 接口与状态

- `GET /ui/config`：目录类目、商品数量、图片检索可用性、库存来源。
- `POST /chat`：保留原有 JSON 接口，增加商品卡片、公开工具执行状态、购物约束。
- `POST /chat/stream`：SSE 事件 `progress`（工具开始/结束）、`result`（会话提交成功后的最终结果）、`error`（本轮失败）；每 10 秒保活。HTTP 建连之后发生的错误使用 `error` 事件传递状态码。当前为工具进度流，非模型逐字输出。
- `POST /chat/image`：multipart 图片与会话 ID；可附 `filters` JSON，显式 `null` 清空预算/类目，省略字段表示继承。
- `GET /chat/history?session_id=...`：返回调用方隔离的最近历史和购物条件，不暴露内部工具载荷或临时图片路径。

“停止等待”会中断浏览器请求、换用新的会话 ID，并隔离旧响应。文本流在工具调用之间检查断开状态，取消未提交的会话更新；已进入外部服务的调用依赖自身超时退出，不保证立刻终止。图片请求的工作线程仍可能执行完毕并写入旧会话，但不会进入新对话。

访问令牌只在页面内存中保留，刷新后重新输入；连接新身份默认开启新会话。浏览器 sessionStorage 只保存随机会话 ID，不保存令牌。当前服务密钥鉴权是调用方隔离，不等同于终端用户账号体系；公开多用户部署需补充登录、限流和 HTTPS。演示模式不应存放个人数据。

## 验证

```bash
npm ci --prefix frontend
npm run build --prefix frontend
python -m pip install -e '.[dev,ann,redis]'
python -m pytest
cd frontend
npx playwright install --with-deps chromium
npm test
```

浏览器测试启动真实 FastAPI 服务，覆盖连续对话、对比、指定库存、排除、刷新恢复、图片上传、手机宽度、错误提示以及旧响应隔离。仅错误和竞态场景使用网络拦截。GitHub Actions 同时运行 Python、浏览器和三副本 Docker 检查。

## 设计参考

- https://github.com/assistant-ui/assistant-ui ：聊天、附件与工具结果组件化。
- https://github.com/langchain-ai/agent-chat-ui ：对话和结果侧栏的布局。
- https://github.com/CopilotKit/CopilotKit ：页面与 Agent 共享购物条件的交互模式。

本实现独立编写，未引入这些框架或复制其源码；现有 Python Agent 不需要迁移到 LangGraph。基础组件使用 React，业务卡片及 API 适配层在 `frontend/src/`，依赖许可证随 npm 包分发。
