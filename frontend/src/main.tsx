import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { ApiError, call, stream } from "./api";
import { ProductPanel, Comparison } from "./ProductPanel";
import type {
  Card,
  Config,
  Constraints,
  Intent,
  Message,
  Reply,
} from "./types";
import "./style.css";
const empty: Constraints = {
  max_price: null,
  category: null,
  excluded_product_ids: [],
};
const names: Record<string, string> = {
  search_products: "检索商品",
  check_inventory: "确认库存",
  compare_products: "对比商品",
};
function uuid() {
  return crypto.randomUUID();
}
function App() {
  const [messages, setMessages] = useState<Message[]>([]),
    [cards, setCards] = useState<Card[]>([]),
    [selected, setSelected] = useState<string[]>([]);
  const [config, setConfig] = useState<Config | null>(null),
    [query, setQuery] = useState(""),
    [budget, setBudget] = useState(""),
    [category, setCategory] = useState("");
  const [constraints, setConstraints] = useState(empty),
    [file, setFile] = useState<File | null>(null),
    [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false),
    [progress, setProgress] = useState(""),
    [error, setError] = useState(""),
    [settings, setSettings] = useState(false),
    [guide, setGuide] = useState(false),
    [tokenInput, setTokenInput] = useState("");
  const token = useRef(""),
    session = useRef(sessionStorage.getItem("shopping-session") || uuid()),
    revision = useRef(0),
    flight = useRef<AbortController | null>(null),
    locked = useRef(false),
    dirty = useRef({ budget: false, category: false });
  const log = useRef<HTMLDivElement>(null),
    upload = useRef<HTMLInputElement>(null),
    settingsDialog = useRef<HTMLDialogElement>(null),
    guideDialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    sessionStorage.setItem("shopping-session", session.current);
    connect(true);
  }, []);
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    log.current?.scrollTo({
      top: log.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, progress]);
  useEffect(() => {
    if (settings) settingsDialog.current?.showModal();
    else settingsDialog.current?.close();
  }, [settings]);
  useEffect(() => {
    if (guide) guideDialog.current?.showModal();
    else guideDialog.current?.close();
  }, [guide]);
  function sync(c: Constraints) {
    setConstraints(c);
    setBudget(c.max_price === null ? "" : String(c.max_price));
    setCategory(c.category || "");
    dirty.current = { budget: false, category: false };
  }
  function reset() {
    revision.current++;
    flight.current?.abort();
    flight.current = null;
    locked.current = false;
    setBusy(false);
    session.current = uuid();
    sessionStorage.setItem("shopping-session", session.current);
    setMessages([]);
    setCards([]);
    setSelected([]);
    sync(empty);
    setQuery("");
    setFile(null);
    setError("");
    setProgress("");
  }
  async function restore() {
    const version = revision.current;
    const data = await call<{ messages: Message[]; constraints: Constraints }>(
      "/chat/history?session_id=" + encodeURIComponent(session.current),
      token.current,
    );
    if (version !== revision.current) return;
    setMessages(data.messages);
    sync(data.constraints);
    const last = [...data.messages].reverse().find((m) => m.reply);
    setCards(last?.reply?.cards || []);
    setSelected([]);
  }
  async function connect(recover = false) {
    const version = revision.current;
    try {
      const c = await call<Config>("/ui/config", token.current);
      if (version !== revision.current) return;
      setConfig(c);
      setError("");
      setSettings(false);
      if (recover) await restore();
    } catch (e) {
      if (version !== revision.current) return;
      setConfig(null);
      setError((e as Error).message);
      if (e instanceof ApiError && e.status === 401) setSettings(true);
    }
  }
  function attach(f?: File) {
    if (!f) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(f.type)) {
      setError("请选择 JPEG、PNG 或 WebP 图片。");
      return;
    }
    if (f.size > 5 * 1024 * 1024) {
      setError("图片不能超过 5 MB。");
      return;
    }
    setFile(f);
    setError("");
  }
  async function send(
    text = query,
    intent: Intent = "auto",
    ids: string[] = [],
  ) {
    if (locked.current) return;
    const input = text.trim();
    if (!input && !file) return;
    if (budget !== "" && (!Number.isFinite(+budget) || +budget < 0)) {
      setError("预算需为大于或等于 0 的数字。");
      return;
    }
    const version = revision.current;
    const ctl = new AbortController();
    flight.current = ctl;
    locked.current = true;
    setBusy(true);
    setError("");
    setProgress("请求已发送，等待处理…");
    const image = file;
    setMessages((m) => [
      ...m,
      {
        role: "user",
        text: input + (image ? `\n[参考图片：${image.name}]` : ""),
      },
    ]);
    const filters: Record<string, unknown> = {};
    if (dirty.current.budget)
      filters.max_price = budget === "" ? null : +budget;
    if (dirty.current.category) filters.category = category || null;
    function result(reply: Reply) {
      if (version !== revision.current) return;
      setMessages((m) => [
        ...m,
        { role: "assistant", text: reply.answer, reply },
      ]);
      setCards(reply.cards);
      setSelected([]);
      sync(reply.constraints);
      setQuery("");
      setFile(null);
      if (upload.current) upload.current.value = "";
    }
    try {
      if (image) {
        const body = new FormData();
        body.set("session_id", session.current);
        body.set("query", input);
        body.set("image", image);
        body.set("filters", JSON.stringify(filters));
        setProgress("正在处理参考图片并查询商品…");
        result(
          await call<Reply>("/chat/image", token.current, {
            method: "POST",
            body,
            signal: ctl.signal,
          }),
        );
      } else
        await stream(
          "/chat/stream",
          token.current,
          {
            session_id: session.current,
            query: input,
            intent,
            product_ids: ids,
            ...filters,
          },
          ctl.signal,
          (event, data) => {
            if (version !== revision.current) return;
            if (event === "result") result(data);
            else if (event === "progress")
              setProgress(
                `${names[data.tool] || "处理需求"} · ${data.phase === "start" ? "进行中" : data.ok ? "完成" : "暂未完成"}`,
              );
          },
        );
    } catch (e) {
      if (version !== revision.current) return;
      setError((e as Error).message);
      if (e instanceof ApiError && e.status === 401) setSettings(true);
    } finally {
      if (version === revision.current) {
        locked.current = false;
        setBusy(false);
        flight.current = null;
        setProgress("");
      }
    }
  }
  function stop() {
    reset();
    setError(
      "已停止等待并开始新对话。正在执行的外部调用可能仍在结束中，旧结果不会进入新对话。",
    );
  }
  return (
    <>
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="选物首页">
          <span className="brandmark">选</span>选物
          <span className="brand-en">SHOPPING AGENT</span>
        </a>
        <button id="new-chat" className="primary" onClick={reset}>
          ＋ 开始新的挑选
        </button>
        <div className="side-label">你的购物搭子</div>
        <p className="side-copy">
          把需求说清楚，
          <br />
          把选择交给有据可查的建议。
        </p>
        <div className="side-note">
          <span className="dot" />
          当前购物条件
          <p>
            预算：
            {constraints.max_price === null
              ? "不限"
              : `¥${constraints.max_price}`}
            <br />
            类目：{constraints.category || "不限"}
            <br />
            已排除：{constraints.excluded_product_ids.join("、") || "无"}
          </p>
        </div>
        <div className="side-bottom">
          <button onClick={() => setSettings(true)}>⚙ 连接设置</button>
          <button
            disabled={busy}
            onClick={() => restore().catch((e) => setError(e.message))}
          >
            ↻ 恢复会话
          </button>
          <button onClick={() => setGuide(true)}>使用说明 ↗</button>
          <small>
            价格与库存来自已接入的目录。
            <br />
            本页面不提供下单或支付。
          </small>
        </div>
      </aside>
      <div className="workspace">
        <header>
          <div>
            <span className="eyebrow">少一点纠结，多一点合适</span>
            <h1>一起挑件好东西</h1>
          </div>
          <span className={`status ${config ? "ready" : ""}`} role="status">
            {config
              ? `${config.inventory_source === "catalog_demo" ? "演示目录" : "已接入库存"} · ${config.product_count} 件商品`
              : "连接待确认"}
          </span>
        </header>
        <main>
          <section className="conversation" aria-label="购物对话">
            <div id="messages" ref={log} role="log" aria-live="polite">
              {!messages.length ? (
                <div id="welcome">
                  <span className="welcome-icon">✳</span>
                  <h2>这次，想找点什么？</h2>
                  <p>
                    告诉我用途、预算或喜欢的风格。
                    <br />
                    也可以上传参考图片，一起缩小选择范围。
                  </p>
                  <div className="suggestions">
                    {[
                      [
                        "01 / 日常出行",
                        "500 元以内的运动鞋",
                        "推荐500元以内的运动鞋",
                      ],
                      [
                        "02 / 通勤好物",
                        "找一款实用的背包",
                        "推荐适合通勤的背包",
                      ],
                    ].map(([label, title, q]) => (
                      <button key={q} disabled={busy} onClick={() => send(q)}>
                        <span>{label}</span>
                        <b>{title}</b>
                        <em>从日常需求出发 ↗</em>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                messages.map((m, i) => (
                  <article className={`message ${m.role}`} key={i}>
                    <div className="who">
                      {m.role === "user" ? "你" : "选物 · 购物助手"}
                    </div>
                    <div className="body">{m.text}</div>
                    {m.reply && (
                      <>
                        {m.reply.intent === "compare" &&
                          m.reply.cards.length >= 2 && (
                            <Comparison cards={m.reply.cards} />
                          )}
                        <details>
                          <summary>
                            查看处理过程 · {m.reply.tool_calls} 次查询
                          </summary>
                          {m.reply.tool_trace.map((t, j) => (
                            <div key={j}>
                              {names[t.tool] || t.tool} ·{" "}
                              {t.ok ? "完成" : "未完成"}
                            </div>
                          ))}
                          {m.reply.context.fallback_used && (
                            <div>模型暂不可用，本轮使用规则决策。</div>
                          )}
                        </details>
                      </>
                    )}
                  </article>
                ))
              )}
              {busy && (
                <div className="pending" role="status">
                  {progress}
                </div>
              )}
            </div>
            <form
              id="composer"
              onSubmit={(e) => {
                e.preventDefault();
                send();
              }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (!busy) attach(e.dataTransfer.files[0]);
              }}
            >
              {file && (
                <div id="attachment">
                  <img id="preview" src={preview} alt="待上传的参考图片" />
                  <span id="filename">{file.name}</span>
                  <button
                    type="button"
                    disabled={busy}
                    aria-label="移除图片"
                    onClick={() => setFile(null)}
                  >
                    ×
                  </button>
                </div>
              )}
              <label className="sr-only" htmlFor="query">
                描述你的购物需求
              </label>
              <textarea
                id="query"
                maxLength={4000}
                rows={2}
                value={query}
                disabled={busy}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="例如：预算 500 元，想找一双白色运动鞋…"
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" &&
                    !e.shiftKey &&
                    !e.nativeEvent.isComposing
                  ) {
                    e.preventDefault();
                    send();
                  }
                }}
              />
              <div className="composer-tools">
                <label className="upload">
                  ＋ 图片
                  <input
                    ref={upload}
                    id="image"
                    type="file"
                    disabled={busy}
                    accept="image/jpeg,image/png,image/webp"
                    onChange={(e) => attach(e.target.files?.[0])}
                  />
                </label>
                <span className="input-tip">
                  Enter 发送 · Shift + Enter 换行
                </span>
                {busy ? (
                  <button type="button" onClick={stop}>
                    停止等待
                  </button>
                ) : (
                  <button id="send" className="primary" type="submit">
                    发送 ↑
                  </button>
                )}
              </div>
              <div className="filters">
                <label>
                  预算上限 ¥{" "}
                  <input
                    id="budget"
                    type="number"
                    min="0"
                    step="0.01"
                    disabled={busy}
                    value={budget}
                    placeholder="不限"
                    onChange={(e) => {
                      setBudget(e.target.value);
                      dirty.current.budget = true;
                    }}
                  />
                </label>
                <label>
                  类目{" "}
                  <select
                    id="category"
                    disabled={busy}
                    value={category}
                    onChange={(e) => {
                      setCategory(e.target.value);
                      dirty.current.category = true;
                    }}
                  >
                    <option value="">不限</option>
                    {config?.categories.map((c) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </label>
              </div>
            </form>
            <p className="footnote">
              {config?.reasoner_backend === "llm" ? "模型决策" : "规则决策"} ·{" "}
              {config?.image_search_available
                ? "支持图片检索"
                : "当前目录没有商品图片，上传图片时请补充文字描述。"}
            </p>
            {error && (
              <div id="error" role="alert">
                {error}
                <button
                  disabled={busy}
                  onClick={() =>
                    restore()
                      .then(() => setError(""))
                      .catch((e) => setError(e.message))
                  }
                >
                  恢复会话
                </button>
              </div>
            )}
          </section>
          <ProductPanel
            cards={cards}
            selected={selected}
            setSelected={setSelected}
            busy={busy}
            send={send}
          />
        </main>
      </div>
      <dialog ref={settingsDialog} onCancel={() => setSettings(false)}>
        <button
          className="close"
          onClick={() => setSettings(false)}
          aria-label="关闭连接设置"
        >
          ×
        </button>
        <span className="eyebrow">连接设置</span>
        <h2>访问你的购物助手</h2>
        <p>
          服务启用鉴权时，请填写管理员提供的访问令牌。令牌只留在当前页面内存，刷新后需重新输入。
        </p>
        <label htmlFor="token">访问令牌</label>
        <input
          id="token"
          type="password"
          autoComplete="off"
          value={tokenInput}
          onChange={(e) => setTokenInput(e.target.value)}
          placeholder="演示模式可留空"
        />
        <button
          className="primary"
          onClick={() => {
            token.current = tokenInput.trim();
            setTokenInput("");
            reset();
            connect();
          }}
        >
          连接并开始新对话
        </button>
        {error && <p role="alert">{error}</p>}
      </dialog>
      <dialog ref={guideDialog} onCancel={() => setGuide(false)}>
        <button
          className="close"
          onClick={() => setGuide(false)}
          aria-label="关闭说明"
        >
          ×
        </button>
        <span className="eyebrow">三步开始</span>
        <h2>让购物需求更具体</h2>
        <ol>
          <li>描述需求，或选择预算和类目后发送。</li>
          <li>继续追问，也可以在卡片上查库存、排除商品。</li>
          <li>勾选两件商品，再点击“对比所选商品”。</li>
        </ol>
        <p>
          图片支持 JPEG、PNG、WebP，最大 5
          MB。默认目录缺少图片，仅按文字检索；真正的图文检索需配置索引。
        </p>
        <p>
          同一浏览器标签页可以恢复最近 20
          轮对话，服务端会话过期后重新开始。鉴权模式下刷新需重新连接，新连接默认创建新会话。演示模式不适合公开承载个人数据。
        </p>
      </dialog>
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
