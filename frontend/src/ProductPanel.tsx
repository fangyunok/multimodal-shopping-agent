import type { Card, Intent } from "./types";
export function ProductPanel({
  cards,
  selected,
  setSelected,
  busy,
  send,
}: {
  cards: Card[];
  selected: string[];
  setSelected: (ids: string[]) => void;
  busy: boolean;
  send: (query: string, intent?: Intent, ids?: string[]) => void;
}) {
  return (
    <aside className="products">
      <div className="panel-title">
        <div>
          <span className="eyebrow">为你整理</span>
          <h2>
            本轮候选 <span id="count">{cards.length}</span>
          </h2>
        </div>
        <span>↗</span>
      </div>
      <div id="cards">
        {!cards.length ? (
          <div className="empty">
            <span>◇</span>
            <h3>好选择，从需求开始</h3>
            <p>
              发送需求后，价格、评分和
              <br />
              本轮确认的库存会显示在这里。
            </p>
          </div>
        ) : (
          cards.map((p) => (
            <article className="card" data-product-id={p.id} key={p.id}>
              <div className="card-top">
                <div className="category-art" aria-hidden="true">
                  {p.category.includes("鞋")
                    ? "⌁"
                    : p.category.includes("包")
                      ? "▢"
                      : "◇"}
                </div>
                <div>
                  <h3>{p.title}</h3>
                  <small>
                    {p.category} · {p.id}
                  </small>
                </div>
              </div>
              <div className="price-line">
                <span className="price">
                  ¥{p.price.toLocaleString("zh-CN")}
                </span>
                <small>评分 {p.rating.toFixed(1)}</small>
              </div>
              <div className={`stock ${p.stock === null ? "unknown" : ""}`}>
                {p.stock === null
                  ? "库存暂未确认"
                  : p.stock > 0
                    ? `有货 · 本轮确认 ${p.stock} 件`
                    : "暂时无货"}
              </div>
              <small>
                {p.inventory_source === "catalog_demo"
                  ? "来源：演示目录"
                  : "来源：已接入库存服务"}
              </small>
              <div className="card-actions">
                <button
                  disabled={busy}
                  onClick={() =>
                    send(`查询 ${p.id} 的库存`, "inventory", [p.id])
                  }
                >
                  查库存
                </button>
                <button
                  disabled={busy}
                  onClick={() => send(`排除 ${p.id}，推荐其他商品`, "search")}
                >
                  排除
                </button>
                <label className="select-product">
                  <input
                    type="checkbox"
                    aria-label={`选择 ${p.title} 进行对比`}
                    checked={selected.includes(p.id)}
                    disabled={
                      busy ||
                      (!selected.includes(p.id) && selected.length === 2)
                    }
                    onChange={(e) =>
                      setSelected(
                        e.target.checked
                          ? [...selected, p.id]
                          : selected.filter((id) => id !== p.id),
                      )
                    }
                  />
                  对比
                </label>
              </div>
            </article>
          ))
        )}
      </div>
      <button
        id="compare"
        disabled={busy || selected.length !== 2}
        onClick={() =>
          send(`对比 ${selected.join(" 和 ")}`, "compare", selected)
        }
      >
        对比所选商品（{selected.length}/2）
      </button>
      <p className="footnote">
        图形为类目示意，非商品实拍。库存为查询时结果，可能随时间变化。
      </p>
    </aside>
  );
}
export function Comparison({ cards }: { cards: Card[] }) {
  return (
    <div className="comparison">
      <table>
        <caption>商品对比 · 本轮工具结果</caption>
        <thead>
          <tr>
            <th>商品</th>
            {cards.map((p) => (
              <th key={p.id}>{p.title}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {["价格", "评分", "库存"].map((name, i) => (
            <tr key={name}>
              <th>{name}</th>
              {cards.map((p) => (
                <td key={p.id}>
                  {i === 0
                    ? `¥${p.price}`
                    : i === 1
                      ? p.rating
                      : p.stock === null
                        ? "未知"
                        : `${p.stock} 件`}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
