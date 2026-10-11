"""Public UI projection: catalog facts plus stock verified in this turn only."""
from .react import extract_candidates


def public_trace(trace):
    # Tool names and outcome only: never expose server paths or provider payloads.
    return [{"tool": x["tool"], "ok": not bool(x.get("error"))}
            for x in trace if x.get("tool")]


def cards_from_trace(trace, tools):
    ids, stocks, reasons = [], {}, {}
    for item in trace:
        tool, payload = item.get("tool"), item.get("result")
        if item.get("error") in {"constraint_violation", "invalid_arguments", "not_found"}:
            continue
        if tool == "check_inventory":
            pid = item.get("arguments", {}).get("product_id")
            if pid in tools.by_id:
                ids.append(pid)
                stocks[pid] = payload.get("stock") if isinstance(payload, dict) and not item.get("error") else None
        elif tool == "search_products" and not item.get("error"):
            for candidate in extract_candidates(payload, limit=20):
                ids.append(candidate["product_id"])
                reasons[candidate["product_id"]] = candidate.get("reasons", [])
        elif tool == "compare_products" and isinstance(payload, list) and not item.get("error"):
            for product in payload:
                pid = product.id if hasattr(product, "id") else product.get("id", product.get("product_id"))
                ids.append(pid)
                stocks[pid] = product.stock if hasattr(product, "stock") else product.get("stock")
    cards = []
    for pid in dict.fromkeys(ids):
        if pid not in tools.by_id:
            continue
        p, stock = tools.by_id[pid], stocks.get(pid)
        cards.append(dict(id=pid, title=p.title, category=p.category, price=p.price,
                          rating=p.rating, stock=stock, inventory_source=tools.inventory.source,
                          reasons=reasons.get(pid, [])))
    return cards
