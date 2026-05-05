def rule_tables_with_total_assets_and_statement_order(doc: dict) -> list[dict]:
    """Match tables appearing after income statement and before cash flow statement, where balance sheet often sits."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if "assets" not in txt and "total assets" not in txt:
                continue
            prev = " ".join((x.get("text") or "").lower() for x in texts[max(0, i-20):i])
            nxt = " ".join((x.get("text") or "").lower() for x in texts[i:i+20])
            if ("income" in prev or "comprehensive income" in prev or "operations" in prev) and ("cash flows" in nxt or "statement of cash flows" in nxt):
                out.append(span)
        return out
    except Exception:
        return []
