def rule_part_ii_common_stock_exchange(doc: dict) -> list[dict]:
    """Retrieve Part II Common Stock discussion that states the exchange where the stock trades."""
    try:
        texts = doc.get("texts", []) or []
        out = []
        for span in texts:
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            label = span.get("label")
            if label not in {"text", "section_header"}:
                continue
            p = path.lower()
            t = text.lower()
            if "part ii" in p and "common stock" in p and (
                "nasdaq global select market" in t or
                "new york stock exchange" in t or
                "traded on" in t or
                "listed on" in t
            ):
                out.append(span)
        return out
    except Exception:
        return []

