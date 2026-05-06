def rule_item1_incorporated_sentence(doc: dict) -> list[dict]:
    """Match Item 1 text spans containing incorporation history, which often restate the exact registrant name."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if "item 1" not in path and "business" not in path:
                continue
            if "incorporated" in low or "reincorporated" in low:
                out.append(span)
        return out
    except Exception:
        return []
