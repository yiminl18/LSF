def rule_tables_with_audited_financial_context(doc: dict) -> list[dict]:
    """Match tables whose path or nearby text mentions reports of independent auditors or financial statements."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            local = path
            for j in range(max(0, i - 3), min(len(texts), i + 3)):
                local += " " + (texts[j].get("text") or "").lower()
            if any(k in local for k in [
                "reports of independent", "independent registered public accounting",
                "financial statements", "supplementary data", "item 8"
            ]):
                out.append(span)
        return out
    except Exception:
        return []
