def rule_notice_item_number_tables(doc: dict) -> list[dict]:
    """Match notice tables that summarize penalty item numbers."""
    try:
        import re

        texts = doc.get("texts", [])
        stop_re = re.compile(r"^\s*Response to this Notice\s*$", re.IGNORECASE)
        cutoff = len(texts)
        for i, span in enumerate(texts):
            text = " ".join((span.get("text") or "").split())
            if stop_re.match(text):
                cutoff = i
                break

        out = []
        for span in texts[:cutoff]:
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            if re.search(r"\|\s*Item number\s*\|", text, re.IGNORECASE):
                out.append(span)
                continue
            table_data = span.get("table_data") or {}
            cells = table_data.get("cells") or []
            if any((cell.get("text") or "").strip().lower() == "item number" for cell in cells):
                out.append(span)
        return out
    except Exception:
        return []
