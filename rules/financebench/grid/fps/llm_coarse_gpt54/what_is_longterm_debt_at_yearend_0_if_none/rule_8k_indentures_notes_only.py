def rule_8k_indentures_notes_only(doc: dict) -> list[dict]:
    """Match 8-K documents focused on indentures, notes, exhibits, or board changes rather than financial statements, useful for 0 answers."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = (span.get("structure", {}) or {}).get("path_text", "").lower()
            if any(k in path for k in ["item 8.01", "item 9.01", "item 5.02", "item 2.01", "item 2.03"]):
                if re.search(r"indenture|supplemental indenture|other events|departure of directors|completion of acquisition|exhibits", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
