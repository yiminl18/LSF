def rule_10q_split_state_and_ein_h2s(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 split cover spans in 10-Qs where state and EIN are separated across headers."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            lvl = span.get("structure", {}).get("level")
            text = (span.get("text", "") or "").strip()
            if span.get("page_no") == 1 and lvl in {"H2", "H3"} and (
                re.fullmatch(r"\d{2}-\d{7}", text)
                or re.fullmatch(r"(Delaware|Washington|New York|Jersey)", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
