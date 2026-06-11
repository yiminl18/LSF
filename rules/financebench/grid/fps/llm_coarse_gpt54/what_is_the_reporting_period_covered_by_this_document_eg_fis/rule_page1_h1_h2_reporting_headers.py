def rule_page1_h1_h2_reporting_headers(doc: dict) -> list[dict]:
    """Match page-1 H1/H2 section headers that contain the reporting period or event date."""
    try:
        out = []
        for span in doc.get("texts", []):
            struct = span.get("structure") or {}
            lvl = struct.get("level")
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and lvl in ["H1", "H2"]:
                if any(k in text for k in ["fiscal year ended", "quarterly period ended", "date of report"]):
                    out.append(span)
        return out
    except Exception:
        return []
