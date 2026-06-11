def rule_exchange_in_cover_table_markdown(doc: dict) -> list[dict]:
    """Match markdown table spans on page 1 containing exchange registration info."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") == "table":
                txt = (span.get("text") or "")
                if re.search(r'title of each class', txt, re.I) and re.search(r'(name of each exchange on which registered|new york stock exchange|nasdaq)', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
