def rule_page1_exact_name_section(doc: dict) -> list[dict]:
    """Match spans under or near the registrant exact-name section on page 1."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and (
                re.search(r"exact name of registrant|exact name of registrant as specified in (its )?charter", path, re.I)
                or re.search(r"exact name of registrant|exact name of registrant as specified in (its )?charter", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
