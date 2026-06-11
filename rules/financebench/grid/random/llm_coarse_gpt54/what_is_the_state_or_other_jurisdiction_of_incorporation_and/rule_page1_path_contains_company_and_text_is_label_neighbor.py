def rule_page1_path_contains_company_and_text_is_label_neighbor(doc: dict) -> list[dict]:
    """Match page-1 spans whose path is the company name and whose nearby text is a label/value pair area."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if span.get("page_no") != 1 or not path or "FORM 10-" in path:
                continue
            window = " ".join((texts[j].get("text", "") or "") for j in range(max(0, i-2), min(len(texts), i+3)))
            if re.search(r"State or other jurisdiction|Employer Identification|I\.?R\.?S\.?", window, re.I):
                out.append(span)
        return out
    except Exception:
        return []
