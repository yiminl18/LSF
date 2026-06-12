def rule_esf_before_special_reports(doc: dict) -> list[dict]:
    """Match tables immediately before Special Reports when ESF is the last international-statistics subsection."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") == "section_header" and re.search(r'SPECIAL REPORTS', span.get("text", ""), re.I):
                for j in range(max(0, i - 10), i):
                    if texts[j].get("label") == "table":
                        out.append(texts[j])
                break
        return out
    except Exception:
        return []
