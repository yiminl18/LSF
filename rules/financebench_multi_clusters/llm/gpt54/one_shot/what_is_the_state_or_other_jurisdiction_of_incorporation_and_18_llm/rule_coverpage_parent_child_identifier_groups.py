def rule_coverpage_parent_child_identifier_groups(doc: dict) -> list[dict]:
    """Match parent-child groups on page 1 where parent is a short identifier header and child is a label."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = (span.get("text", "") or "").strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header" and (
                re.fullmatch(r"\d{2}-\d{7}", text)
                or re.fullmatch(r"(Delaware|Washington|New York|Jersey(?:\s*\(Channel Islands\))?)", text, re.I)
            ):
                out.append(span)
                for child in texts:
                    if child.get("structure", {}).get("parent_id") == i:
                        out.append(child)
        return out
    except Exception:
        return []
