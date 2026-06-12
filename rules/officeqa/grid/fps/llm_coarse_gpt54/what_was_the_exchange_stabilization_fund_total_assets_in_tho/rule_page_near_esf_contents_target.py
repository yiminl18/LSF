def rule_page_near_esf_contents_target(doc: dict) -> list[dict]:
    """Use contents-page ESF-1 page references to return table spans on the referenced page and nearby pages."""
    import re
    try:
        target_pages = set()
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            if re.search(r'\bESF-?1\b', text, re.I):
                for m in re.finditer(r'\bESF-?1\b.*?(\d{1,3})\b', text, re.I | re.S):
                    try:
                        target_pages.add(int(m.group(1)))
                    except Exception:
                        pass
        out = []
        for span in doc.get("texts", []):
            p = span.get("page_no")
            if isinstance(p, int) and any(abs(p - tp) <= 1 for tp in target_pages):
                if span.get("label") == "table":
                    out.append(span)
        return out
    except Exception:
        return []
