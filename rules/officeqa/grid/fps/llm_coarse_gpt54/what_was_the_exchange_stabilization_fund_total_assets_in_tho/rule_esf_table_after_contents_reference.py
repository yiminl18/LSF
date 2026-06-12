def rule_esf_table_after_contents_reference(doc: dict) -> list[dict]:
    """Use contents references to ESF-1 and return later table spans with matching page numbers."""
    import re
    try:
        refs = set()
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            if re.search(r'\bESF-?1\b', txt, re.I):
                nums = re.findall(r'\b(\d{1,3})\b', txt)
                for n in nums:
                    try:
                        refs.add(int(n))
                    except Exception:
                        pass
        out = []
        for span in doc.get("texts", []):
            p = span.get("page_no")
            if span.get("label") == "table" and isinstance(p, int) and p in refs:
                out.append(span)
        return out
    except Exception:
        return []
