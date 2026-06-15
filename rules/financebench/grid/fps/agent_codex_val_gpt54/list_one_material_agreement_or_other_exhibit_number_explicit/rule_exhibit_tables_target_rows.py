def rule_exhibit_tables_target_rows(doc: dict) -> list[dict]:
    """Match exhibit-section tables containing qualifying 10.x, 4.x, 99.x, or 104 rows."""
    try:
        import re

        texts = doc.get("texts", [])
        head = " ".join((s.get("text") or "") for s in texts[:40]).upper()
        form_8k = "FORM 8-K" in head

        exhibit_path_re = re.compile(
            r"(item\s*15|item\s*6\.?\s*exhibits|item\s*9\.01|\b3\. exhibits\b|\b2\. exhibits\b|\(3\) exhibits|index to exhibits|exhibits and financial statement schedules|financial statements and exhibits)",
            re.I,
        )
        exhibit_header_re = re.compile(
            r"(exhibit no\.?|exhibit number|description of exhibit|\(d\) exhibits|item 6\. exhibits|item 9\.01|exhibits index)",
            re.I,
        )
        row_10_re = re.compile(r"(^|\|)\s*\**\s*10\s*\.\s*\d+[A-Za-z0-9\.]*(?=\s*\||\s|$)", re.I)
        row_4_re = re.compile(r"(^|\|)\s*\**\s*4\s*\.\s*\d+[A-Za-z0-9\.]*(?=\s*\||\s|$)", re.I)
        row_99_re = re.compile(r"(^|\|)\s*\**\s*99\s*\.\s*\d+[A-Za-z0-9\.]*(?=\s*\||\s|$)", re.I)
        row_104_re = re.compile(r"(^|\|)\s*\**\s*104(?=\s*\||\s|$)", re.I)

        matches: list[dict] = []
        for span in texts:
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            in_exhibit_context = bool(exhibit_path_re.search(path) or exhibit_header_re.search(text))
            if not in_exhibit_context:
                continue
            if row_10_re.search(text) or row_4_re.search(text) or row_99_re.search(text) or (form_8k and row_104_re.search(text)):
                matches.append(span)
        return matches
    except Exception:
        return []
