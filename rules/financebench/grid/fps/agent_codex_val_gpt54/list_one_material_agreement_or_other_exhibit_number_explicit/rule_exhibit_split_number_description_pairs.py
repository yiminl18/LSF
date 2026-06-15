def rule_exhibit_split_number_description_pairs(doc: dict) -> list[dict]:
    """Match exhibit-number spans whose descriptions are split into the next sibling span."""
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
        line_10_re = re.compile(r"^\s*(?:exhibit\s+)?\**\s*10\s*\.\s*\d+[A-Za-z0-9\.]*(?:\.)?(?=\s|$)", re.I)
        line_4_re = re.compile(r"^\s*(?:exhibit\s+)?\**\s*4\s*\.\s*\d+[A-Za-z0-9\.]*(?:\.)?(?=\s|$)", re.I)
        line_99_re = re.compile(r"^\s*(?:exhibit\s+)?\**\s*99\s*\.\s*\d+[A-Za-z0-9\.]*(?:\.)?(?=\s|$)", re.I)
        line_104_re = re.compile(r"^\s*(?:exhibit\s+)?\**\s*104(?:\.)?(?=\s|$)", re.I)

        matches: list[dict] = []
        for i, span in enumerate(texts[:-1]):
            text = " ".join((span.get("text") or "").split())
            path = ((span.get("structure") or {}).get("path_text") or "")
            in_exhibit_context = bool(exhibit_path_re.search(path) or exhibit_header_re.search(text))
            if not in_exhibit_context or len(text) > 16:
                continue

            is_target = bool(line_10_re.search(text) or line_4_re.search(text) or line_99_re.search(text) or (form_8k and line_104_re.search(text)))
            if not is_target:
                continue

            next_span = texts[i + 1]
            next_text = " ".join((next_span.get("text") or "").split())
            next_path = ((next_span.get("structure") or {}).get("path_text") or "")
            next_in_exhibit_context = bool(exhibit_path_re.search(next_path) or exhibit_header_re.search(next_text))
            if next_span.get("page_no") != span.get("page_no") or not next_in_exhibit_context:
                continue
            if len(next_text) < 10 or exhibit_header_re.search(next_text):
                continue
            matches.extend([span, next_span])
        return matches
    except Exception:
        return []
