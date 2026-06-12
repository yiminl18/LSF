def rule_page_near_fd6_or_fd8_or_fd9(doc: dict) -> list[dict]:
    """Match spans on pages where contents indicate FD/FO debt-subject-to-statutory-limit tables appear."""
    import re
    try:
        target_pages = set()
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "contents" not in path.lower():
                continue
            if re.search(r"\b(FD|FO)[-\s]?(6|8|9)\b", text, re.I) and re.search(r"statutory (limit|limitation)|debt subject to statutory|status and application of statutory", text, re.I):
                nums = re.findall(r"\b\d+\b", text)
                for n in nums:
                    try:
                        target_pages.add(int(n))
                    except Exception:
                        pass
        if not target_pages:
            return []
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") in target_pages:
                out.append(span)
        return out
    except Exception:
        return []
