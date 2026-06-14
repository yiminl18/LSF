def rule_page123_first_non191_part_after_reporting_item(doc: dict) -> list[dict]:
    """When an early Part 191 reporting item appears first, match the first later non-191 cited section on pages 1-3."""
    try:
        import re

        texts = doc.get("texts", [])
        cite_re = re.compile(r"^\s*(?:\d+\s*[\.\)]\s*)?(?:[§s]\s*)?(19\d)\.\d+\b", re.IGNORECASE)

        first_part = None
        for span in texts:
            if span.get("page_no", 99) > 2:
                continue
            match = cite_re.search((span.get("text") or "").strip())
            if match:
                first_part = match.group(1)
                break
        if first_part != "191":
            return []

        for span in texts:
            if span.get("page_no", 99) > 3:
                continue
            match = cite_re.search((span.get("text") or "").strip())
            if match and match.group(1) != "191":
                return [span]
        return []
    except Exception:
        return []
