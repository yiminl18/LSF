def rule_pages_after_contents_canadian_listing(doc: dict) -> list[dict]:
    """Match spans on the page number listed for Canadian dollar positions in contents, when recoverable from text."""
    try:
        import re
        texts = doc.get("texts", [])
        target_pages = set()
        for span in texts:
            txt = (span.get("text") or "")
            low = txt.lower()
            if "canadian dollar positions" in low:
                nums = re.findall(r"\b(\d{2,3})\b", txt)
                for n in nums:
                    try:
                        val = int(n)
                        if 50 <= val <= 200:
                            target_pages.add(val)
                    except Exception:
                        pass
        return [s for s in texts if s.get("page_no") in target_pages]
    except Exception:
        return []
