def rule_modern_first_page_title_month(doc: dict) -> list[dict]:
    """Match first-page title/header month spans in 1992+ and 2008+ bulletins."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"^(MARCH|JUNE|SEPTEMBER|DECEMBER)\s+\d{4}$|TREASURY\s+BULLETIN\s+(MARCH|JUNE|SEPTEMBER|DECEMBER)\s+\d{4}",
            re.I,
        )
        return [s for s in texts if s.get("page_no") == 1 and pat.search((s.get("text") or "").strip())]
    except Exception:
        return []
