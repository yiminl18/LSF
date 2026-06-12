def rule_march_june_september_december_header(doc: dict) -> list[dict]:
    """Match modern quarterly issue month headers using March/June/September/December."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(r"^(MARCH|JUNE|SEPTEMBER|DECEMBER)\s+\d{4}$", re.I)
        return [s for s in texts if pat.match((s.get("text") or "").strip())]
    except Exception:
        return []
