def rule_principal_offices_press_release_dateline(doc: dict) -> list[dict]:
    """Return the press-release dateline when the filing is a first-page 8-K/news release style document."""
    try:
        import re

        full_text = " ".join((doc.get("text") or "").split())
        if not full_text:
            return []

        dateline_re = re.compile(
            r"\b[A-Z][A-Za-z .&'\-]+,\s*[A-Z][a-z]{1,3}\.?,\s*[A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4}\b"
        )

        m = dateline_re.search(full_text)
        if m:
            return [{"text": " ".join(m.group(0).split())}]

        return []
    except Exception:
        return []
