def rule_page1_date_under_mail_path(doc: dict) -> list[dict]:
    """Match standalone page-1 dates whose section path is a mail or delivery header block."""
    try:
        import re

        date_re = re.compile(
            r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s*,?\s*\d{4}$",
            re.I,
        )
        path_re = re.compile(
            r"(?:via\s+electronic\s+mail|via\s+e-?mail|electronic\s+mail|overnight\s+express\s+delivery|return\s+receipt\s+requested)",
            re.I,
        )

        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") != "text":
                continue
            raw = span.get("text") or ""
            norm = re.sub(r"(?<=\d)\s+(?=\d)", "", " ".join(raw.split()))
            path = ((span.get("structure") or {}).get("path_text") or "")
            if date_re.match(norm) and path_re.search(path):
                out.append(span)
        return out
    except Exception:
        return []
