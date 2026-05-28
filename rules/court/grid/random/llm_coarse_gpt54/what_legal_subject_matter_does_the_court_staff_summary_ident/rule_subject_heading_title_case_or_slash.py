def rule_subject_heading_title_case_or_slash(doc: dict) -> list[dict]:
    """Match likely subject headings that are short title-like phrases, often including slashes or legal act names."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if not txt or len(txt) > 80:
                continue
            up = txt.upper()
            if any(bad in up for bad in ["SUMMARY", "COUNSEL", "OPINION", "BACKGROUND"]):
                continue
            if span.get("page_no") not in {1, 2, 3}:
                continue
            if span.get("bold") != 1:
                continue
            if re.search(r"/", txt) or re.search(r"\b(Act|Law|Amendment|Immigration|Bankruptcy|Corpus|Nobis|Mootness|Discovery)\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
