def rule_page1_h1_with_uppercase_name(doc: dict) -> list[dict]:
    """Match page-1 H1 spans with mostly uppercase alphabetic content, typical of cover-page registrant names."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            letters = re.sub(r"[^A-Za-z]", "", txt)
            if not letters:
                continue
            upper_ratio = sum(1 for c in letters if c.isupper()) / max(len(letters), 1)
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and upper_ratio > 0.7
                and "form 10-" not in txt.lower()
                and "securities and exchange commission" not in txt.lower()
            ):
                out.append(span)
        return out
    except Exception:
        return []
