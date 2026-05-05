def rule_path_text_is_company_name(doc: dict) -> list[dict]:
    """Match spans whose own path_text is just the company name heading near the top of page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        banned = {
            "form 10-k",
            "united states securities and exchange commission",
            "documents incorporated by reference",
            "part i",
            "or",
        }
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").strip()
            low = path.lower()
            if span.get("page_no") != 1:
                continue
            if not path or low in banned:
                continue
            if "|" in path:
                continue
            if "exact name of registrant" in low:
                continue
            if any(ch.isalpha() for ch in path):
                out.append(span)
        return out
    except Exception:
        return []
