def rule_page1_first_non_boilerplate_h1(doc: dict) -> list[dict]:
    """Match the first page-1 H1 that is not SEC/form/current-report boilerplate."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("structure", {}).get("level") != "H1":
                continue
            txt = (span.get("text") or "").lower()
            if any(k in txt for k in [
                "securities and exchange commission",
                "form 10-k", "form 10-q", "form 8-k",
                "current report", "washington, d.c. 20549"
            ]):
                continue
            out.append(span)
            break
        return out
    except Exception:
        return []
