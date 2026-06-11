def rule_page1_h1_not_boilerplate(doc: dict) -> list[dict]:
    """Match any page-1 H1 span that is not obvious boilerplate."""
    try:
        out = []
        boiler = {
            "form 10-k", "form 10-q", "form 8-k", "current report",
            "securities and exchange commission", "united states securities and exchange commission",
            "washington, d.c. 20549", "securities and exchange commission washington, d.c. 20549"
        }
        for span in doc.get("texts", []):
            txt = " ".join((span.get("text") or "").lower().split())
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and txt not in boiler
            ):
                out.append(span)
        return out
    except Exception:
        return []
