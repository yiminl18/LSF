def rule_page1_h1_excluding_known_boilerplate(doc: dict) -> list[dict]:
    """Match page-1 H1 spans excluding common boilerplate headers and captions."""
    try:
        boiler = {
            "form 10-k", "form 10-q", "form 8-k", "current report",
            "securities and exchange commission", "united states securities and exchange commission",
            "washington, d.c. 20549", "securities and exchange commission washington, d.c. 20549"
        }
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip().lower()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and txt not in boiler
            ):
                out.append(span)
        return out
    except Exception:
        return []
