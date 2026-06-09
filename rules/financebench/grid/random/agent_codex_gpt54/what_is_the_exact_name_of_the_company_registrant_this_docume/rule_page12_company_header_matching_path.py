def rule_page12_company_header_matching_path(doc: dict) -> list[dict]:
    """Match short page-1/2 spans whose text matches the breadcrumb terminal and that terminal looks like a company name."""
    try:
        import re

        def normalize(text: str) -> str:
            text = (text or "").lower().replace("&", " and ")
            text = re.sub(r"[^a-z0-9]+", " ", text)
            return " ".join(text.split())

        blocked = {
            "united states securities and exchange commission",
            "securities and exchange commission",
            "form 10 k",
            "form 10 q",
            "form 8 k",
            "current report",
            "or",
            "washington d c 20549",
            "table of contents",
            "signatures",
        }
        company_re = re.compile(
            r"\b(?:inc\.?|corporation|corp\.?|company|plc|limited|ltd\.?|holdings?)\b|^\d+[a-z]?\s+company\b",
            re.IGNORECASE,
        )

        hits: list[dict] = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            path_text = ((span.get("structure") or {}).get("path_text") or "").split("|")[-1].strip()
            if (
                (span.get("page_no") or 99) <= 2
                and span.get("label") in {"section_header", "text"}
                and 0 < len(text) <= 120
                and normalize(text) == normalize(path_text)
                and normalize(text) not in blocked
                and company_re.search(path_text or text)
            ):
                hits.append(span)
        return hits
    except Exception:
        return []
