def rule_page12_company_suffix_section_header(doc: dict) -> list[dict]:
    """Match short page-1/2 section headers that look like a legal company name."""
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

        return [
            span for span in doc.get("texts", [])
            if (span.get("page_no") or 99) <= 2
            and span.get("label") == "section_header"
            and 0 < len((span.get("text") or "").strip()) <= 120
            and normalize(span.get("text") or "") not in blocked
            and company_re.search((span.get("text") or "").strip())
        ]
    except Exception:
        return []
