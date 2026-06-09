def rule_page12_cover_company_near_tax_or_state(doc: dict) -> list[dict]:
    """Match short page-1/2 company-name spans that appear just before tax, state, or file-number cover fields."""
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

        def looks_company(text: str) -> bool:
            text = (text or "").strip()
            if not text or len(text) > 120:
                return False
            if normalize(text) in blocked:
                return False
            return bool(company_re.search(text))

        hits: list[dict] = []
        texts = doc.get("texts", [])
        for idx, span in enumerate(texts):
            marker = normalize(span.get("text") or "")
            if (
                "i r s employer identification no" in marker
                or "commission file number" in marker
                or "state or other jurisdiction of incorporation" in marker
            ):
                for prev_idx in range(max(0, idx - 5), idx):
                    candidate = texts[prev_idx]
                    if (candidate.get("page_no") or 99) <= 2 and candidate.get("label") != "table" and looks_company(candidate.get("text") or ""):
                        hits.append(candidate)
        return hits
    except Exception:
        return []
