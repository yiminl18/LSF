def rule_page12_inline_ein_cover_span(doc: dict) -> list[dict]:
    """Match short page-1/2 cover spans that inline an EIN together with the employer-identification label."""
    try:
        import re

        def _norm(text: str) -> str:
            text = (text or "").replace("I.R.S.", "IRS").replace("i.r.s.", "irs")
            return re.sub(r"\s+", " ", text).strip()

        def _looks_ein(text: str) -> bool:
            return bool(re.search(r"\b\d{2}-\d{7}\b", _norm(text)))

        results = []
        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 2:
                continue

            text = _norm(span.get("text") or "")
            lowered = text.lower()
            if not _looks_ein(text):
                continue
            if "employer identification" not in lowered:
                continue
            if any(
                marker in lowered
                for marker in ("trust deed", "securities act of 1933", "transfer agent")
            ):
                continue

            results.append(span)

        return results
    except Exception:
        return []
