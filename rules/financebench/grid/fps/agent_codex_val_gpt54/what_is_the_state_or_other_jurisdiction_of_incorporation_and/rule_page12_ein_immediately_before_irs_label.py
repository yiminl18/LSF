def rule_page12_ein_immediately_before_irs_label(doc: dict) -> list[dict]:
    """Match page-1/2 EIN values that appear immediately before an employer-identification label span."""
    try:
        import re

        def _norm(text: str) -> str:
            text = (text or "").replace("I.R.S.", "IRS").replace("i.r.s.", "irs")
            return re.sub(r"\s+", " ", text).strip()

        def _looks_ein(text: str) -> bool:
            return bool(re.search(r"\b\d{2}-\d{7}\b", _norm(text)))

        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            lowered = _norm(span.get("text") or "").lower()
            if "employer identification" not in lowered:
                continue

            if i - 1 < 0:
                continue
            candidate = texts[i - 1]
            candidate_text = _norm(candidate.get("text") or "")
            if candidate.get("page_no") != span.get("page_no"):
                continue
            if not _looks_ein(candidate_text):
                continue
            if "commission file" in candidate_text.lower():
                continue

            results.append(candidate)

        return results
    except Exception:
        return []
