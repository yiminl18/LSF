def rule_page12_ein_before_commission_with_following_irs(doc: dict) -> list[dict]:
    """Match page-1/2 EIN values that sit before a commission-file label when the IRS label follows nearby."""
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
            if "commission file" not in lowered:
                continue

            following = " ".join(
                _norm(texts[j].get("text") or "").lower()
                for j in range(i + 1, min(len(texts), i + 5))
                if texts[j].get("page_no") == span.get("page_no")
            )
            if "employer identification no" not in following:
                continue

            if i - 1 < 0:
                continue
            candidate = texts[i - 1]
            if candidate.get("page_no") != span.get("page_no"):
                continue
            if _looks_ein(candidate.get("text") or ""):
                results.append(candidate)

        return results
    except Exception:
        return []
