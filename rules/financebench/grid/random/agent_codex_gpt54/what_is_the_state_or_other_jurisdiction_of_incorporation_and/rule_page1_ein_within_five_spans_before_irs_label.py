def rule_page1_ein_within_five_spans_before_irs_label(doc: dict) -> list[dict]:
    """Match the nearest page-1 EIN value within five spans before the IRS employer-identification cover label."""
    try:
        import re

        def normalize(text: str) -> str:
            text = (text or "").strip().lower()
            text = text.replace("i.r.s.", "irs")
            return re.sub(r"\s+", " ", text)

        def looks_ein_value(text: str) -> bool:
            return bool(re.search(r"\b\d{2}-\d{7}\b", (text or "").strip()))

        hits: list[dict] = []
        texts = doc.get("texts", [])
        for idx, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if "employer identification no" not in normalize(span.get("text") or ""):
                continue
            for prev_idx in range(idx - 1, max(-1, idx - 6), -1):
                candidate = texts[prev_idx]
                candidate_low = normalize(candidate.get("text") or "")
                if candidate.get("page_no") != 1:
                    continue
                if (
                    looks_ein_value(candidate.get("text") or "")
                    and "commission file" not in candidate_low
                    and "employer identification" not in candidate_low
                ):
                    hits.append(candidate)
                    break
        return hits
    except Exception:
        return []
