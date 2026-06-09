def rule_page1_inline_or_combined_ein_cover_value(doc: dict) -> list[dict]:
    """Match page-1 EIN spans that either inline the IRS label or combine the state and EIN before dual cover labels."""
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
            text = span.get("text") or ""
            low = normalize(text)
            if not looks_ein_value(text) or "commission file" in low:
                continue
            inline_label = "employer identification" in low
            following = " ".join(
                normalize(texts[j].get("text") or "")
                for j in range(idx + 1, min(len(texts), idx + 3))
                if texts[j].get("page_no") == 1
            )
            combined_cover = (
                "state or other jurisdiction of incorporation" in following
                and "employer identification" in following
            )
            if inline_label or combined_cover:
                hits.append(span)
        return hits
    except Exception:
        return []
