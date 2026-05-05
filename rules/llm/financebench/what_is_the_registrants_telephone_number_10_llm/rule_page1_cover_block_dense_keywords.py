def rule_page1_cover_block_dense_keywords(doc: dict) -> list[dict]:
    """Match spans in a dense cover block with multiple cover-page metadata keywords."""
    try:
        out = []
        keywords = [
            "exact name of registrant",
            "state or other jurisdiction",
            "address of principal executive offices",
            "zip code",
            "employer identification",
            "securities registered pursuant to section 12(b)",
        ]
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            score = sum(1 for k in keywords if k in text)
            if score >= 3:
                out.append(span)
        return out
    except Exception:
        return []
