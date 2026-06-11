def rule_cover_page_company_header_before_state_ein_sequence(doc: dict) -> list[dict]:
    """Match the cover-page company header before the common state/EIN/address sequence."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "FORM 10-" not in (span.get("text") or "").upper()
            ):
                window = texts[i + 1:i + 25]
                joined = " ".join((w.get("text") or "").lower() for w in window)
                if (
                    ("state or other jurisdiction" in joined or "jurisdiction of incorporation" in joined)
                    and ("i.r.s. employer identification no." in joined or "irs employer identification no." in joined)
                ):
                    out.append(span)
        return out
    except Exception:
        return []
