def rule_page1_before_state_or_ein(doc: dict) -> list[dict]:
    """Match page-1 spans immediately preceding state/EIN/address metadata blocks in the cover page header."""
    try:
        texts = doc.get("texts", [])
        out = []
        trigger_phrases = [
            "state or other jurisdiction of incorporation",
            "state of incorporation",
            "i.r.s. employer identification no",
            "irs employer identification no",
            "address of principal executive offices",
        ]
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if any(p in txt for p in trigger_phrases):
                for j in range(max(0, i - 2), i):
                    cand = texts[j]
                    if cand.get("page_no") == 1 and any(ch.isalpha() for ch in (cand.get("text") or "")):
                        out.append(cand)
                parent_id = span.get("structure", {}).get("parent_id")
                if parent_id is not None and 0 <= parent_id < len(texts):
                    out.append(texts[parent_id])
        return out
    except Exception:
        return []
