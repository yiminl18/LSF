def rule_page1_exchange_after_registered_header_generic(doc: dict) -> list[dict]:
    """Return page-1 spans after generic exchange-registration headers in OCR-fragmented layouts."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            low = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and ("name of each exchange" in low or "exchange on which registered" in low):
                for j in range(i + 1, min(i + 6, len(texts))):
                    s = texts[j]
                    if s.get("page_no") != 1:
                        break
                    out.append(s)
        return out
    except Exception:
        return []
