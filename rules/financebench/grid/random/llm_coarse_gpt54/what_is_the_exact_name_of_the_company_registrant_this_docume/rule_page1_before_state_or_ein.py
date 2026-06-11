def rule_page1_before_state_or_ein(doc: dict) -> list[dict]:
    """Match page-1 spans immediately before state/EIN boilerplate when they look like the company name."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if (
                span.get("page_no") == 1
                and (
                    "state or other jurisdiction" in txt
                    or "i.r.s. employer identification no." in txt
                    or "irs employer identification no." in txt
                )
                and i > 0
            ):
                prev = texts[i - 1]
                if prev.get("page_no") == 1:
                    out.append(prev)
        return out
    except Exception:
        return []
