def rule_page1_h1_before_address_block(doc: dict) -> list[dict]:
    """Match page-1 H1 headers whose nearby following spans include address/telephone labels."""
    try:
        texts = doc.get("texts", [])
        out = []
        keys = [
            "address of principal executive offices",
            "address and telephone number",
            "registrant’s telephone number",
            "registrant's telephone number",
        ]
        for i, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                window = texts[i + 1:i + 20]
                joined = " ".join((w.get("text") or "").lower() for w in window)
                if any(k in joined for k in keys):
                    if "FORM 10-" not in (span.get("text") or "").upper():
                        out.append(span)
        return out
    except Exception:
        return []
