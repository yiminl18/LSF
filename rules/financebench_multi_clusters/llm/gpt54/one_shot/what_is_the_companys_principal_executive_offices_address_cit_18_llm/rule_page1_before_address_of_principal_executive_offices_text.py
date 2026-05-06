def rule_page1_before_address_of_principal_executive_offices_text(doc: dict) -> list[dict]:
    """Match the immediately preceding span before a standalone '(Address of principal executive offices)' text span."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().lower()
            if txt == "(address of principal executive offices)":
                if i - 1 >= 0 and texts[i - 1].get("page_no") == 1:
                    out.append(texts[i - 1])
        return out
    except Exception:
        return []
