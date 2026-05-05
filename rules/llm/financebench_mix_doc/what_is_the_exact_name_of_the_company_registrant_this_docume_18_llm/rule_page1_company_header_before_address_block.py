def rule_page1_company_header_before_address_block(doc: dict) -> list[dict]:
    """Match page-1 H1 spans followed by address/telephone block content within the next few spans."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            found = False
            for j in range(i + 1, min(i + 15, len(texts))):
                nxt = texts[j]
                if nxt.get("page_no") != 1:
                    break
                t = nxt.get("text") or ""
                if "Address of principal executive offices" in t or "Registrant’s telephone number" in t or "Registrant's telephone number" in t:
                    found = True
                    break
            if found:
                out.append(span)
        return out
    except Exception:
        return []
