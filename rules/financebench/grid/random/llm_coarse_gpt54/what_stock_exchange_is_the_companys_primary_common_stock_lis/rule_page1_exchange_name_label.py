def rule_page1_exchange_name_label(doc: dict) -> list[dict]:
    """Match page-1 spans that explicitly mention the exchange-registration label."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") == 1:
                txt = (span.get("text") or "").lower()
                if "name of each exchange on which registered" in txt or "name of each exchange on which registered" in (span.get("structure", {}).get("path_text", "") or "").lower():
                    out.append(span)
        return out
    except Exception:
        return []
