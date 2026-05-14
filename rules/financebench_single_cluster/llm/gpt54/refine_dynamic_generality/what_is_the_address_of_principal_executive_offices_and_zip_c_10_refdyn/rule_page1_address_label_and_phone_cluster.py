def rule_page1_address_label_and_phone_cluster(doc: dict) -> list[dict]:
    """Match clusters containing both principal-office label and registrant phone on page 1."""
    try:
        import re
        spans = doc.get("texts", [])
        out_idx = set()
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "principal executive offices" in txt:
                for j in range(i, min(len(spans), i + 6)):
                    t2 = ((spans[j].get("text") or "") + " " + (spans[j].get("text_span") or "")).lower()
                    if spans[j].get("page_no") == 1:
                        out_idx.add(j)
                    if re.search(r"telephone number|registrant.?s telephone number|\(\d{3}\)\s*\d{3}[- ]?\d{4}", t2):
                        out_idx.add(j)
        return [spans[i] for i in sorted(out_idx)]
    except Exception:
        return []
