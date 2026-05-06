def rule_page1_address_near_exact_name_of_registrant(doc: dict) -> list[dict]:
    """Return spans near the exact-name-of-registrant label, where cover-page address metadata usually begins."""
    try:
        spans = doc.get("texts", [])
        out_idx = set()
        for i, s in enumerate(spans):
            if s.get("page_no") != 1:
                continue
            txt = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
            if "exact name of registrant" in txt:
                for j in range(i, min(len(spans), i + 10)):
                    if spans[j].get("page_no") == 1:
                        out_idx.add(j)
        return [spans[i] for i in sorted(out_idx)]
    except Exception:
        return []
