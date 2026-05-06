def rule_page1_parent_of_state_ein_address_cluster(doc: dict) -> list[dict]:
    """Match parent headings of state/EIN/address cover-page metadata clusters."""
    try:
        texts = doc.get("texts", [])
        out = []
        triggers = [
            "state or other jurisdiction of incorporation",
            "state of incorporation",
            "i.r.s. employer identification no",
            "irs employer identification no",
            "address of principal executive offices",
            "address and telephone number, including area code, of registrant",
        ]
        for span in texts:
            txt = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if any(t in txt for t in triggers):
                pid = span.get("structure", {}).get("parent_id")
                if pid is not None and 0 <= pid < len(texts):
                    out.append(texts[pid])
        return out
    except Exception:
        return []
