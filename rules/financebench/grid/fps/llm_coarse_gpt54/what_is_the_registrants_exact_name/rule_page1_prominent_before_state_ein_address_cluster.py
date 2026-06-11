def rule_page1_prominent_before_state_ein_address_cluster(doc: dict) -> list[dict]:
    """Match prominent page-1 spans before the cluster of state, EIN, and address fields."""
    try:
        texts = doc.get("texts", [])
        out = []
        cluster_idx = None
        for i, span in enumerate(texts):
            low = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "state or other jurisdiction of incorporation" in low or
                "i.r.s. employer identification" in low or
                "irs employer identification" in low or
                "address of principal executive offices" in low
            ):
                cluster_idx = i
                break
        if cluster_idx is None:
            return []
        for j in range(max(0, cluster_idx - 5), cluster_idx):
            s = texts[j]
            if s.get("page_no") == 1 and s.get("bold") == 1 and float(s.get("size") or 0) >= 10:
                out.append(s)
        return out
    except Exception:
        return []
