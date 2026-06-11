def rule_page1_state_and_ein_cluster(doc: dict) -> list[dict]:
    """Match clusters on page 1 where a state value and EIN value appear within a short window."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts)):
            if texts[i].get("page_no") != 1:
                continue
            window = texts[i:i+6]
            joined = " ".join((s.get("text") or "") for s in window)
            if re.search(r"(Delaware|New York|New Jersey|Washington|California|Minnesota|Jersey)", joined, re.I) and re.search(r"\d{2}-\d{7}", joined):
                out.extend(window)
        # dedupe preserving order
        seen = set()
        dedup = []
        for s in out:
            key = id(s)
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        return dedup
    except Exception:
        return []
