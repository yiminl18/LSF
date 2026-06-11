def rule_parent_of_state_jurisdiction_cluster(doc: dict) -> list[dict]:
    """Match the common parent span of state/jurisdiction, file number, EIN, and address cover fields."""
    try:
        texts = doc.get("texts", [])
        parent_counts = {}
        for child in texts:
            if child.get("page_no") != 1:
                continue
            blob = ((child.get("text", "") or "") + " " + (child.get("text_span", "") or "")).lower()
            if any(k in blob for k in [
                "state or other jurisdiction of incorporation",
                "commission file number",
                "i.r.s. employer identification no",
                "irs employer identification no",
                "address of principal executive offices"
            ]):
                pid = child.get("structure", {}).get("parent_id")
                if pid is not None:
                    parent_counts[pid] = parent_counts.get(pid, 0) + 1
        out = []
        for pid, cnt in parent_counts.items():
            if cnt >= 2 and 0 <= pid < len(texts):
                out.append(texts[pid])
        return out
    except Exception:
        return []
