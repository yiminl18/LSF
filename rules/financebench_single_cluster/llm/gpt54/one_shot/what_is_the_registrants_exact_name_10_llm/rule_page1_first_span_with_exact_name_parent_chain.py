def rule_page1_first_span_with_exact_name_parent_chain(doc: dict) -> list[dict]:
    """Match ancestors and nearby spans around the exact-name caption to maximize recall."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "exact name of registrant" in txt:
                ids = []
                pid = span.get("structure", {}).get("parent_id")
                if pid is not None:
                    ids.append(pid)
                    p2 = texts[pid].get("structure", {}).get("parent_id") if 0 <= pid < len(texts) else None
                    if p2 is not None:
                        ids.append(p2)
                for idx in ids:
                    if idx is not None and 0 <= idx < len(texts):
                        out.append(texts[idx])
                for j in range(max(0, i - 3), i):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
