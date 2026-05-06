def rule_debt_or_borrowings_note_table(doc: dict) -> list[dict]:
    """Tables under a Debt / Borrowings / Long-Term Debt note section."""
    try:
        import re
        out = []
        for sp in doc.get("texts", []):
            if sp.get("label") != "table":
                continue
            path = ((sp.get("structure") or {}).get("path_text") or "").lower()
            segs = [s.strip() for s in path.split("|")]
            ok = False
            for seg in segs:
                if re.search(r"^(?:note\s*)?\d+\.?\s*[-—–]?\s*(?:long[- ]term\s+)?(debt|borrowings)\b", seg):
                    ok = True
                    break
                if seg in {"debt", "borrowings", "long-term debt", "long term debt", "long-term borrowings"}:
                    ok = True
                    break
                if seg.endswith("long-term debt") or seg.endswith("long term debt"):
                    ok = True
                    break
            if ok:
                out.append(sp)
        return out
    except Exception:
        return []
