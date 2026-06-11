def rule_page1_h1_near_exact_name_and_state(doc: dict) -> list[dict]:
    """Match page-1 H1 spans when nearby following spans include exact-name label and state label within a short window."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            window = texts[i+1:i+12]
            has_exact = any("exact name of registrant" in (w.get("text") or "").lower() for w in window)
            has_state = any("state or other jurisdiction" in (w.get("text") or "").lower() for w in window)
            if has_exact and has_state:
                out.append(span)
        return out
    except Exception:
        return []
