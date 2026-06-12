def rule_esf_path_text(doc: dict) -> list[dict]:
    """Match spans under a path_text containing Exchange Stabilization Fund."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "EXCHANGE STABILIZATION FUND" in path.upper():
                out.append(span)
        return out
    except Exception:
        return []
