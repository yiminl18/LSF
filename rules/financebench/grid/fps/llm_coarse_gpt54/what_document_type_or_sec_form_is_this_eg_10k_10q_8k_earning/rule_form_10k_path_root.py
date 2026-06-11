def rule_form_10k_path_root(doc: dict) -> list[dict]:
    """Match spans under a root path_text exactly equal to FORM 10-K."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "").strip().upper()
            if path == "FORM 10-K":
                out.append(span)
        return out
    except Exception:
        return []
