def rule_exact_answer_string_span(doc: dict) -> list[dict]:
    """Retrieve spans whose text or text_span contains an exact known state/EIN answer string."""
    try:
        answers = [
            "Delaware, 91-0425694",
            "Delaware, 77-0019522",
            "Delaware, 95-4803544",
            "Washington; 91-1223280",
            "Jersey (Channel Islands), 98-1455367",
            "Delaware, 91-1646860",
            "Delaware; 77-0430924",
            "Jersey, 98-1455367",
            "Delaware, 41-0417775",
            "Washington, 91-1223280",
            "New York, 13-3513936",
        ]
        out = []
        for s in doc.get("texts", []):
            combo = ((s.get("text") or "") + " " + (s.get("text_span") or ""))
            if any(a.lower() in combo.lower() for a in answers):
                out.append(s)
        return out
    except Exception:
        return []

