def rule_tables_before_notes_section(doc: dict) -> list[dict]:
    """Match tables that appear before notes to financial statements within Item 8, where primary statements usually sit."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "item 8" not in path and "financial statements and supplementary data" not in path:
                continue
            later = texts[i:i+20]
            later_text = " ".join((x.get("text") or "").lower() for x in later)
            if "notes to consolidated financial statements" in later_text or "notes to financial statements" in later_text:
                out.append(span)
        return out
    except Exception:
        return []
