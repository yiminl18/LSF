def rule_page1_2_section12b_window_without_table(doc: dict) -> list[dict]:
    """Capture the first page-1/2 Section 12(b) span cluster when no exchange table is present."""
    try:
        texts = doc.get("texts", [])
        has_table = any(
            s.get("page_no", 999) <= 2
            and s.get("label") == "table"
            and "trading symbol" in s.get("text", "").lower()
            and "exchange" in s.get("text", "").lower()
            for s in texts
        )
        if has_table:
            return []

        out = []
        seen = set()
        for i, s in enumerate(texts):
            if (
                s.get("page_no", 999) <= 2
                and "securities registered pursuant to section 12(b)"
                in s.get("text", "").lower()
            ):
                page = s.get("page_no")
                for j in range(i, min(len(texts), i + 10)):
                    sj = texts[j]
                    if sj.get("page_no") != page:
                        continue
                    idx = id(sj)
                    if idx not in seen:
                        out.append(sj)
                        seen.add(idx)
                break
        return out
    except Exception:
        return []
