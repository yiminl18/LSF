def rule_page1_h1_with_no_pipe_and_no_item(doc: dict) -> list[dict]:
    """Match clean page-1 H1 spans that are not item headings, table headings, or pipe-formatted text."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("level") == "H1"
                and "|" not in txt
                and not re.search(r"\bItem\b|\bPART\b|TABLE OF CONTENTS|INDEX|CURRENT REPORT|FORM 10-|FORM 8-K|COMMISSION", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
