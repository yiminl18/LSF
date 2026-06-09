def rule_early_8k_exhibit_text_lines(doc: dict) -> list[dict]:
    """Match page-1/3 8-K exhibit list lines that explicitly name an exhibit number and description."""
    try:
        import re

        exhibit_line_re = re.compile(r"^\s*(?:exhibit\s*)?\d+(?:\.\d+)*[A-Z]?\s*[:.]?\s+\S", re.IGNORECASE)

        return [
            span for span in doc.get("texts", [])
            if (span.get("page_no") or 99) <= 3
            and span.get("label") in {"text", "list_item", "section_header"}
            and "item 9.01" in " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            and exhibit_line_re.search(" ".join((span.get("text") or "").split()))
        ]
    except Exception:
        return []
