def rule_page1_exact_exchange_value(doc: dict) -> list[dict]:
    """Match short page-1 spans whose text is itself an exchange name."""
    try:
        exact_values = {
            "new york stock exchange",
            "the new york stock exchange",
            "nyse",
            "nasdaq",
            "the nasdaq global select market",
            "nasdaq global select market",
            "the nasdaq global market",
            "nasdaq global market",
            "nasdaq capital market",
        }

        def normalize(text: str) -> str:
            return " ".join((text or "").split()).strip().lower()

        def is_exact_exchange(text: str) -> bool:
            normalized = normalize(text)
            if normalized in exact_values:
                return True
            parts = normalized.split()
            if len(parts) > 2 and len(parts) % 2 == 0:
                mid = len(parts) // 2
                first = " ".join(parts[:mid])
                second = " ".join(parts[mid:])
                return first == second and first in exact_values
            return False

        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") != "table"
            and is_exact_exchange(span.get("text") or "")
        ]
    except Exception:
        return []
