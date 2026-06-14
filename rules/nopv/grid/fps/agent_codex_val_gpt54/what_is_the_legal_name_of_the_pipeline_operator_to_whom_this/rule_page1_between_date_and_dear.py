def rule_page1_between_date_and_dear(doc: dict) -> list[dict]:
    """Match page-1 text spans between the first date line and the Dear salutation."""
    try:
        import re

        texts = doc.get("texts", [])
        month_re = re.compile(
            r"^(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            re.I,
        )

        date_idx = None
        for i, span in enumerate(texts):
            if span.get("page_no") != 1 or span.get("label") != "text":
                continue
            text = (span.get("text") or "").strip()
            if month_re.search(text):
                date_idx = i
                break

        if date_idx is None:
            return []

        out = []
        for span in texts[date_idx + 1:]:
            if span.get("page_no") != 1:
                break
            if span.get("label") != "text":
                continue
            text = (span.get("text") or "").strip()
            low = text.lower()
            if low.startswith("dear ") or "cpf" in low:
                break
            if text:
                out.append(span)
        return out
    except Exception:
        return []
