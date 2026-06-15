def rule_page12_zip_value_and_cue(doc: dict) -> list[dict]:
    """Match early page-1/2 ZIP-code value spans together with the ZIP cue span."""
    try:
        import re

        texts = doc.get("texts", [])
        out = []
        seen = set()

        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if "zip code" not in low:
                continue

            key = id(span)
            if key not in seen:
                out.append(span)
                seen.add(key)

            if i == 0:
                continue

            prev = texts[i - 1]
            if prev.get("page_no") != span.get("page_no"):
                continue

            prev_text = (prev.get("text") or "").replace("\n", " ").strip()
            prev_text = prev_text.replace("(Zip Code)", "").strip()
            if re.fullmatch(r"(?:\d{5}(?:-\d{4})?|[A-Z]{1,3}\d[A-Z\d]?\s?\d[A-Z]{2})", prev_text):
                prev_key = id(prev)
                if prev_key not in seen:
                    out.append(prev)
                    seen.add(prev_key)

        return out
    except Exception:
        return []
