def rule_page1_top_date_before_dear_and_cpf(doc: dict) -> list[dict]:
    """Match top-of-page standalone dates that sit between the notice header and the CPF block."""
    try:
        import re

        texts = doc.get("texts", [])
        date_re = re.compile(
            r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\s*,?\s*\d{4}$",
            re.I,
        )
        notice_re = re.compile(r"(?:notice of probable violation|proposed compliance order|proposed civil penalty)", re.I)

        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1 or span.get("label") != "text":
                continue
            size = span.get("size")
            if size is None or float(size) < 11.0 or i > 10:
                continue
            raw = span.get("text") or ""
            norm = re.sub(r"(?<=\d)\s+(?=\d)", "", " ".join(raw.split()))
            if not date_re.match(norm):
                continue

            prev_spans = [s for s in texts[max(0, i - 4):i] if s.get("page_no") == 1]
            if not any(notice_re.search((s.get("text") or "")) for s in prev_spans):
                continue

            next_spans = [s for s in texts[i + 1:i + 6] if s.get("page_no") == 1]
            if not any(((s.get("text") or "").strip().lower().startswith("dear ")) for s in next_spans):
                continue
            if not any("cpf" in ((s.get("text") or "").lower()) for s in texts[i + 1:i + 8] if s.get("page_no") == 1):
                continue

            out.append(span)
        return out
    except Exception:
        return []
