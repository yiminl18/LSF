def rule_statutory_debt_limit_suspended(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        paragraphs = doc.get("paragraphs") or []
        spans = []
        seen = set()

        suspend_re = re.compile(r"\bsuspend(?:ed|sion|ing)?\b", re.I)
        key_re = re.compile(r"(statutory debt limit|debt limit|borrowing limit|debt ceiling)", re.I)

        def add_span(text: str, meta: dict) -> None:
            text = (text or "").strip()
            if not text:
                return
            key = (
                meta.get("page_no"),
                meta.get("line_no"),
                meta.get("paragraph_no"),
                text,
            )
            if key in seen:
                return
            seen.add(key)
            span = dict(meta)
            span["text"] = text
            spans.append(span)

        for para in paragraphs:
            text = (para.get("text") or "").strip()
            lo = text.lower()
            if not text:
                continue
            if suspend_re.search(text) and key_re.search(text):
                add_span(
                    text,
                    {
                        "page_no": para.get("page_no"),
                        "paragraph_no": para.get("paragraph_no"),
                    },
                )

        for idx, line in enumerate(lines):
            text = (line.get("text") or "").strip()
            lo = text.lower()
            if not text:
                continue

            direct_hit = False
            if suspend_re.search(text) and key_re.search(text):
                direct_hit = True
            elif "borrowing limit" in lo and suspend_re.search(text):
                direct_hit = True

            if not direct_hit:
                continue

            start = idx
            end = min(len(lines), idx + 3)
            window = []
            for item in lines[start:end]:
                item_text = (item.get("text") or "").rstrip()
                if item_text:
                    window.append(item_text)
            if window:
                add_span(
                    "\n".join(window),
                    {
                        "page_no": line.get("page_no"),
                        "line_no": line.get("line_no"),
                    },
                )

        return spans[:4]
    except Exception:
        return []
