import re


def rule_individual_income_tax_receipts_quarter(doc: dict) -> list[dict]:
    try:
        paragraphs = doc.get("paragraphs") or []
        text = doc.get("text") or ""

        def norm(value: str) -> str:
            return re.sub(r"\s+", " ", (value or "")).strip().lower()

        def make_span(item: dict, answer: str) -> dict:
            span = {"text": answer}
            if item.get("page_no") is not None:
                span["page_no"] = item.get("page_no")
            if item.get("paragraph_no") is not None:
                span["paragraph_no"] = item.get("paragraph_no")
            if item.get("line_no") is not None:
                span["line_no"] = item.get("line_no")
            return span

        patterns = [
            re.compile(
                r"individual income tax receipts(?:\s*,?\s*net of refunds,?)?\s*were\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)\s*billion",
                re.I,
            ),
            re.compile(
                r"individual income taxes[—-]\s*individual income tax receipts(?:\s*,?\s*net of refunds,?)?\s*were\s*\$?([0-9][0-9,]*(?:\.[0-9]+)?)\s*billion",
                re.I,
            ),
        ]

        for paragraph in paragraphs:
            ptext = paragraph.get("text") or ""
            low = norm(ptext)
            if "individual income tax" not in low or "receipt" not in low:
                continue
            for pat in patterns:
                m = pat.search(low)
                if m:
                    return [make_span(paragraph, m.group(1))]

        low_text = norm(text)
        if "individual income tax" in low_text and "receipt" in low_text:
            for pat in patterns:
                m = pat.search(low_text)
                if m:
                    return [make_span({}, m.group(1))]

        return []
    except Exception:
        return []
