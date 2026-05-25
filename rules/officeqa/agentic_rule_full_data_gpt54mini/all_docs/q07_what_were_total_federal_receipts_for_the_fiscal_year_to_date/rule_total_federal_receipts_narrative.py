import re


def rule_total_federal_receipts_narrative(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").strip()).lower()

        def make_span(item: dict) -> dict:
            span = {"text": item.get("text") or ""}
            if item.get("page_no") is not None:
                span["page_no"] = item.get("page_no")
            if item.get("paragraph_no") is not None:
                span["paragraph_no"] = item.get("paragraph_no")
            if item.get("line_no") is not None:
                span["line_no"] = item.get("line_no")
            return span

        for idx, item in enumerate(lines):
            text = item.get("text") or ""
            low = norm(text)
            if not re.search(r"\bfiscal year\s+(20[2-9]\d)\s+to date\b", low):
                continue
            tail = text.split("to date", 1)[-1]
            numbers = re.findall(r"\d[\d,]*", tail)
            if not numbers:
                for j in range(idx + 1, min(len(lines), idx + 4)):
                    numbers.extend(re.findall(r"\d[\d,]*", lines[j].get("text") or ""))
                    if numbers:
                        break
            if not numbers:
                continue
            return [make_span({**item, "text": numbers[-1]})]

        return []
    except Exception:
        return []
