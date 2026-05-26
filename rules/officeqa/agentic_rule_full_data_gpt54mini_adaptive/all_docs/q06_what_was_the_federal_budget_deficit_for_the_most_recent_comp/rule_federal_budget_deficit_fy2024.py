import re


def rule_federal_budget_deficit_fy2024(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", text or "").strip()

        def make_span(item: dict, text: str) -> dict:
            span = {"text": text}
            if "page_no" in item:
                span["page_no"] = item["page_no"]
            if "paragraph_no" in item:
                span["paragraph_no"] = item["paragraph_no"]
            if "line_no" in item:
                span["line_no"] = item["line_no"]
            return span

        def has_answer_shape(text: str) -> bool:
            low = text.lower()
            if "deficit" not in low or "gdp" not in low:
                return False
            if not re.search(r"\bfy\s+\d{4}\b", low):
                return False
            return (
                "% of gdp" in low
                or "percent of gdp" in low
                or "equal to" in low
                or re.search(r"\b\d+(?:\.\d+)?%\s+of\s+gdp\b", low) is not None
                or re.search(r"\b\d+(?:\.\d+)?\s+percent\s+of\s+gdp\b", low) is not None
            )

        paragraphs = doc.get("paragraphs") or []
        heading_idx = None
        for idx, para in enumerate(paragraphs):
            text = norm(para.get("text", ""))
            if "federal budget deficit and debt" in text.lower():
                heading_idx = idx
                if has_answer_shape(text):
                    return [make_span(para, text)]
                break

        if heading_idx is not None:
            for para in paragraphs[heading_idx + 1 : heading_idx + 5]:
                text = norm(para.get("text", ""))
                if has_answer_shape(text):
                    return [make_span(para, text)]

        lines = doc.get("lines") or []
        heading_line_idx = None
        for idx, line in enumerate(lines):
            if "federal budget deficit and debt" in norm(line.get("text", "")).lower():
                heading_line_idx = idx
                break
        if heading_line_idx is not None:
            for i in range(heading_line_idx, min(len(lines), heading_line_idx + 8)):
                window_items = lines[i : i + 5]
                window_text = norm(" ".join(norm(item.get("text", "")) for item in window_items))
                if has_answer_shape(window_text):
                    return [make_span(window_items[0], window_text)]

        pages = doc.get("pages") or []
        for page in pages:
            text = norm(page.get("text", ""))
            if "federal budget deficit and debt" in text.lower() and has_answer_shape(text):
                return [make_span(page, text)]

        full_text = norm(doc.get("text", ""))
        low = full_text.lower()
        heading = low.find("federal budget deficit and debt")
        if heading != -1:
            start = max(0, heading - 40)
            end = min(len(full_text), heading + 900)
            snippet = full_text[start:end]
            if has_answer_shape(snippet):
                return [{"text": snippet}]

        return []
    except Exception:
        return []
