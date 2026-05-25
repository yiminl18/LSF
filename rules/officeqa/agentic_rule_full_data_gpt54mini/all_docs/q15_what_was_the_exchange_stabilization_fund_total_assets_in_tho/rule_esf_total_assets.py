import re


def rule_esf_total_assets(doc: dict) -> list[dict]:
    try:
        heading_re = re.compile(
            r"^\s*(?:INTRODUCTION[:\-—]?\s*)?EXCHANGE STABILIZATION FUND\.?\s*$",
            re.IGNORECASE,
        )
        total_assets_re = re.compile(r"^\s*total assets\b", re.IGNORECASE)

        def make_span(items: list[dict], span_type: str) -> dict:
            first = items[0]
            last = items[-1]
            text = "\n".join(str(item.get("text", "")) for item in items if str(item.get("text", "")).strip())
            span = {
                "text": text,
                "span_type": span_type,
            }
            if first.get("page_no") is not None:
                span["page_no"] = first.get("page_no")
            if first.get("line_no") is not None:
                span["line_no"] = first.get("line_no")
            if first.get("paragraph_no") is not None:
                span["paragraph_no"] = first.get("paragraph_no")
            if last.get("page_no") is not None:
                span["end_page_no"] = last.get("page_no")
            if last.get("line_no") is not None:
                span["end_line_no"] = last.get("line_no")
            if last.get("paragraph_no") is not None:
                span["end_paragraph_no"] = last.get("paragraph_no")
            return span

        spans: list[dict] = []

        paragraphs = doc.get("paragraphs") or []
        heading_idx = None
        for idx, paragraph in enumerate(paragraphs):
            text = str(paragraph.get("text", ""))
            if heading_re.fullmatch(text.strip()):
                heading_idx = idx
                break

        if heading_idx is not None:
            total_idx = None
            for idx in range(heading_idx + 1, len(paragraphs)):
                text = str(paragraphs[idx].get("text", ""))
                if total_assets_re.search(text):
                    total_idx = idx
                    break
            if total_idx is not None:
                start = max(0, total_idx - 2)
                end = min(len(paragraphs), total_idx + 4)
                block = [p for p in paragraphs[start:end] if str(p.get("text", "")).strip()]
                if block:
                    spans.append(make_span(block, "paragraph"))

        lines = doc.get("lines") or []
        if lines:
            line_heading_idx = None
            for idx, line in enumerate(lines):
                text = str(line.get("text", ""))
                if heading_re.fullmatch(text.strip()):
                    line_heading_idx = idx
                    break
            if line_heading_idx is not None:
                total_idx = None
                for idx in range(line_heading_idx + 1, len(lines)):
                    text = str(lines[idx].get("text", ""))
                    if total_assets_re.search(text):
                        total_idx = idx
                        break
                if total_idx is not None:
                    start = max(0, total_idx - 20)
                    end = min(len(lines), total_idx + 8)
                    block = [ln for ln in lines[start:end] if str(ln.get("text", "")).strip()]
                    if block:
                        line_span = make_span(block, "line")
                        if not spans or line_span["text"] != spans[0]["text"]:
                            spans.append(line_span)

        return spans
    except Exception:
        return []
