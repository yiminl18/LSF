import re


def rule_fd5_avg_length_recent_fiscal_year(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        def line_text(item: dict) -> str:
            return (item.get("text") or "").strip()

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", text).lower()

        header_idx = None
        for i, item in enumerate(lines):
            text = norm(line_text(item))
            if text.startswith("table fd-5") and "maturity distribution" in text and "average length" in text:
                header_idx = i
                break

        if header_idx is None:
            return []

        block_end = len(lines)
        for j in range(header_idx + 1, min(len(lines), header_idx + 500)):
            text = norm(line_text(lines[j]))
            if text.startswith("fd-6") or text.startswith("table fd-6") or text.startswith("fd-7") or text.startswith("table fd-7") or text.startswith("fiscal service operations"):
                block_end = j
                break

        block = lines[header_idx:block_end]

        year_rows = []
        for i, item in enumerate(block):
            text = line_text(item)
            if re.match(r"^20\d{2}\s+\.{3,}\s*$", text) and "-" not in text and "sept" not in text.lower():
                year_rows.append(i)

        if not year_rows:
            return []

        row_start = year_rows[-1]
        row_end = row_start + 1
        while row_end < len(block):
            next_text = line_text(block[row_end])
            if re.match(r"^20\d{2}\b", next_text) and "-" not in next_text and "sept" not in next_text.lower():
                break
            if norm(next_text).startswith("fd-6") or norm(next_text).startswith("table fd-6") or norm(next_text).startswith("fd-7") or norm(next_text).startswith("table fd-7"):
                break
            row_end += 1

        row_lines = block[row_start:row_end]
        row_text = "\n".join(line_text(item) for item in row_lines if line_text(item))
        if not row_text:
            return []

        first = row_lines[0]
        span = {"text": row_text}
        if "page_no" in first:
            span["page_no"] = first.get("page_no")
        if "line_no" in first:
            span["line_no"] = first.get("line_no")
        if "line_no" in first and row_lines:
            span["line_no_end"] = row_lines[-1].get("line_no")

        return [span]
    except Exception:
        return []
