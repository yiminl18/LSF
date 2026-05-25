def rule_court_of_appeals_docket_numbers(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        spans = []
        seen = set()

        docket_re = re.compile(r"\b\d{1,2}(?:A)?-\d{1,6}(?:-\d{1,6})?\b")
        label_only_re = re.compile(r"(?i)^Nos?\.?$")
        label_anywhere_re = re.compile(r"(?i)\bNos?\.?")

        def add_matches(text: str, line: dict) -> None:
            for match in docket_re.finditer(text):
                value = match.group(0).rstrip(".,;:")
                key = (line.get("page_no"), line.get("line_no"), value)
                if key in seen:
                    continue
                seen.add(key)
                span = {"text": value}
                if "page_no" in line:
                    span["page_no"] = line["page_no"]
                if "line_no" in line:
                    span["line_no"] = line["line_no"]
                spans.append(span)

        limit = min(len(lines), 60)

        for i in range(limit):
            line = lines[i] or {}
            text = (line.get("text") or "").strip()
            if not text:
                continue
            if text.upper().startswith("D.C."):
                continue

            if label_anywhere_re.search(text):
                if docket_re.search(text):
                    add_matches(text, line)
                    if text.rstrip().endswith(",") or text.rstrip().endswith(";"):
                        for j in range(i + 1, min(limit, i + 6)):
                            next_line = lines[j] or {}
                            next_text = (next_line.get("text") or "").strip()
                            if not next_text:
                                continue
                            if next_text.upper().startswith("D.C."):
                                continue
                            if docket_re.search(next_text):
                                add_matches(next_text, next_line)
                    continue

                if label_only_re.fullmatch(text):
                    for j in range(i + 1, min(limit, i + 6)):
                        next_line = lines[j] or {}
                        next_text = (next_line.get("text") or "").strip()
                        if not next_text:
                            continue
                        if next_text.upper().startswith("D.C."):
                            continue
                        if docket_re.search(next_text):
                            add_matches(next_text, next_line)

        return spans
    except Exception:
        return []
