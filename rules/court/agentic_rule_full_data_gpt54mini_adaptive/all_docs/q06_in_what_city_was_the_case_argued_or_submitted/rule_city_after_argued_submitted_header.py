def rule_city_after_argued_submitted_header(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        if not isinstance(lines, list):
            return []

        lines = sorted(
            [line for line in lines if isinstance(line, dict)],
            key=lambda x: (x.get("page_no", 0), x.get("line_no", 0)),
        )

        header_re = re.compile(r"\b(argued|submitted|resubmitted)\b", re.I)
        city_re = re.compile(
            r"^[A-Z][A-Za-z0-9'.-]*(?: [A-Z][A-Za-z0-9'.-]*)*, [A-Z][A-Za-z .'-]{2,}$"
        )

        def norm_text(x):
            return (x or "").strip()

        for idx, line in enumerate(lines[:120]):
            text = norm_text(line.get("text"))
            if not text or not header_re.search(text):
                continue
            if not re.search(r"\b\d{4}\b", text):
                continue
            for nxt in lines[idx + 1 : idx + 5]:
                nxt_text = norm_text(nxt.get("text"))
                if not nxt_text:
                    continue
                if city_re.match(nxt_text):
                    out = {"text": nxt_text}
                    if "page_no" in nxt:
                        out["page_no"] = nxt.get("page_no")
                    if "line_no" in nxt:
                        out["line_no"] = nxt.get("line_no")
                    return [out]
                if "Filed" in nxt_text or "Before:" in nxt_text:
                    break
        return []
    except Exception:
        return []
