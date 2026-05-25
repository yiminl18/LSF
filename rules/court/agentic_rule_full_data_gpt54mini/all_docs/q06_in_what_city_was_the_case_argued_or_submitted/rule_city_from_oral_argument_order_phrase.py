def rule_city_from_oral_argument_order_phrase(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        if not isinstance(lines, list):
            return []

        lines = sorted(
            [line for line in lines if isinstance(line, dict)],
            key=lambda x: (x.get("page_no", 0), x.get("line_no", 0)),
        )

        trigger_re = re.compile(
            r"(reschedule oral argument|oral argument to the week|submission for this case is vacated|submission is vacated)",
            re.I,
        )
        city_line_re = re.compile(
            r"^[A-Z][A-Za-z0-9'.-]*(?: [A-Z][A-Za-z0-9'.-]*)*, [A-Z][A-Za-z .'-]{2,}$"
        )

        def clean(text: str) -> str:
            return (text or "").strip().rstrip(" .;:")

        for idx, line in enumerate(lines[:80]):
            window = [clean(line.get("text", ""))]
            for nxt in lines[idx + 1 : idx + 3]:
                window.append(clean(nxt.get("text", "")))
            text = " ".join(part for part in window if part)
            if not text or not trigger_re.search(text):
                continue
            m = re.search(r"\bin\s+([A-Z][^.;:\n]+)", text)
            if m:
                city_text = clean(m.group(1))
                if not city_line_re.match(city_text):
                    continue
                out = {"text": city_text}
                if "page_no" in line:
                    out["page_no"] = line.get("page_no")
                if "line_no" in line:
                    out["line_no"] = line.get("line_no")
                return [out]
            for nxt in lines[idx + 1 : idx + 4]:
                nxt_text = clean(nxt.get("text", ""))
                if city_line_re.match(nxt_text):
                    out = {"text": nxt_text}
                    if "page_no" in nxt:
                        out["page_no"] = nxt.get("page_no")
                    if "line_no" in nxt:
                        out["line_no"] = nxt.get("line_no")
                    return [out]
        return []
    except Exception:
        return []
