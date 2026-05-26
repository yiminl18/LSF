def rule_cover_page_state_ein_block(doc: dict) -> list[dict]:
    try:
        import re
        def n(t): return " ".join(str(t or "").replace("\u00a0", " ").split())
        def is_state(t):
            s = n(t).strip("()[]:;,. ")
            l = s.lower()
            bad = ("commission", "identification", "address", "telephone", "registrant", "exact name")
            return bool(
                s
                and len(s) <= 28
                and "," not in s
                and not any(ch.isdigit() for ch in s)
                and not any(b in l for b in bad)
                and re.fullmatch(r"[A-Za-z][A-Za-z .()'/-]{1,27}", s)
            )

        lines = sorted(
            [x for x in (doc.get("lines") or []) if (x.get("page_no") or 99) <= 2],
            key=lambda x: (x.get("page_no", 99), x.get("line_no", 9999)),
        )[:220]
        best, best_score = None, -1
        for i, line in enumerate(lines):
            page = line.get("page_no")
            part = " ".join(n(lines[j].get("text")) for j in range(i, min(len(lines), i + 4)) if lines[j].get("page_no") == page)
            low = part.lower()
            if "employer identification" not in low and not ("state" in low and "jurisdiction" in low and ("incorporation" in low or "organization" in low)):
                continue
            block = [x for x in lines[max(0, i - 3): min(len(lines), i + 5)] if x.get("page_no") == page]
            text = " ".join(n(x.get("text")) for x in block)
            score = 2 * ("employer identification" in text.lower()) + 2 * ("state" in text.lower() and "jurisdiction" in text.lower()) + 2 * any(is_state(x.get("text")) for x in block)
            if re.search(r"\b\d{2}-\d{7}\b", text) and score > best_score:
                best, best_score = block, score
        if best and best_score >= 4:
            return [{
                "text": "\n".join(n(x.get("text")) for x in best if n(x.get("text"))),
                "page_no": best[0].get("page_no"),
                "line_no": best[0].get("line_no"),
                "end_line_no": best[-1].get("line_no"),
            }]
        return []
    except Exception:
        return []
