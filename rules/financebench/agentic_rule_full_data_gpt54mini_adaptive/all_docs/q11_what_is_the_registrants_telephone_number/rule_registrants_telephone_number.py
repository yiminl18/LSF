import re


def rule_registrants_telephone_number(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return " ".join((text or "").replace("\u00a0", " ").replace("\u2019", "'").replace("\u2018", "'").split())

        phone_patterns = [
            re.compile(r"\+\d{1,3}(?:[\s().-]*\d){6,}"),
            re.compile(r"\(\d{2,4}\)\s*\d{3}[\s.-]?\d{4}"),
            re.compile(r"\d{3}[\s.-]\d{3}[\s.-]\d{4}"),
            re.compile(r"\d{7,}"),
        ]
        label_pat = re.compile(r"(?i)\b(?:registrant'?s\s+)?telephone\s+number\b")

        def extract_phone(text: str) -> str:
            text = norm(text)
            if not text:
                return ""
            for pat in phone_patterns:
                m = pat.search(text)
                if m:
                    return m.group(0).strip(" .;,)")
            return ""

        def is_good_label(text: str) -> bool:
            low = text.lower()
            if not label_pat.search(text) and "telephone" not in low:
                return False
            return True

        lines = sorted(
            doc.get("lines") or [],
            key=lambda x: (x.get("page_no", 10**9), x.get("line_no", 10**9)),
        )
        paragraphs = sorted(
            doc.get("paragraphs") or [],
            key=lambda x: (x.get("page_no", 10**9), x.get("paragraph_no", 10**9)),
        )

        def make_span(text: str, src: dict) -> dict:
            span = {"text": text}
            if src.get("page_no") is not None:
                span["page_no"] = src.get("page_no")
            if src.get("line_no") is not None:
                span["line_no"] = src.get("line_no")
            if src.get("paragraph_no") is not None:
                span["paragraph_no"] = src.get("paragraph_no")
            return span

        def scan_items(items: list[dict], kind: str) -> dict | None:
            limit = 140 if kind == "line" else 40
            ordered = items[:limit]
            for idx, item in enumerate(ordered):
                text = norm(item.get("text"))
                if not text:
                    continue
                if not is_good_label(text):
                    continue

                window_start = max(0, idx - 3)
                window_end = min(len(ordered), idx + 4)
                for j in range(window_start, window_end):
                    candidate = ordered[j]
                    candidate_text = norm(candidate.get("text"))
                    if not candidate_text:
                        continue
                    phone = extract_phone(candidate_text)
                    if phone:
                        return make_span(phone, candidate)
            return None

        hit = scan_items(lines, "line")
        if hit:
            return [hit]

        hit = scan_items(paragraphs, "paragraph")
        if hit:
            return [hit]

        full_text = norm(doc.get("text") or "")
        if full_text:
            for marker in [
                "registrant's telephone number, including area code",
                "registrant’s telephone number, including area code",
                "telephone number, including area code",
                "telephone number",
            ]:
                idx = full_text.lower().find(marker)
                if idx == -1:
                    continue
                tail = full_text[idx : idx + 220]
                phone = extract_phone(tail)
                if phone:
                    return [{"text": phone}]

        return []
    except Exception:
        return []
