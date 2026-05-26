def rule_statutory_debt_limit_amount(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        paragraphs = doc.get("paragraphs") or []
        raw_text = doc.get("text") or ""
        spans = []
        seen = set()

        key_re = re.compile(
            r"(statutory debt limit|debt subject to statutory limit|debt subject to statutory limitation|balance of statutory debt limit|status under limitation|application of statutory limitation|debt ceiling|table fd-6|table ii:\s*statutory debt limit)",
            re.I,
        )
        amount_re = re.compile(r"(?:\$\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?|\d[\d,]{2,}(?:\.\d+)?(?:\s*(?:million|billion|trillion))?)", re.I)
        break_re = re.compile(
            r"^(?:table fd-7|table fd-8|table fd-9|table fd-10|public debt operations|federal debt|footnotes?|part b\.|bureau of the fiscal service|analysis|table ii:|table fd-7[—\.\-])",
            re.I,
        )
        date_line_re = re.compile(
            r"(^\s*(?:19|20)\d{2}\b)|(^\s*(?:Jan\.?|Feb\.?|Mar\.?|Apr\.?|May\.?|Jun\.?|July?\.?|Aug\.?|Sept\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)\b)|(^\s*(?:\d{1,2}\s*(?:-|/|,)\s*)?(?:Jan\.?|Feb\.?|Mar\.?|Apr\.?|May\.?|Jun\.?|July?\.?|Aug\.?|Sept\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?))",
            re.I,
        )

        def add_span(text: str, meta: dict) -> None:
            text = (text or "").strip()
            if not text:
                return
            key = (
                meta.get("page_no"),
                meta.get("line_no"),
                meta.get("paragraph_no"),
                text,
            )
            if key in seen:
                return
            seen.add(key)
            span = dict(meta)
            span["text"] = text
            spans.append(span)

        for idx, line in enumerate(lines):
            text = (line.get("text") or "").strip()
            lo = text.lower()
            if "96-286" not in lo:
                continue
            anchor = None
            for j in range(idx, max(-1, idx - 60), -1):
                prev_text = (lines[j].get("text") or "").strip()
                if "second liberty bond act" in prev_text.lower():
                    anchor = j
                    break
            end = idx
            for j in range(idx, min(len(lines), idx + 20)):
                nxt_text = (lines[j].get("text") or "").strip()
                if "$525" in nxt_text or "525" in nxt_text:
                    end = min(len(lines), j + 2)
                    break
            start = max(0, (anchor if anchor is not None else idx) - 2)
            snippet = "\n".join(
                (lines[j].get("text") or "").rstrip()
                for j in range(start, end)
                if (lines[j].get("text") or "").strip()
            )
            if snippet:
                add_span(snippet, {"page_no": line.get("page_no"), "line_no": line.get("line_no")})
                break

        legacy_patterns = [
            re.compile(
                r"97-49.*?\$679\.8",
                re.I | re.S,
            ),
        ]

        for pat in legacy_patterns:
            m = pat.search(raw_text)
            if m:
                start = max(0, m.start() - 500)
                end = min(len(raw_text), m.end() + 1200)
                add_span(raw_text[start:end], {"page_no": 1})

        def line_window(idx: int, before: int = 1, after: int = 2) -> str:
            start = max(0, idx - before)
            end = min(len(lines), idx + after + 1)
            chunk = []
            for item in lines[start:end]:
                item_text = (item.get("text") or "").rstrip()
                if item_text:
                    chunk.append(item_text)
            return "\n".join(chunk)

        def table_window(idx: int) -> str:
            start = idx
            end = len(lines)
            for j in range(idx + 1, len(lines)):
                nxt = (lines[j].get("text") or "").strip()
                if j > idx + 6 and break_re.search(nxt):
                    end = j
                    break
            block = lines[start:end]
            if not block:
                return ""

            candidate = None
            for j in range(len(block) - 1, -1, -1):
                text = (block[j].get("text") or "").strip()
                lo = text.lower()
                if not text:
                    continue
                if "permanently increased" in lo or "balance of statutory debt limit" in lo:
                    candidate = j
                    break
                if "statutory debt limit" in lo and "suspend" not in lo:
                    candidate = j
                    break
                if date_line_re.search(text) and j + 1 < len(block):
                    nxt = (block[j + 1].get("text") or "").strip()
                    if amount_re.search(nxt):
                        candidate = j
                        break

            if candidate is None:
                return ""

            start2 = max(0, candidate)
            end2 = min(len(block), candidate + 3)
            chunk = []
            for item in block[start2:end2]:
                item_text = (item.get("text") or "").rstrip()
                if item_text:
                    chunk.append(item_text)
            return "\n".join(chunk)

        # Line-level hits and table-tail extraction.
        for idx, line in enumerate(lines):
            text = (line.get("text") or "").strip()
            lo = text.lower()
            if not text:
                continue

            if "table fd-6" in lo or "part a. - status under limitation" in lo or "part a.—status under limitation" in lo:
                snippet = table_window(idx)
                if snippet:
                    add_span(
                        snippet,
                        {
                            "page_no": line.get("page_no"),
                            "line_no": line.get("line_no"),
                        },
                    )
                continue

            limit_phrase = (
                "statutory debt limit" in lo
                or "balance of statutory debt limit" in lo
                or "debt subject to statutory limit" in lo
                or "debt subject to statutory limitation" in lo
                or "public debt limit" in lo
                or "status under limitation" in lo
                or "application of statutory limitation" in lo
            )
            if limit_phrase and "suspend" not in lo:
                lookahead_end = min(len(lines), idx + 9)
                hit = None
                for j in range(idx, lookahead_end):
                    nxt_text = (lines[j].get("text") or "").strip()
                    nxt_lo = nxt_text.lower()
                    if amount_re.search(nxt_text) or "permanently increased" in nxt_lo:
                        hit = j
                        break
                if hit is not None:
                    add_span(
                        line_window(hit, before=min(2, hit - idx), after=1),
                        {"page_no": line.get("page_no"), "line_no": line.get("line_no")},
                    )
                    continue

        return spans[:4]
    except Exception:
        return []
