import re


def rule_michigan_consumer_sentiment_latest_reading(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return " ".join((text or "").split())

        trigger_pat = re.compile(
            r"(?i)\b(?:reuters/?michigan|university of michigan|michigan index of consumer sentiment|"
            r"consumer sentiment survey|consumer sentiment index|index of consumer sentiment|"
            r"sentiment index)\b"
        )
        exclude_pat = re.compile(r"(?i)\b(?:conference board|consumer confidence)\b")

        reading_pats = [
            re.compile(
                r"(?i)\b(?:has\s+|had\s+)?(?:declined(?:\s+(?:\d+(?:\.\d+)?|[A-Za-z]+)){0,4}?\s+to|"
                r"fallen(?:\s+(?:\d+(?:\.\d+)?|[A-Za-z]+)){0,4}?\s+to|"
                r"fell(?:\s+(?:\d+(?:\.\d+)?|[A-Za-z]+)){0,4}?\s+to|"
                r"rose(?:\s+(?:\d+(?:\.\d+)?|[A-Za-z]+)){0,4}?\s+to|"
                r"stood at|standing at|"
                r"moving(?:\s+(?:\d+(?:\.\d+)?|[A-Za-z]+)){0,4}?\s+up\s+to|"
                r"moved(?:\s+(?:\d+(?:\.\d+)?|[A-Za-z]+)){0,4}?\s+up\s+to|"
                r"moved(?:\s+(?:\d+(?:\.\d+)?|[A-Za-z]+)){0,4}?\s+higher,\s+reaching|"
                r"trended higher,\s+reaching|reaching)\s+"
                r"(?:just|only|about|roughly|around|approximately|nearly|close to|slightly|modestly|marginally|"
                r"somewhat|a bit|a record low of|a record high of|a multi-year low of|a multi-year high of|"
                r"a 14-year high of|a 18-year high of)?\s*"
                r"\d+(?:\.\d+)?(?:\s+(?:in|during|by|at|as of|for)\s+[^.;,\n]{1,40})?"
            ),
            re.compile(
                r"(?i)\b(?:reached)\s+(?:a\s+)?(?:record low|record high|multi-year low|multi-year high|"
                r"14-year high|18-year high)\s+of\s+\d+(?:\.\d+)?"
            ),
        ]

        def best_phrase(block: str) -> str:
            block = norm(block)
            if not block:
                return ""
            best = ""
            best_pos = -1
            for pat in reading_pats:
                for m in pat.finditer(block):
                    if m.start() >= best_pos:
                        best_pos = m.start()
                        best = norm(m.group(0).rstrip(" ,;:"))
            return best

        def scan_text(text: str) -> list[str]:
            text = norm(text)
            if not text or not trigger_pat.search(text):
                return []

            block = text
            exclude_match = exclude_pat.search(block)
            if exclude_match:
                block = block[: exclude_match.start()]

            out: list[str] = []
            for pat in reading_pats:
                for m in pat.finditer(block):
                    phrase = norm(m.group(0).rstrip(" ,;:"))
                    if phrase:
                        out.append(phrase)
            return out

        def scan_windows(items: list[dict], key: str, meta_key: str, window_size: int = 12) -> list[dict]:
            ordered = [item for item in items if item.get(key)]
            best_hit: dict | None = None
            best_idx = 10**9
            for idx in range(len(ordered)):
                window = ordered[idx : idx + window_size]
                lead_block = norm(" ".join((item.get(key) or "") for item in window[:2]))
                if not lead_block or not trigger_pat.search(lead_block):
                    continue
                block = norm(" ".join((item.get(key) or "") for item in window))

                exclude_match = exclude_pat.search(block)
                if exclude_match:
                    block = block[: exclude_match.start()]

                phrase = best_phrase(block)
                if phrase:
                    if idx < best_idx:
                        best_idx = idx
                        first = window[0]
                        span = {"text": phrase}
                        if first.get("page_no") is not None:
                            span["page_no"] = first.get("page_no")
                        if first.get(meta_key) is not None:
                            span[meta_key] = first.get(meta_key)
                        best_hit = span
            return [best_hit] if best_hit else []

        ordered_paragraphs = sorted(
            [item for item in (doc.get("paragraphs") or []) if item.get("text")],
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("paragraph_no", 10**9),
            ),
        )
        ordered_lines = sorted(
            [item for item in (doc.get("lines") or []) if item.get("text")],
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )
        hits = scan_windows(ordered_lines, "text", "line_no", window_size=12)
        if hits:
            return hits

        for item in ordered_paragraphs:
            phrases = scan_text(item.get("text") or "")
            if phrases:
                return [
                    {
                        "text": phrases[-1],
                        "page_no": item.get("page_no"),
                        "paragraph_no": item.get("paragraph_no"),
                    }
                ]

        return []
    except Exception:
        return []
