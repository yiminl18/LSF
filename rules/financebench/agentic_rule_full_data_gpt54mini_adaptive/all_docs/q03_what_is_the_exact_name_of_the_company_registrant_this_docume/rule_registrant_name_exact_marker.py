import re


def rule_registrant_name_exact_marker(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []

        def norm(value: str) -> str:
            return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()

        def make_span(raw_text: str, source: dict | None = None) -> dict:
            span = {"text": norm(raw_text)}
            if source:
                if source.get("page_no") is not None:
                    span["page_no"] = source["page_no"]
                if source.get("line_no") is not None:
                    span["line_no"] = source["line_no"]
            return span

        def add_span(spans: list[dict], seen: set[str], raw_text: str, source: dict | None = None) -> None:
            cleaned = norm(raw_text)
            if not cleaned or cleaned in seen:
                return
            seen.add(cleaned)
            spans.append(make_span(cleaned, source))

        def is_separator(text: str) -> bool:
            return bool(re.fullmatch(r"[_\-\.\s]+", text or ""))

        noise_phrases = (
            "united states",
            "securities and exchange commission",
            "washington, d.c.",
            "form 8-k",
            "form 10-q",
            "form 10-k",
            "current report",
            "annual report",
            "quarterly report",
            "transition report",
            "pursuant to section",
            "for the quarterly period ended",
            "for the quarter ended",
            "for the fiscal year ended",
            "for the year ended",
            "for the six months ended",
            "for the nine months ended",
            "for the transition period from",
            "date of report",
            "date of earliest event reported",
            "reported):",
            "commission file",
            "commission file number",
            "commission file no",
            "state or other jurisdiction",
            "irs employer identification",
            "address of principal executive offices",
            "registrant's telephone number",
            "registrant’s telephone number",
            "securities registered pursuant",
            "title of each class",
            "trading symbol",
            "name of each exchange",
            "indicate by check mark",
            "table of contents",
            "cautionary note regarding forward-looking statements",
            "documents incorporated by reference",
        )

        def is_noise(text: str) -> bool:
            cleaned = norm(text)
            if not cleaned or is_separator(cleaned):
                return True
            low = cleaned.lower()
            if not any(ch.isalpha() for ch in cleaned):
                return True
            if low in {"yes", "no", "o", "x", "or", "and", "of", "the", "☒", "☐"}:
                return True
            if any(phrase in low for phrase in noise_phrases):
                return True
            return False

        def is_marker_text(text: str) -> bool:
            low = norm(text).lower()
            return "exact name of registrant" in low

        def score_candidate(text: str, slice_len: int) -> int:
            low = norm(text).lower()
            score = 0
            if any(ch.isalpha() for ch in text):
                score += 3
            if 2 <= len(text) <= 90:
                score += 2
            if slice_len > 1:
                score += 4
            if re.search(r"\b(inc\.?|corp\.?|corporation|company|co\.?|plc|ltd\.?|limited|incorporated|holdings|group)\b", low):
                score += 4
            if re.fullmatch(r"[A-Z0-9 &'.,()\-\/]+", text):
                score += 2
            if re.search(r"\b(exact|registrant|commission|address|telephone|state|pursuant|exchange|issuer|report)\b", low):
                score -= 10
            if len(text) > 120:
                score -= 3
            return score

        spans: list[dict] = []
        seen: set[str] = set()

        scan_lines = lines[:80] if lines else []
        marker_idx = None
        for idx, line in enumerate(scan_lines):
            low = norm(line.get("text", "")).lower()
            if "exact name of registrant" in low:
                marker_idx = idx
                break
            if "exact" in low:
                window = " ".join(
                    norm(scan_lines[j].get("text", ""))
                    for j in range(idx, min(len(scan_lines), idx + 4))
                )
                if "name of registrant" in window.lower():
                    marker_idx = idx
                    break

        if marker_idx is None:
            return spans

        lookback_start = max(0, marker_idx - 6)
        lookback_lines = []
        for line in scan_lines[lookback_start:marker_idx]:
            raw = norm(line.get("text", ""))
            if not raw or is_noise(raw):
                continue
            lookback_lines.append(line)

        # The registrant name is typically one to three lines immediately above the marker.
        tail = lookback_lines[-4:] if lookback_lines else []
        best_text = ""
        best_source = None
        best_score = -10**9

        for start in range(len(tail)):
            for end in range(start, min(len(tail), start + 3)):
                slice_lines = tail[start : end + 1]
                candidate_text = norm(" ".join(norm(item.get("text", "")) for item in slice_lines))
                if not candidate_text or is_noise(candidate_text):
                    continue
                score = score_candidate(candidate_text, len(slice_lines))
                if score > best_score:
                    best_score = score
                    best_text = candidate_text
                    best_source = slice_lines[0]

        if best_text:
            add_span(spans, seen, best_text, best_source)
        else:
            # If the cover-page company line is missing from OCR, fall back to the
            # earliest sentence-initial legal name in the opening narrative.
            prefix = norm((doc.get("text") or "")[:25000])
            region = prefix
            item1_idx = region.lower().find("item 1. business")
            if item1_idx != -1:
                region = region[item1_idx : item1_idx + 8000]
            fallback_patterns = [
                r"(?:^|[\n\.]\s*)([A-Z0-9][A-Za-z0-9&’'.,\-]*(?:\s+[A-Z0-9][A-Za-z0-9&’'.,\-]*){0,6},\s*Inc\.)\s*,\s*incorporated\b",
                r"(?:^|[\n\.]\s*)([A-Z0-9][A-Za-z0-9&’'.,\-]*(?:\s+[A-Z0-9][A-Za-z0-9&’'.,\-]*){0,6},\s*Inc\.)\s*,\s*is\b",
                r"(?:^|[\n\.]\s*)(The\s+[A-Z0-9][A-Za-z0-9&’'.,\-]*(?:\s+[A-Z0-9][A-Za-z0-9&’'.,\-]*){0,6}\s+Company)\s*,?\s*(?:is|was|incorporated)\b",
                r"(?:^|[\n\.]\s*)([A-Z0-9][A-Za-z0-9&’'.,\-]*(?:\s+[A-Z0-9][A-Za-z0-9&’'.,\-]*){0,6}\s+(?:Corporation|Corp\.|PLC|Ltd\.|Limited))\s*,?\s*(?:is|was|incorporated)\b",
            ]

            def clean_fallback_candidate(candidate: str) -> str:
                parts = candidate.split()
                while parts and parts[0].lower().strip(".,") in {"business", "item", "part", "page"}:
                    parts = parts[1:]
                while len(parts) > 3 and parts and parts[0].lower().strip(".,") == "general":
                    parts = parts[1:]
                return norm(" ".join(parts))

            for pat in fallback_patterns:
                m = re.search(pat, region, flags=re.IGNORECASE)
                if not m:
                    continue
                candidate = clean_fallback_candidate(m.group(1))
                if candidate and not is_noise(candidate):
                    add_span(spans, seen, candidate)
                    break

        return spans
    except Exception:
        return []
