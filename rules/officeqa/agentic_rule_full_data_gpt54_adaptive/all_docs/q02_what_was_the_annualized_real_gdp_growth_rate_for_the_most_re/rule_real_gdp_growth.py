import re


HEADING_COMPACTS = {
    "economicgrowth",
    "realgrossdomesticproduct",
    "realgrossdomesticproductgdp",
    "growthofrealgdp",
    "growthofrealgrossdomesticproduct",
}


def rule_real_gdp_growth(doc: dict) -> list[dict]:
    try:
        doc_name = doc.get("doc_name") or ""
        month_match = re.search(r"treasury_bulletin_\d{4}_(\d{2})", doc_name)
        target_quarter = {
            "03": "fourthquarter",
            "06": "firstquarter",
            "09": "secondquarter",
            "12": "thirdquarter",
        }.get(month_match.group(1) if month_match else "")

        def normalize(text: str) -> str:
            text = text or ""
            text = text.replace("\u00ad", "")
            text = re.sub(r"(\w)\s*-\s+(\w)", r"\1\2", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        def compact(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

        def has_gdp(text: str) -> bool:
            low = (text or "").lower()
            comp = compact(text)
            return (
                "realgdp" in comp
                or "realgrossdomesticproduct" in comp
                or ("grossdomesticproduct" in comp and "real" in low)
            )

        def has_quarter(text: str) -> bool:
            comp = compact(text)
            return any(
                token in comp
                for token in (
                    "firstquarter",
                    "secondquarter",
                    "thirdquarter",
                    "fourthquarter",
                    "latestquarter",
                    "mostrecentquarter",
                )
            )

        def has_annual(text: str) -> bool:
            comp = compact(text)
            return "annualrate" in comp or "annualized" in comp or "annualised" in comp

        def has_motion(text: str) -> bool:
            low = (text or "").lower()
            return any(
                cue in low
                for cue in (
                    "grew",
                    "rose",
                    "increased",
                    "declined",
                    "decreased",
                    "fell",
                    "expanded",
                    "slowed",
                    "accelerated",
                    "contracted",
                    "moderated",
                    "growth",
                )
            )

        def is_direct_gdp_rate(text: str) -> bool:
            low = (text or "").lower()
            return bool(
                re.search(
                    r"\b(?:real\s+gdp|real gross domestic product|gdp)\b[^.]{0,80}\b(?:grew|grow|rose|increased|declined|decreased|fell|expanded|slowed|accelerated|contracted|moderated)\b",
                    low,
                )
            )

        def is_candidate(text: str) -> bool:
            norm = normalize(text)
            low = norm.lower()
            if not norm:
                return False
            if "table of contents" in low or low == "contents":
                return False
            return has_gdp(norm) and has_quarter(norm) and has_annual(norm) and has_motion(norm)

        def sentence_span(text: str) -> str:
            norm = normalize(text)
            if not norm:
                return ""
            sentences = re.split(r"(?<=[.!?])\s+", norm)
            if not sentences:
                sentences = [norm]
            scored: list[tuple[int, int, str]] = []
            for idx, sent in enumerate(sentences):
                sent = sent.strip()
                if not sent or not has_quarter(sent):
                    continue
                prev_sent = sentences[idx - 1].strip() if idx > 0 else ""
                gdp_ctx = has_gdp(sent) or has_gdp(prev_sent)
                annual_ctx = has_annual(sent) or (has_gdp(prev_sent) and has_annual(prev_sent))
                motion_ctx = has_motion(sent) or has_motion(norm)
                if not (gdp_ctx and annual_ctx and motion_ctx):
                    continue
                sent_low = sent.lower()
                if (
                    ("contribution to real gdp" in sent_low or "percentage point" in sent_low)
                    and not is_direct_gdp_rate(sent)
                ):
                    continue
                score = 0
                sent_compact = compact(sent)
                if target_quarter and target_quarter in sent_compact:
                    score += 25
                if has_annual(sent):
                    score += 20
                if re.search(r"\d(?:[\d.,]|\s)*\s*(?:percent|per cent)\b", sent, flags=re.IGNORECASE):
                    score += 15
                if has_gdp(sent):
                    score += 10
                if is_direct_gdp_rate(sent):
                    score += 15
                if "advance estimate" in sent.lower() or "preliminary estimate" in sent.lower():
                    score += 5
                other_quarters = [
                    token
                    for token in ("firstquarter", "secondquarter", "thirdquarter", "fourthquarter")
                    if token != target_quarter and token in sent_compact
                ]
                score -= 8 * len(other_quarters)
                score += idx
                if not has_gdp(sent) and has_gdp(prev_sent):
                    sent = f"{prev_sent} {sent}".strip()
                scored.append((score, idx, sent))
            if scored:
                scored.sort(reverse=True)
                return scored[0][2]
            return norm if is_candidate(norm) else ""

        lines = doc.get("lines") or []
        for idx, line in enumerate(lines):
            if compact(line.get("text") or "") not in HEADING_COMPACTS:
                continue
            window_lines = []
            nonblank = 0
            for j in range(idx + 1, len(lines)):
                txt = (lines[j].get("text") or "").strip()
                if not txt:
                    if window_lines:
                        break
                    continue
                if compact(txt) in HEADING_COMPACTS and j > idx + 1:
                    break
                window_lines.append(txt)
                nonblank += 1
                if nonblank >= 14:
                    break
            joined = normalize(" ".join(window_lines))
            if not is_candidate(joined):
                continue
            snippet = sentence_span(joined) or joined
            return [{
                "page_no": line.get("page_no"),
                "line_no": line.get("line_no"),
                "text": snippet,
            }]

        paragraphs = doc.get("paragraphs") or []
        for para in paragraphs:
            text = para.get("text") or ""
            norm = normalize(text)
            if not norm:
                continue
            if "profile of the economy" not in norm.lower() and not is_candidate(norm):
                continue
            snippet = sentence_span(text) or sentence_span(norm)
            if snippet and (is_candidate(snippet) or is_candidate(norm)):
                return [{
                    "page_no": para.get("page_no"),
                    "paragraph_no": para.get("paragraph_no"),
                    "text": snippet,
                }]

        for idx in range(len(lines)):
            window_lines = []
            nonblank = 0
            for j in range(idx, len(lines)):
                txt = (lines[j].get("text") or "").strip()
                if not txt:
                    if window_lines:
                        break
                    continue
                window_lines.append(txt)
                nonblank += 1
                if nonblank >= 8:
                    break
            joined = normalize(" ".join(window_lines))
            if not is_candidate(joined):
                continue
            snippet = sentence_span(joined) or joined
            return [{
                "page_no": lines[idx].get("page_no"),
                "line_no": lines[idx].get("line_no"),
                "text": snippet,
            }]

        return []
    except Exception:
        return []
