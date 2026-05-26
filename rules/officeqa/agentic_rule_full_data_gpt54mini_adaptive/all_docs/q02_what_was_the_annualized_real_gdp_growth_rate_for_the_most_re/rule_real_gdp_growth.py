import re


def rule_real_gdp_growth(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        heading_idx = None
        for idx, line in enumerate(lines):
            low = (line.get("text") or "").lower()
            if "chart" in low or "contents" in low:
                continue
            if (
                "real gross domestic product" in low
                or "economic growth" in low
                or "real gdp growth" in low
                or "growth of real gdp" in low
            ):
                heading_idx = idx
                break

        if heading_idx is not None:
            end_idx = min(len(lines), heading_idx + 70)
            for idx in range(heading_idx + 1, end_idx - 1):
                window_lines = []
                for j in range(idx, min(idx + 4, end_idx)):
                    txt = lines[j].get("text", "")
                    if txt:
                        window_lines.append(txt)
                if not window_lines:
                    continue
                window = re.sub(r"\s+", " ", " ".join(window_lines))
                low = window.lower()
                compact = re.sub(r"[^a-z0-9]+", "", low)
                if "chart" in low or "contents" in low:
                    continue
                if not (
                    "quarter" in compact
                    or "latestquarter" in compact
                    or "mostrecentquarter" in compact
                    or "firstquarter" in compact
                    or "secondquarter" in compact
                    or "thirdquarter" in compact
                    or "fourthquarter" in compact
                ):
                    continue
                if not (
                    "annualrate" in compact
                    or "annualized" in compact
                    or "annualised" in compact
                ):
                    continue
                if not (
                    "realgdp" in compact
                    or "grossdomesticproduct" in compact
                    or "gdpgrowth" in compact
                    or "growth" in low
                    or "grew" in low
                    or "rose" in low
                    or "increased" in low
                    or "declined" in low
                    or "decreased" in low
                    or "fell" in low
                    or "expanded" in low
                    or "slowed" in low
                    or "accelerated" in low
                    or "contracted" in low
                    or "moderated" in low
                ):
                    continue
                return [{
                    "page_no": lines[idx].get("page_no"),
                    "line_no": lines[idx].get("line_no"),
                    "text": window,
                }]

        paras = doc.get("paragraphs") or []
        for para in paras:
            text = para.get("text", "")
            if not text:
                continue
            low = text.lower()
            compact = re.sub(r"[^a-z0-9]+", "", low)
            if "chart" in low or "contents" in low:
                continue
            if not (
                "realgdp" in compact
                or "grossdomesticproduct" in compact
                or "gdpgrowth" in compact
            ):
                continue
            if not (
                "quarter" in compact
                or "latestquarter" in compact
                or "mostrecentquarter" in compact
                or "firstquarter" in compact
                or "secondquarter" in compact
                or "thirdquarter" in compact
                or "fourthquarter" in compact
            ):
                continue
            if not (
                "annualrate" in compact
                or "annualized" in compact
                or "annualised" in compact
                or "grew" in low
                or "rose" in low
                or "increased" in low
                or "declined" in low
                or "decreased" in low
                or "fell" in low
                or "expanded" in low
                or "slowed" in low
                or "accelerated" in low
                or "contracted" in low
                or "moderated" in low
            ):
                continue

            norm = re.sub(r"\s+", " ", text)
            norm = re.sub(r"(\w)-\s+(\w)", r"\1\2", norm)
            sentences = re.split(r"(?<=[.!?])\s+", norm)
            for sent in sentences:
                s_low = sent.lower()
                s_compact = re.sub(r"[^a-z0-9]+", "", s_low)
                if (
                    ("quarter" in s_compact or "latestquarter" in s_compact or "mostrecentquarter" in s_compact)
                    and ("annualrate" in s_compact or "annualized" in s_compact or "annualised" in s_compact)
                ):
                    return [{
                        "page_no": para.get("page_no"),
                        "paragraph_no": para.get("paragraph_no"),
                        "text": sent.strip(),
                    }]
        return []
    except Exception:
        return []
