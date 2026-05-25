def rule_full_text_budget_sentence(doc: dict) -> list[dict]:
    try:
        import re

        text = doc.get("text") or ""
        if not text:
            return []

        spans: list[dict] = []
        seen = set()

        def compact(s: str) -> str:
            return "".join(ch for ch in s.lower() if ch.isalnum())

        patterns = [
            re.compile(
                r"(?i)(?:fiscal(?: year)?\s*\d{2,4}[^.:\n]{0,220}?\b\d+(?:\.\d+)?\s*(?:%|percent|per cent)\s*(?:of|in relation to|relative to|as a share of|share of)?\s*(?:gdp|gnp|nominal gdp|gross national product)\b[^.\n]{0,200})"
            ),
            re.compile(
                r"(?i)(?:\b\d+(?:\.\d+)?\s*(?:%|percent|per cent)\s*(?:of|in relation to|relative to|as a share of|share of)?\s*(?:gdp|gnp|nominal gdp|gross national product)\b[^.\n]{0,200})"
            ),
            re.compile(
                r"(?i)(?:\b(?:gdp|gnp|nominal gdp|gross national product)\b[^.\n]{0,200}?\b\d+(?:\.\d+)?\s*(?:%|percent|per cent)\b)"
            ),
        ]

        for pat in patterns:
            for m in pat.finditer(text):
                phrase = m.group(0).strip()
                start = max(0, text.rfind("\n", 0, m.start()) + 1)
                left_period = text.rfind(".", 0, m.start())
                left_nl = text.rfind("\n", 0, m.start())
                start = max(start, left_period + 1 if left_period != -1 else 0, left_nl + 1 if left_nl != -1 else 0)
                end_candidates = [
                    text.find(".", m.end()),
                    text.find("\n", m.end()),
                ]
                end_candidates = [x for x in end_candidates if x != -1]
                end = min(end_candidates) if end_candidates else min(len(text), m.end() + 240)
                snippet = text[start:end].strip()
                for candidate in (phrase, snippet):
                    if not candidate:
                        continue
                    low = candidate.lower()
                    ctext = compact(candidate)
                    if not any(term in low or term in ctext for term in ("fiscal", "fy", "budget", "deficit", "surplus", "receipts", "outlays")):
                        continue
                    key = (candidate,)
                    if key in seen:
                        continue
                    seen.add(key)
                    spans.append({"text": candidate})

        return spans
    except Exception:
        return []
