def rule_page12_state_within_four_spans_before_incorporation_label(doc: dict) -> list[dict]:
    """Match short page-1/2 state values within four spans before the incorporation cover label."""
    try:
        import re

        def _norm(text: str) -> str:
            text = (text or "").replace("I.R.S.", "IRS").replace("i.r.s.", "irs")
            return re.sub(r"\s+", " ", text).strip()

        def _looks_ein(text: str) -> bool:
            return bool(re.search(r"\b\d{2}-\d{7}\b", _norm(text)))

        def _looks_state_value(text: str) -> bool:
            text = _norm(text)
            lowered = text.lower()
            if not text or len(text) > 80:
                return False
            if any(ch.isdigit() for ch in text):
                return False
            if not any(ch.isalpha() for ch in text):
                return False
            if "&" in text:
                return False
            if any(
                marker in lowered
                for marker in (
                    "exact name",
                    "state or other jurisdiction",
                    "employer identification",
                    "commission file",
                    "address of",
                    "zip code",
                    "telephone number",
                    "securities registered",
                    "washington, d.c. 20549",
                    "washington d.c. 20549",
                    "form 10-k",
                    "form 10-q",
                    "form 8-k",
                    "current report",
                    "annual report",
                    "quarterly report",
                )
            ):
                return False
            if re.search(r"\b(inc|corp|corporation|company|plc|ltd|limited|johnson)\b", lowered):
                return False
            return len(text.split()) <= 4

        results = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue

            text = _norm(span.get("text") or "")
            lowered = text.lower()
            if "state or other jurisdiction of" not in lowered:
                continue
            if _looks_ein(text) or "commission file" in lowered or "employer identification" in lowered:
                continue

            next_text = ""
            if i + 1 < len(texts) and texts[i + 1].get("page_no") == span.get("page_no"):
                next_text = _norm(texts[i + 1].get("text") or "").lower()

            if (
                "incorporation" not in lowered
                and "incorporation" not in next_text
                and "organization" not in lowered
                and "organization" not in next_text
            ):
                continue

            for j in range(i - 1, max(-1, i - 5), -1):
                candidate = texts[j]
                if candidate.get("page_no") != span.get("page_no"):
                    continue
                if _looks_state_value(candidate.get("text") or ""):
                    results.append(candidate)
                    break

        return results
    except Exception:
        return []
