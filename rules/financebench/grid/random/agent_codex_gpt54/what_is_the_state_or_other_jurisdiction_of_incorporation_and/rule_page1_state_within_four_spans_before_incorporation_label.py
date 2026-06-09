def rule_page1_state_within_four_spans_before_incorporation_label(doc: dict) -> list[dict]:
    """Match the nearest short page-1 state value within four spans before the incorporation cover label."""
    try:
        import re

        def normalize(text: str) -> str:
            text = (text or "").strip().lower()
            text = text.replace("i.r.s.", "irs")
            return re.sub(r"\s+", " ", text)

        def looks_state_value(text: str) -> bool:
            text = (text or "").strip()
            low = normalize(text)
            if not text or len(text) > 80:
                return False
            if any(ch.isdigit() for ch in text):
                return False
            if not any(ch.isalpha() for ch in text):
                return False
            if any(
                marker in low
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
            if re.search(r"\b(inc|corp|corporation|company|plc|ltd|limited)\b", low):
                return False
            if "," in text and "(" not in text:
                return False
            return len(text.split()) <= 4

        hits: list[dict] = []
        texts = doc.get("texts", [])
        for idx, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if "state or other jurisdiction of incorporation" not in normalize(span.get("text") or ""):
                continue
            for prev_idx in range(idx - 1, max(-1, idx - 5), -1):
                candidate = texts[prev_idx]
                if candidate.get("page_no") != 1:
                    continue
                if looks_state_value(candidate.get("text") or ""):
                    hits.append(candidate)
                    break
        return hits
    except Exception:
        return []
