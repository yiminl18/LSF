def rule_page12_inline_state_incorporation_span(doc: dict) -> list[dict]:
    """Match page-1/2 spans that inline the state value before the incorporation label."""
    try:
        import re

        def _norm(text: str) -> str:
            text = (text or "").replace("I.R.S.", "IRS").replace("i.r.s.", "irs")
            return re.sub(r"\s+", " ", text).strip()

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
                    "form 10-k",
                    "form 10-q",
                    "form 8-k",
                )
            ):
                return False
            if re.search(r"\b(inc|corp|corporation|company|plc|ltd|limited|johnson)\b", lowered):
                return False
            return len(text.split()) <= 4

        results = []
        for span in doc.get("texts", []):
            if span.get("page_no", 999) > 2:
                continue

            text = _norm(span.get("text") or "")
            lowered = text.lower()
            if "state or other jurisdiction of incorporation" not in lowered:
                continue
            if "commission file" in lowered or "employer identification" in lowered:
                continue

            before_label = re.split(
                r"state or other jurisdiction of incorporation(?: or organization)?",
                text,
                maxsplit=1,
                flags=re.I,
            )[0].strip(" -;,:()")
            before_label = before_label.split("(", 1)[0].strip()
            if _looks_state_value(before_label):
                results.append(span)

        return results
    except Exception:
        return []
