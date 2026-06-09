def rule_page13_signature_registrant_company_spans(doc: dict) -> list[dict]:
    """Match short company-name spans in page-1/3 registrant and signature blocks."""
    try:
        import re

        def _norm(text: str) -> str:
            return " ".join((text or "").split())

        def _looks_company(text: str) -> bool:
            text = _norm(text)
            lowered = text.lower()
            if len(text) < 4 or len(text) > 140:
                return False
            if any(
                marker in lowered
                for marker in (
                    "securities and exchange commission",
                    "commission file",
                    "current report",
                    "annual report",
                    "quarterly report",
                    "transition report",
                    "form 10-k",
                    "form 10-q",
                    "form 8-k",
                    "washington, d.c.",
                    "washington, dc",
                    "date of report",
                    "news release",
                    "signatures",
                    "item ",
                    "trading symbol",
                    "title of each class",
                    "name of each exchange",
                    "state or other jurisdiction",
                    "address of principal executive offices",
                    "registrant's telephone",
                )
            ):
                return False
            if re.fullmatch(r"[0-9\W]+", text):
                return False

            words = re.findall(r"[A-Za-z&.,'\-]+", text)
            if len(words) < 2 or sum(ch.isalpha() for ch in text) < 4:
                return False

            return (
                bool(re.search(r"\b(?:inc|corporation|company|plc|ltd|limited|holdings|incorporated)\b", lowered))
                or ("&" in text and len(words) >= 2)
                or (text.isupper() and len(words) >= 2)
                or (sum(word[:1].isupper() for word in words if word) >= max(2, len(words) - 1))
            )

        texts = doc.get("texts", [])
        results = []
        seen = set()

        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 3:
                continue

            text = span.get("text", "")
            lowered = text.lower()
            cleaned = text.replace("(Registrant)", "").replace("(registrant)", "")

            if "(registrant)" in lowered and _looks_company(cleaned):
                idx = id(span)
                if idx not in seen:
                    seen.add(idx)
                    results.append(span)
                continue

            if "has duly caused this report to be signed on its behalf" not in lowered:
                continue

            for candidate in texts[i + 1:i + 4]:
                if candidate.get("page_no", 999) > 3:
                    continue
                candidate_text = candidate.get("text", "")
                candidate_cleaned = candidate_text.replace("(Registrant)", "").replace("(registrant)", "")
                if not _looks_company(candidate_cleaned):
                    continue

                idx = id(candidate)
                if idx not in seen:
                    seen.add(idx)
                    results.append(candidate)

        return results
    except Exception:
        return []
