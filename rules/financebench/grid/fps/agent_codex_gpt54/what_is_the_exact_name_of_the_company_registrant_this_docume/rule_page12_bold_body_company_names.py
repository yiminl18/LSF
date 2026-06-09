def rule_page12_bold_body_company_names(doc: dict) -> list[dict]:
    """Match early bold body-text company names such as Apple's duplicated cover title."""
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
                    "securities registered",
                )
            ):
                return False
            if re.fullmatch(r"[0-9\W]+", text):
                return False

            words = re.findall(r"[A-Za-z&.,'\-]+", text)
            if len(words) < 2 or sum(ch.isalpha() for ch in text) < 4:
                return False

            return bool(re.search(r"\b(?:inc|corporation|company|plc|ltd|limited|holdings|incorporated)\b", lowered)) or ("&" in text and len(words) >= 2)

        results = []
        seen = set()

        for span in doc.get("texts", [])[:25]:
            if span.get("page_no", 999) > 2:
                continue
            if span.get("label") != "text":
                continue

            text = span.get("text", "")
            lowered = text.lower()
            if not _looks_company(text):
                continue
            if span.get("bold") != 1 and (span.get("size") or 0) < 12:
                continue
            if any(
                marker in lowered
                for marker in (
                    "today reported",
                    "today announced",
                    "today issued",
                    "today released",
                )
            ):
                continue

            idx = id(span)
            if idx not in seen:
                seen.add(idx)
                results.append(span)

        return results
    except Exception:
        return []
