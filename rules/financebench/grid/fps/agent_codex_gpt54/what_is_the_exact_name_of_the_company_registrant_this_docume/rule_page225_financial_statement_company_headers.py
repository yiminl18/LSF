def rule_page225_financial_statement_company_headers(doc: dict) -> list[dict]:
    """Match short later-page financial headers that begin with the company name."""
    try:
        import re

        def _norm(text: str) -> str:
            return " ".join((text or "").split())

        def _looks_company(text: str) -> bool:
            text = _norm(text)
            lowered = text.lower()
            if len(text) < 4 or len(text) > 200:
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
                    "trading symbol",
                    "title of each class",
                    "name of each exchange",
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

        results = []
        seen = set()

        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            if span.get("page_no", 0) < 2 or span.get("page_no", 999) > 25:
                continue

            text = _norm(span.get("text", ""))
            lowered = text.lower()
            if not _looks_company(text):
                continue
            if not any(
                marker in lowered
                for marker in (
                    "consolidated",
                    "selected financial data",
                    "and subsidiaries",
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
