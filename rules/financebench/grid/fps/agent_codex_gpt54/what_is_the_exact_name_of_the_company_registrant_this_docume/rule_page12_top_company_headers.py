def rule_page12_top_company_headers(doc: dict) -> list[dict]:
    """Match early page-1/2 header spans whose text is the company name."""
    try:
        import re

        def _norm(text: str) -> str:
            return " ".join((text or "").split())

        def _looks_company(text: str) -> bool:
            text = _norm(text)
            lowered = text.lower()
            if len(text) < 4 or len(text) > 160:
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
            if "http://" in lowered or "https://" in lowered:
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
            )

        results = []
        seen = set()

        for span in doc.get("texts", [])[:25]:
            if span.get("page_no", 999) > 2:
                continue
            if span.get("label") != "section_header":
                continue
            if (span.get("structure") or {}).get("level") not in ("H1", "H2", "H3"):
                continue

            text = span.get("text", "")
            lowered = text.lower()
            if not _looks_company(text):
                continue
            if any(
                marker in lowered
                for marker in (
                    "news release",
                    "form ",
                    "current report",
                    "annual report",
                    "quarterly report",
                    "transition report",
                    "signatures",
                    "part i",
                    "item ",
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
