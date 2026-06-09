def rule_page12_company_span_before_exact_name_anchor(doc: dict) -> list[dict]:
    """Match page-1/2 company-name spans immediately before the exact-name anchor."""
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
                or (sum(word[:1].isupper() for word in words if word) >= max(2, len(words) - 1))
            )

        texts = doc.get("texts", [])
        results = []
        seen = set()

        for i, span in enumerate(texts):
            if span.get("page_no", 999) > 2:
                continue
            if "exact name of registrant" not in span.get("text", "").lower():
                continue

            for j in range(max(0, i - 3), i):
                candidate = texts[j]
                if candidate.get("page_no") != span.get("page_no"):
                    continue
                if candidate.get("label") not in ("section_header", "text"):
                    continue
                if not _looks_company(candidate.get("text", "")):
                    continue

                idx = id(candidate)
                if idx not in seen:
                    seen.add(idx)
                    results.append(candidate)

        return results
    except Exception:
        return []
