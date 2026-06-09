def rule_page1_path_terminal_before_exact_name_label(doc: dict) -> list[dict]:
    """Match page-1 spans before the exact-name label whose text matches the breadcrumb terminal."""
    try:
        import re

        def normalize(text: str) -> str:
            text = (text or "").lower().replace("&", " and ")
            text = re.sub(r"[^a-z0-9]+", " ", text)
            return " ".join(text.split())

        def looks_name(text: str) -> bool:
            text = (text or "").strip()
            low = normalize(text)
            if not text or len(text) > 120 or not any(ch.isalpha() for ch in text):
                return False
            blocked_prefixes = (
                "exact name of registrant",
                "commission file",
                "state or other jurisdiction",
                "irs employer identification",
                "address of principal executive offices",
                "registrant s telephone number",
                "telephone number",
                "date of report",
                "for the transition period",
                "for the fiscal year ended",
                "form 10 k",
                "form 10 q",
                "form 8 k",
                "current report",
                "quarterly report",
                "annual report",
                "united states securities and exchange commission",
                "securities and exchange commission",
                "washington d c 20549",
                "securities registered pursuant",
                "name of each exchange",
                "title of each class",
                "trading symbol",
                "common stock",
                "new york stock exchange",
                "table of contents",
                "signatures",
            )
            if low in {"or", "x", "o"}:
                return False
            if any(low.startswith(prefix) for prefix in blocked_prefixes):
                return False
            words = text.split()
            if len(words) > 10:
                return False
            if len(words) == 1 and not re.match(r"^\d+[a-z]+$", text, re.IGNORECASE):
                return False
            return True

        label_re = re.compile(
            r"exact\s+name\s+of\s+registrant\s+as\s+specified\s+in\s+(?:its\s+)?charter",
            re.IGNORECASE,
        )

        def next_label_matches(texts: list[dict], idx: int) -> bool:
            parts: list[str] = []
            for j in range(idx + 1, min(len(texts), idx + 3)):
                parts.append(texts[j].get("text") or "")
                if label_re.search(" ".join(parts)):
                    return True
            return False

        hits: list[dict] = []
        texts = doc.get("texts", [])
        for idx, span in enumerate(texts[:-1]):
            path_text = ((span.get("structure") or {}).get("path_text") or "").split("|")[-1].strip()
            if (
                span.get("page_no") == 1
                and span.get("label") in {"section_header", "text"}
                and looks_name(span.get("text") or "")
                and normalize(span.get("text") or "") == normalize(path_text)
                and next_label_matches(texts, idx)
            ):
                hits.append(span)
        return hits
    except Exception:
        return []
