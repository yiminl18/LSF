def rule_page12_exact_name_cue_window(doc: dict) -> list[dict]:
    """Capture the cover-page window around the exact-name cue on page 1-2."""
    try:
        import re

        texts = doc.get("texts", [])
        out = []
        seen = set()
        bad_phrases = (
            "united states",
            "securities and exchange commission",
            "form 10-",
            "form 8-k",
            "form 10 q",
            "form 10 k",
            "current report",
            "annual report",
            "quarterly report",
            "transition report",
            "commission file",
            "date of report",
            "exact name of registrant",
            "state or other jurisdiction",
            "registrant's telephone",
            "address of principal",
            "title of each class",
            "trading symbol",
            "name of each exchange",
            "washington, d.c.",
            "news release",
            "summary financial results",
            "financial outlook",
            "table of contents",
            "signatures",
            "item ",
            "part ",
        )
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if (span.get("page_no") or 999) > 2:
                continue
            if "exact name of registrant as specified in its charter" not in text:
                continue

            page = span.get("page_no")
            idx = id(span)
            if idx not in seen:
                seen.add(idx)
                out.append(span)

            for j in range(max(0, i - 10), i):
                other = texts[j]
                if other.get("page_no") != page:
                    continue
                if other.get("label") not in {"text", "section_header"}:
                    continue

                other_text = (other.get("text") or "").replace("\n", " ").strip()
                other_low = other_text.lower()
                if (
                    not other_text
                    or len(other_text) > 120
                    or len(other_text.split()) > 8
                    or any(p in other_low for p in bad_phrases)
                    or re.fullmatch(r"[\W\d]+", other_text or "")
                    or sum(ch.isalpha() for ch in other_text) < 4
                ):
                    continue

                idx = id(other)
                if idx not in seen:
                    seen.add(idx)
                    out.append(other)
        return out
    except Exception:
        return []
