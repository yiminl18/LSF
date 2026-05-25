def rule_page1_incorporation_ein_line_window(doc: dict) -> list[dict]:
    """Retrieve page-1 cover-page lines around incorporation state and EIN labels."""
    try:
        import re

        lines = [item for item in (doc.get("lines") or []) if item.get("page_no") == 1]
        if not lines:
            return []

        def compact(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

        markers = (
            "state or other jurisdiction of incorporation or organization",
            "state or other jurisdiction of incorporation",
            "state of incorporation",
            "irs employer identification no",
            "i r s employer identification no",
            "employer identification no",
        )

        out: list[dict] = []
        seen: set[tuple[object, object, object]] = set()

        def add(idx: int) -> None:
            if 0 <= idx < len(lines):
                span = lines[idx]
                key = (span.get("page_no"), span.get("line_no"), span.get("text"))
                if key in seen:
                    return
                text = (span.get("text") or "").strip()
                if not text:
                    return
                seen.add(key)
                out.append(span)

        # Labels are frequently split across blank lines, so scan short windows and
        # collapse whitespace before matching.
        for i in range(len(lines)):
            for width in (1, 2, 3, 4, 5, 6, 7, 8):
                if i + width > len(lines):
                    continue
                raw_window = " ".join((lines[j].get("text") or "").lower() for j in range(i, i + width) if (lines[j].get("text") or "").strip())
                if not raw_window.strip():
                    continue
                window = compact(raw_window)
                ein_hit = bool(re.search(r"\b\d{2}-\d{7}\b", raw_window) or re.search(r"\b\d{2}\s+\d{7}\b", window))
                state_hit = (
                    "state or other jurisdiction" in window
                    or "state of incorporation" in window
                    or "jurisdiction of incorporation" in window
                )
                label_hit = (
                    "incorporation or organization" in window
                    or "incorporation" in window
                    or "irs employer identification" in window
                    or "i r s employer identification" in window
                    or "employer identification no" in window
                )
                hit = ein_hit and (state_hit or label_hit or any(marker in window for marker in markers))
                if hit:
                    for j in range(max(0, i - 4), min(len(lines), i + width + 4)):
                        add(j)
                    break

        # Fallback: if the page text clearly contains the labels but the line scan did not
        # isolate them, return the top of page 1 to preserve the cover-page answer block.
        if not out:
            page_text = ""
            for page in (doc.get("pages") or []):
                if page.get("page_no") == 1:
                    page_text = compact(page.get("text") or "")
                    break
            if page_text and (
                any(marker in page_text for marker in markers)
                or re.search(r"\b\d{2}\s*-\s*\d{7}\b", page_text)
            ):
                for j in range(min(len(lines), 30)):
                    add(j)

        return out
    except Exception:
        return []
