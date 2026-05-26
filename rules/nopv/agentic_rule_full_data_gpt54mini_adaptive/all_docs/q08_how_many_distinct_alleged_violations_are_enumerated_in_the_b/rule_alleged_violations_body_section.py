import re


_INTRO_RE = re.compile(
    r"\b(?:the\s+)?item[s]?\s+inspected\s+and\s+the\s+probable\s+violation[s]?\s+(?:is|are)"
    r"(?:\s+as\s+follows)?\s*:?",
    re.IGNORECASE,
)
_ALT_INTRO_RE = re.compile(
    r"\bthe\s+probable\s+violation[s]?\s+(?:is|are)(?:\s+as\s+follows)?\s*:?",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"^\s*(\d+)\.(?!\d)\s*(.*)$")
_CITATION_RE = re.compile(r"^\s*(§|49\s+U\.S\.C\.|49\s+CFR\b|part\b)", re.IGNORECASE)
_STOP_RE = re.compile(
    r"^\s*(proposed\s+civil\s+penalty|proposed\s+compliance\s+order|"
    r"respectfully|sincerely|regards|enclosure|enclosures|cc\b|attachments?)\b",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)


def rule_alleged_violations_body_section(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def low(text: str) -> str:
            return norm(text).lower()

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        scan_limit = min(len(lines), 260)
        anchor_idx = None
        for i in range(scan_limit):
            for width in range(1, 7):
                if i + width > scan_limit:
                    break
                chunk = " ".join(
                    norm(lines[j].get("text") or "")
                    for j in range(i, i + width)
                    if norm(lines[j].get("text") or "")
                )
                if not chunk:
                    continue
                if _INTRO_RE.search(chunk) or _ALT_INTRO_RE.search(chunk):
                    anchor_idx = i
                    break
            if anchor_idx is not None:
                break

        if anchor_idx is None:
            # Fallback: many notices keep the intro phrase on a single line.
            for i, item in enumerate(lines[:scan_limit]):
                text = low(item.get("text") or "")
                if "probable violation" in text and "inspect" in text:
                    anchor_idx = i
                    break

        if anchor_idx is None:
            for i, item in enumerate(lines[:scan_limit]):
                if "NOTICE OF PROBABLE VIOLATION" in norm(item.get("text") or ""):
                    anchor_idx = i
                    break

        if anchor_idx is None:
            return []

        start_idx = anchor_idx
        for i in range(anchor_idx, len(lines)):
            if _NUM_RE.match(norm(lines[i].get("text") or "")):
                start_idx = i
                break

        end_idx = len(lines)
        for i in range(start_idx, len(lines)):
            text = norm(lines[i].get("text") or "")
            if not text:
                continue
            if _STOP_RE.match(text):
                end_idx = i
                break

        block_lines = []
        for i in range(start_idx, end_idx):
            text = norm(lines[i].get("text") or "")
            if not text:
                continue
            if _PAGE_RE.match(text):
                continue
            if re.fullmatch(r"cpf\s+\d-\d{4}-\d{3}-nopv", text.lower()):
                continue
            if text.lower().endswith(".docx"):
                continue
            block_lines.append(text)

        if not block_lines:
            return []

        span = {"text": "\n".join(block_lines)}
        for key in ("page_no", "line_no", "paragraph_no"):
            if key in lines[start_idx]:
                span[key] = lines[start_idx][key]
        return [span]
    except Exception:
        return []
