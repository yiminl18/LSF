import re


_ANCHOR_RE = re.compile(
    r"\bthe\s+item[s]?\s+inspected\s+and\s+the\s+probable\s+violation[s]?\s+(?:is|are)"
    r"(?:\s+as\s+follows)?\s*:",
    re.IGNORECASE,
)
_START_RE = re.compile(r"^\s*1\.\s*(?:$|\S)", re.IGNORECASE)
_STOP_RE = re.compile(r"^\s*(proposed civil penalty|proposed compliance order)\b", re.IGNORECASE)
_CLOSE_RE = re.compile(r"^\s*(respectfully|sincerely|regards|enclosure|cc\b|attachments?)\b", re.IGNORECASE)


def rule_alleged_violations_body_section(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def low(text: str) -> str:
            return norm(text).lower()

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        # Find the lead-in sentence that introduces the enumerated violations.
        anchor_idx = None
        scan_limit = min(len(lines), 140)
        for i in range(scan_limit):
            for width in range(1, 7):
                if i + width > scan_limit:
                    break
                chunk = " ".join(
                    norm(lines[j].get("text") or "") for j in range(i, i + width) if norm(lines[j].get("text") or "")
                )
                if chunk and _ANCHOR_RE.search(chunk):
                    anchor_idx = i
                    break
            if anchor_idx is not None:
                break

        if anchor_idx is None:
            # Fallback: look for any early line that mentions probable violations.
            for i, item in enumerate(lines[:scan_limit]):
                text = low(item.get("text") or "")
                if "probable violation" in text and "inspect" in text:
                    anchor_idx = i
                    break

        if anchor_idx is None:
            return []

        start_idx = None
        for i in range(anchor_idx, len(lines)):
            if _START_RE.match(norm(lines[i].get("text") or "")):
                start_idx = i
                break

        if start_idx is None:
            return []

        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            text = low(lines[i].get("text") or "")
            if _STOP_RE.match(text) or _CLOSE_RE.match(text):
                end_idx = i
                break

        item_spans = []
        expected = 1
        for item in lines[start_idx:end_idx]:
            text = norm(item.get("text") or "")
            if not text:
                continue
            if re.fullmatch(r"page\s+\d+\s+of\s+\d+", text.lower()):
                continue
            if re.fullmatch(r"\d+", text):
                continue
            if text.lower().endswith(".docx"):
                continue
            if re.fullmatch(r"cpf\s+\d-\d{4}-\d{3}-nopv", text.lower()):
                continue

            match = re.match(rf"^\s*{expected}\.\s*(?:$|\S)", text)
            if not match:
                continue

            span = {"text": text}
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in item:
                    span[key] = item[key]
            item_spans.append(span)
            expected += 1

        if not item_spans:
            return []

        return item_spans
    except Exception:
        return []
