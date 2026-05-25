import re


_SINCERE_RE = re.compile(r"\bSincerely\b", re.IGNORECASE)
_SIGNED_RE = re.compile(r"\b(?:Digitally\s+)?signed\s+by\b", re.IGNORECASE)
_ORG_RE = re.compile(
    r"\b(?:Pipeline and Hazardous Materials Safety Administration|Office of Pipeline Safety|Enclosures:|cc:)\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(
    r"^[A-Z][A-Za-z.'-]*(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z]\.)?){0,4}$"
)


def rule_phmsa_notice_signature_block(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def items_for_doc() -> list[dict]:
            for key in ("lines", "paragraphs", "pages"):
                items = [s for s in (doc.get(key) or []) if isinstance(s, dict)]
                if items:
                    return items
            return []

        items = items_for_doc()
        if not items:
            return []

        texts = [norm(item.get("text") or "") for item in items]

        def build_span(anchor_idx: int) -> dict | None:
            start = anchor_idx
            if _SIGNED_RE.search(texts[anchor_idx]):
                for back in range(1, 9):
                    prev_idx = anchor_idx - back
                    if prev_idx >= 0 and _SINCERE_RE.search(texts[prev_idx]):
                        start = prev_idx
                        break

            end = anchor_idx
            for idx in range(anchor_idx + 1, len(items)):
                text = texts[idx]
                if not text:
                    continue
                if _ORG_RE.search(text):
                    break
                end = idx
                if idx - anchor_idx >= 10:
                    break

            if end < start:
                end = anchor_idx

            combined = "\n".join(texts[i] for i in range(start, end + 1) if texts[i])
            if not combined:
                return None
            if not (
                _SIGNED_RE.search(combined)
                or _SINCERE_RE.search(combined)
                or "director" in combined.lower()
            ):
                return None

            block_lines = [texts[i] for i in range(start, end + 1) if texts[i]]
            title_idx = None
            for rel_idx, text in enumerate(block_lines):
                if "director" in text.lower():
                    title_idx = rel_idx
                    break

            def is_name_line(text: str) -> bool:
                t = norm(text)
                if not t or len(t) > 70:
                    return False
                if any(ch.isdigit() for ch in t):
                    return False
                low = t.lower()
                if "director" in low or "sincerely" in low:
                    return False
                if _SIGNED_RE.search(t) or _ORG_RE.search(t):
                    return False
                return bool(_NAME_RE.match(t))

            if title_idx is not None:
                for rel_idx in range(title_idx - 1, -1, -1):
                    candidate = block_lines[rel_idx]
                    if is_name_line(candidate):
                        span = {"text": candidate}
                        abs_idx = start + rel_idx
                        for key in ("page_no", "line_no", "paragraph_no"):
                            if key in items[abs_idx]:
                                span[key] = items[abs_idx][key]
                        return span

            for rel_idx in range(len(block_lines) - 1, -1, -1):
                candidate = block_lines[rel_idx]
                if is_name_line(candidate):
                    span = {"text": candidate}
                    abs_idx = start + rel_idx
                    for key in ("page_no", "line_no", "paragraph_no"):
                        if key in items[abs_idx]:
                            span[key] = items[abs_idx][key]
                    return span

            span = {"text": combined}
            for key in ("page_no", "line_no", "paragraph_no"):
                if key in items[start]:
                    span[key] = items[start][key]
            return span

        best = None
        for idx in range(len(items) - 1, -1, -1):
            text = texts[idx]
            if not text:
                continue
            if not (_SIGNED_RE.search(text) or _SINCERE_RE.search(text)):
                continue
            span = build_span(idx)
            if not span:
                continue
            if best is None or len(span["text"]) > len(best["text"]):
                best = span

        return [best] if best else []
    except Exception:
        return []
