import re


_DISPOSITION_RE = re.compile(
    r"\b(?:AFFIRM(?:ED|S)?(?:\s+IN\s+PART)?|REVERSE(?:D|S)?(?:\s+IN\s+PART)?|VACATE(?:D|S)?(?:\s+IN\s+PART)?|DISMISS(?:ED|ES|S)?(?:\s+IN\s+PART)?|DEN(?:Y|IED|IES|YING)?|REMAND(?:ED|S)?|GRANT(?:ED|S)?|HEARD\s+EN\s+BANC|REHEARD\s+EN\s+BANC)\b",
    re.IGNORECASE,
)
_STRONG_FORM_RE = re.compile(
    r"\b(?:the\s+(?:judgment|order|appeal|case|motion|petition)|we|court|panel)\b.*?"
    r"\b(?:is|are|was|were|hereby|therefore|thus)?\s*"
    r"(?:AFFIRM(?:ED|S)?|REVERSE(?:D|S)?|VACATE(?:D|S)?|DISMISS(?:ED|ES|S)?|DEN(?:Y|IED|IES|YING)?|REMAND(?:ED|S)?|GRANT(?:ED|S)?|HEARD\s+EN\s+BANC|REHEARD\s+EN\s+BANC)",
    re.IGNORECASE,
)
_PROCEDURAL_RE = re.compile(
    r"\b(?:petition|motion|appeal|case|order|stay|rehearing|en\s+banc|administrative\s+stay|panel\s+rehearing)\b.*?"
    r"\b(?:AFFIRM(?:ED|S)?|REVERSE(?:D|S)?|VACATE(?:D|S)?|DISMISS(?:ED|ES|S)?|DEN(?:Y|IED|IES|YING)?|REMAND(?:ED|S)?|GRANT(?:ED|S)?|HEARD\s+EN\s+BANC|REHEARD\s+EN\s+BANC)\b",
    re.IGNORECASE,
)
_DISSENT_CONTEXT_RE = re.compile(
    r"\b(?:dissent|concurring|respectfully\s+dissent|would\s+(?:affirm|reverse|vacate|dismiss|deny|remand))\b",
    re.IGNORECASE,
)
_ALLCAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,;.'()&/\-]*[A-Z0-9.]?$")
_CONCLUSION_RE = re.compile(r"^\s*(?:[IVX]+\.\s*)?CONCLUSION\b", re.IGNORECASE)


def rule_final_disposition(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def trim_disposition_text(text: str) -> str:
            t = norm(text)
            if not t:
                return t
            if len(t) <= 220:
                return t

            matches = list(_DISPOSITION_RE.finditer(t))
            if not matches:
                return t[:220].rstrip()

            m = matches[-1]
            start = 0
            for marker in (". ", "? ", "! ", "; "):
                pos = t.rfind(marker, 0, m.start())
                if pos != -1:
                    start = max(start, pos + 2)
            end = len(t)
            for marker in (". ", "? ", "! "):
                pos = t.find(marker, m.end())
                if pos != -1:
                    end = min(end, pos + 1)

            snippet = t[start:end].strip()
            return snippet if snippet else t[:220].rstrip()

        def is_candidate(text: str) -> bool:
            t = norm(text)
            if not t or _DISSENT_CONTEXT_RE.search(t):
                return False
            if not _DISPOSITION_RE.search(t):
                return False
            if _ALLCAPS_RE.match(t):
                return True
            if _STRONG_FORM_RE.search(t):
                return True
            if _PROCEDURAL_RE.search(t):
                return True
            if len(t) <= 120 and t.upper() == t:
                return True
            if t.lower().startswith(("the judgment", "the case", "the appeal", "the motion", "the petition", "we ", "accordingly", "for these reasons")):
                return True
            return False

        def should_attach_prev(prev_text: str, cur_text: str) -> bool:
            prev = norm(prev_text)
            cur = norm(cur_text)
            if not prev or not cur or _DISSENT_CONTEXT_RE.search(prev):
                return False
            if not _DISPOSITION_RE.search(prev):
                return False
            if prev.endswith(("and", ",", ";", ":")):
                return True
            if cur and cur[0].islower() and len(cur) <= 120:
                return True
            if _DISPOSITION_RE.search(cur) and prev.endswith("."):
                return True
            return False

        def build_span(items: list[dict], idx: int) -> dict:
            start = idx
            while start > 0 and should_attach_prev(items[start - 1].get("text") or "", items[start].get("text") or ""):
                start -= 1

            text = " ".join(
                norm(items[j].get("text") or "")
                for j in range(start, idx + 1)
                if norm(items[j].get("text") or "")
            )
            span = {"text": trim_disposition_text(text)}
            for field in ("page_no", "line_no", "paragraph_no"):
                if field in items[start]:
                    span[field] = items[start][field]
            return span

        def scan_items(items: list[dict], start_idx: int, end_idx: int) -> dict | None:
            for idx in range(end_idx - 1, start_idx - 1, -1):
                text = norm(items[idx].get("text") or "")
                if is_candidate(text):
                    return build_span(items, idx)
            return None

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if lines:
            conclusion_idx = None
            for idx in range(len(lines) - 1, -1, -1):
                if _CONCLUSION_RE.match(norm(lines[idx].get("text") or "")):
                    conclusion_idx = idx
                    break
            if conclusion_idx is not None:
                anchor_span = scan_items(lines, conclusion_idx, min(len(lines), conclusion_idx + 180))
                if anchor_span is not None:
                    return [anchor_span]

            tail_start = max(0, len(lines) - 350)
            tail_span = scan_items(lines, tail_start, len(lines))
            if tail_span is not None:
                return [tail_span]

        paragraphs = [item for item in (doc.get("paragraphs") or []) if isinstance(item, dict)]
        if paragraphs:
            tail_start = max(0, len(paragraphs) - 100)
            para_span = scan_items(paragraphs, tail_start, len(paragraphs))
            if para_span is not None:
                return [para_span]

        return []
    except Exception:
        return []
