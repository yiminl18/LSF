import re


_REGION = r"(?:Central|Eastern|Southern|Southwest|Western)"
_DIRECTOR_REGION_RE = re.compile(
    rf"\b(?:Acting\s+)?Director(?:\s+of)?[, ]+(?:the\s+)?(?:PHMSA[, ]+)?{_REGION}\s+Region\b",
    re.IGNORECASE,
)
_REGION_DIRECTOR_RE = re.compile(rf"\b(?:Acting\s+)?{_REGION}\s+Region\s+Director\b", re.IGNORECASE)
_PHMSA_REGION_RE = re.compile(rf"\bPHMSA[, ]+{_REGION}\s+Region\b", re.IGNORECASE)
_OFFICE_RE = re.compile(
    r"\b(?:Office of Pipeline Safety|Pipeline and Hazardous Materials Safety Administration)\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(
    r"^[A-Z][A-Za-z.'-]*(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z]\.)?){0,4}$"
)


def rule_phmsa_region_office_signature(doc: dict) -> list[dict]:
    try:
        lines = [s for s in (doc.get("lines") or []) if isinstance(s, dict)]
        if not lines:
            return []

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def meta(span: dict) -> dict:
            out = {"text": norm(span.get("text") or "")}
            if "page_no" in span:
                out["page_no"] = span["page_no"]
            if "line_no" in span:
                out["line_no"] = span["line_no"]
            return out

        def is_name_line(text: str) -> bool:
            t = norm(text)
            if not t or len(t) > 70:
                return False
            if any(ch.isdigit() for ch in t):
                return False
            if _OFFICE_RE.search(t) or "director" in t.lower():
                return False
            return bool(_NAME_RE.match(t))

        def score_candidate(index: int, text: str) -> int:
            t = norm(text)
            low = t.lower()
            score = 0
            if _DIRECTOR_REGION_RE.search(t):
                score += 5
            if _REGION_DIRECTOR_RE.search(t):
                score += 4
            if _PHMSA_REGION_RE.search(t):
                score += 3
            if _OFFICE_RE.search(t):
                score += 3
            if "director" in low and _OFFICE_RE.search(t):
                score += 2
            if index >= len(lines) // 2:
                score += 1
            prev_text = norm(lines[index - 1].get("text") or "") if index > 0 else ""
            next_text = norm(lines[index + 1].get("text") or "") if index + 1 < len(lines) else ""
            if is_name_line(prev_text):
                score += 2
            if _OFFICE_RE.search(next_text):
                score += 2
            return score

        best = None
        best_score = 0
        for i, span in enumerate(lines):
            text = norm(span.get("text") or "")
            if not text:
                continue
            if not (
                _DIRECTOR_REGION_RE.search(text)
                or _REGION_DIRECTOR_RE.search(text)
                or _PHMSA_REGION_RE.search(text)
            ):
                continue

            score = score_candidate(i, text)
            if score < 5:
                continue

            start = i
            end = i
            if i > 0 and is_name_line(lines[i - 1].get("text") or ""):
                start = i - 1
            if i + 1 < len(lines) and _OFFICE_RE.search(norm(lines[i + 1].get("text") or "")):
                end = i + 1

            combined = " ".join(norm(lines[j].get("text") or "") for j in range(start, end + 1))
            candidate = {
                "text": combined,
                **{k: lines[start][k] for k in ("page_no", "line_no") if k in lines[start]},
            }

            if score > best_score or (score == best_score and best is not None and len(candidate["text"]) > len(best["text"])):
                best = candidate
                best_score = score

        return [best] if best else []
    except Exception:
        return []
