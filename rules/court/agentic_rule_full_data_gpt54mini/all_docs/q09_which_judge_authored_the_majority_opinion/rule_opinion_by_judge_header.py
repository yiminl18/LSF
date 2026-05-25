import re


_OPINION_BY_JUDGE_RE = re.compile(r"^\s*Opinion by Judge\s+(.+?)(?:[;.]?\s*)$", re.IGNORECASE)


def rule_opinion_by_judge_header(doc: dict) -> list[dict]:
    try:
        lines = [s for s in (doc.get("lines") or []) if isinstance(s, dict)]
        if not lines:
            return []

        matches: list[dict] = []
        seen = set()

        # The author of the majority opinion is usually stated near the top of the
        # document in a single header line.
        for span in lines[:180]:
            text = (span.get("text") or "").strip()
            if not text:
                continue
            if not _OPINION_BY_JUDGE_RE.match(text):
                continue

            key = (span.get("page_no"), span.get("line_no"), text)
            if key in seen:
                continue
            seen.add(key)
            out = {"text": text}
            if "page_no" in span:
                out["page_no"] = span["page_no"]
            if "line_no" in span:
                out["line_no"] = span["line_no"]
            matches.append(out)

        return matches
    except Exception:
        return []
