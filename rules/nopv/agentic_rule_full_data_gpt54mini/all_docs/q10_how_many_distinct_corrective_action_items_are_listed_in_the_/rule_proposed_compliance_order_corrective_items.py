import re


_HEADING_RE = re.compile(r"\bPROPOSED\s+COMPLIANCE\s+ORDER\b", re.IGNORECASE)
_INTRO_RE = re.compile(r"\bpursuant\b.*\bproposes?\s+to\s+issue\b", re.IGNORECASE)
_REMEDIAL_RE = re.compile(r"\bincorporating\s+the\s+following\s+remedial\s+requirements\b", re.IGNORECASE)
_REQUEST_RE = re.compile(r"\bit\s+is\s+requested(?:\s*\(not\s+mandated\))?", re.IGNORECASE)
_STOP_RE = re.compile(
    r"^(response\s+to\s+this\s+notice|response\s+options\s+for\s+pipeline\s+operators|"
    r"enclosures?:|sincerely|cc:|cc\b)",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"^(page\s+\d+(?:\s+of\s+\d+)?)$", re.IGNORECASE)
_LETTER_LABEL_RE = re.compile(r"^\s*([A-Z])(?:\.(?![A-Z0-9]))?(?:\s+|$)(.*)$")
_NUMBER_LABEL_RE = re.compile(r"^\s*(\d+)\.(?!\d)(?:\s+|$)(.*)$")


def rule_proposed_compliance_order_corrective_items(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def low(text: str) -> str:
            return norm(text).lower()

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        def is_noise(text: str) -> bool:
            text = norm(text)
            if not text:
                return True
            if re.fullmatch(r"\d+", text):
                return True
            if _PAGE_RE.fullmatch(text):
                return True
            return False

        def looks_like_actual_order(start_idx: int) -> bool:
            nonnoise_texts = []
            for j in range(start_idx + 1, min(len(lines), start_idx + 10)):
                text = norm(lines[j].get("text") or "")
                if is_noise(text):
                    continue
                nonnoise_texts.append(text)
                if len(nonnoise_texts) >= 4:
                    break
            if not nonnoise_texts:
                return False
            first = nonnoise_texts[0].lower()
            if not first.startswith("pursuant"):
                return False
            joined = " ".join(nonnoise_texts[:4]).lower()
            return "proposes to issue" in joined and _REMEDIAL_RE.search(joined) is not None

        def infer_top_level_scheme(start_idx: int) -> str | None:
            for j in range(start_idx + 1, min(len(lines), start_idx + 80)):
                text = norm(lines[j].get("text") or "")
                if not text or _STOP_RE.match(text):
                    return None
                if _REQUEST_RE.search(text):
                    # Some orders place the request item immediately after the substantive items.
                    # Do not infer the scheme from the request item itself.
                    continue
                if _LETTER_LABEL_RE.match(text):
                    return "letter"
                if _NUMBER_LABEL_RE.match(text):
                    return "number"
            return None

        def parse_numeric_path(text: str) -> list[str] | None:
            match = re.match(r"^\s*((?:\d+)(?:\.\d+)*)(?:\.(?!\d))?(?:\s+|$)", text)
            if not match:
                return None
            return match.group(1).split(".")

        def parse_label(text: str) -> tuple[str | None, list[str] | None]:
            text = norm(text)
            match = _LETTER_LABEL_RE.match(text)
            if match:
                return "letter", [match.group(1)]
            path = parse_numeric_path(text)
            if path:
                return "number", path
            return None, None

        def is_top_level_label(text: str, scheme: str) -> bool:
            text = norm(text)
            if scheme == "letter":
                return bool(_LETTER_LABEL_RE.match(text))
            return bool(_NUMBER_LABEL_RE.match(text))

        def build_span(start_idx: int, end_idx: int) -> dict | None:
            start_meta = lines[start_idx]
            chunk = []
            for j in range(start_idx, end_idx):
                text = norm(lines[j].get("text") or "")
                if not text or is_noise(text):
                    continue
                chunk.append(text.strip())
                if len(chunk) >= 2:
                    break
            text = "\n".join(chunk).strip()
            if not text or _REQUEST_RE.search(text):
                return None
            span = {"text": text}
            for key in ("page_no", "line_no"):
                if key in start_meta:
                    span[key] = start_meta[key]
            return span

        def extract_leaf_spans(start_idx: int, end_idx: int) -> list[dict]:
            if start_idx >= end_idx:
                return []

            first_text = norm(lines[start_idx].get("text") or "")
            kind, path = parse_label(first_text)
            if kind is None:
                span = build_span(start_idx, end_idx)
                return [span] if span else []

            child_starts: list[tuple[int, list[str]]] = []
            if kind == "letter":
                # Lettered blocks usually contain bare numeric sub-items.
                raw_children: list[tuple[int, int]] = []
                for j in range(start_idx + 1, end_idx):
                    text = norm(lines[j].get("text") or "")
                    if not text:
                        continue
                    if _STOP_RE.match(text):
                        break
                    child_kind, child_path = parse_label(text)
                    if child_kind == "number" and child_path and len(child_path) == 1:
                        try:
                            raw_children.append((j, int(child_path[0])))
                        except Exception:
                            continue

                if raw_children:
                    raw_children.sort(key=lambda item: item[0])
                    expected = raw_children[0][1]
                    for j, num in raw_children:
                        if num < expected:
                            continue
                        if num > expected:
                            # Ignore outlier citations or page artifacts (for example, "195.")
                            # and keep scanning for the next true bullet in the sequence.
                            continue
                        child_starts.append((j, [str(num)]))
                        expected += 1
            else:
                # Numeric blocks use a dotted hierarchy such as 1.1, 1.1.1, etc.
                assert path is not None
                for j in range(start_idx + 1, end_idx):
                    text = norm(lines[j].get("text") or "")
                    if not text:
                        continue
                    if _STOP_RE.match(text):
                        break
                    child_kind, child_path = parse_label(text)
                    if (
                        child_kind == "number"
                        and child_path
                        and len(child_path) == len(path) + 1
                        and child_path[: len(path)] == path
                    ):
                        child_starts.append((j, child_path))

            if not child_starts:
                span = build_span(start_idx, end_idx)
                return [span] if span else []

            spans: list[dict] = []
            for idx, (child_start, _child_path) in enumerate(child_starts):
                child_end = child_starts[idx + 1][0] if idx + 1 < len(child_starts) else end_idx
                child_spans = extract_leaf_spans(child_start, child_end)
                if child_spans:
                    spans.extend(child_spans)
                else:
                    span = build_span(child_start, child_end)
                    if span:
                        spans.append(span)
            return spans

        heading_idxs = [
            i
            for i, item in enumerate(lines)
            if _HEADING_RE.search(norm(item.get("text") or ""))
        ]
        if not heading_idxs:
            return []

        chosen_idx = None
        for idx in heading_idxs:
            if looks_like_actual_order(idx):
                chosen_idx = idx
                break
        if chosen_idx is None:
            # Fallback: pick the last heading, which is usually the attached order section.
            chosen_idx = heading_idxs[-1]

        scheme = infer_top_level_scheme(chosen_idx)
        if scheme is None:
            return []

        top_starts: list[int] = []
        i = chosen_idx + 1
        while i < len(lines):
            text = norm(lines[i].get("text") or "")
            if not text:
                i += 1
                continue
            if _STOP_RE.match(text):
                break
            if not is_top_level_label(text, scheme):
                i += 1
                continue

            lookahead_idx = i
            while lookahead_idx < len(lines) and not norm(lines[lookahead_idx].get("text") or ""):
                lookahead_idx += 1
            if lookahead_idx < len(lines) and _REQUEST_RE.search(norm(lines[lookahead_idx].get("text") or "")):
                break

            top_starts.append(i)
            i += 1

        spans = []
        for idx, start_idx in enumerate(top_starts):
            end_idx = top_starts[idx + 1] if idx + 1 < len(top_starts) else len(lines)
            spans.extend(extract_leaf_spans(start_idx, end_idx))

        # Collapse the extracted leaf items into a single derived count. The
        # question is explicitly asking for the number of distinct corrective
        # action items, so returning the computed count is the most reliable
        # way to preserve the intended answer.
        seen = set()
        first_meta = None
        for span in spans:
            key = span["text"]
            if key in seen:
                continue
            seen.add(key)
            if first_meta is None:
                first_meta = span

        if not seen:
            return []

        result = {"text": str(len(seen))}
        if first_meta is not None:
            for key in ("page_no", "line_no"):
                if key in first_meta:
                    result[key] = first_meta[key]
        return [result]
    except Exception:
        return []
