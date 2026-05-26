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
_TOP_MARKER_RE = re.compile(r"^\s*(\d+)\.(?!\d)\s*(.*)$")
_CITATION_RE = re.compile(
    r"^\s*(?:§+\s*\d{3}\.\d+\b|\d{3}\.\d+\b|49\s+C\.?\s*F\.?\s*R\.?\b|49\s+U\.?\s*S\.?\s*C\.?\b|part\s+\d{3}\b)",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"^\s*(?:proposed\s+civil\s+penalty|proposed\s+compliance\s+order|warning\s+items?|"
    r"response\s+options|notice\s+response\s+options|sincerely|respectfully|regards|"
    r"enclosures?|attachments?|cc\b)\b",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)
_CPF_RE = re.compile(r"^\s*cpf\s+\d-\d{4}-\d{3,4}-nopv\s*$", re.IGNORECASE)
_DOCX_RE = re.compile(r"\.docx\s*$", re.IGNORECASE)
_FOOTNOTE_DIGIT_RE = re.compile(r"^\s*\d+\s*$")
_ITEM_REF_RE = re.compile(
    r"\b(?:with\s+respect\s+to|in\s+regard\s+to|regarding|pertaining\s+to)\s+item[s]?\b|\bitem\s+number\b",
    re.IGNORECASE,
)


def rule_alleged_violations_numbered_items(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\ufeff", "").replace("\f", " ")).strip()

        def is_artifact(text: str) -> bool:
            s = norm(text)
            if not s:
                return True
            return bool(
                _PAGE_RE.match(s)
                or _CPF_RE.match(s)
                or _DOCX_RE.search(s)
                or s.lower() == "this page intentionally left blank"
            )

        def iter_nonblank_indices(start: int, stop: int) -> list[int]:
            return [
                idx
                for idx in range(start, stop)
                if not is_artifact(lines[idx].get("text") or "")
            ]

        def add_span(spans: list[dict], seen_texts: set[str], idx: int, text: str) -> None:
            text = norm(text)
            if not text or text in seen_texts:
                return
            span = {"text": text}
            for key in ("page_no", "line_no", "paragraph_no"):
                value = lines[idx].get(key)
                if value is not None:
                    span[key] = value
            spans.append(span)
            seen_texts.add(text)

        def has_failure_narrative(start_idx: int, stop_idx: int) -> bool:
            probe_indices = iter_nonblank_indices(start_idx + 1, min(stop_idx, start_idx + 35))
            for probe_idx in probe_indices:
                probe_text = norm(lines[probe_idx].get("text") or "").lower()
                if _STOP_RE.match(probe_text):
                    break
                if _CITATION_RE.match(probe_text):
                    break
                if " failed " in f" {probe_text} " or probe_text.startswith("failed ") or "did not comply" in probe_text:
                    return True
            return False

        def extract_item_ref_numbers(text: str) -> set[int]:
            lowered = norm(text)
            match = re.search(
                r"\bitems?\b(.*?)(?:\bpursuant\b|\bof\s+the\s+notice\b|[.;:]|$)",
                lowered,
                re.IGNORECASE,
            )
            if not match:
                return set()
            return {int(value) for value in re.findall(r"\b\d+\b", match.group(1))}

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        scan_limit = min(len(lines), 320)
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
            for i, item in enumerate(lines[:scan_limit]):
                text = norm(item.get("text") or "").lower()
                if "probable violation" in text and "inspect" in text:
                    anchor_idx = i
                    break

        if anchor_idx is None:
            anchor_idx = 0

        numbered_headings = []
        numbered_item_nos = []
        for i in range(anchor_idx, len(lines)):
            raw = lines[i].get("text") or ""
            text = norm(raw)
            marker_match = _TOP_MARKER_RE.match(text)
            if not marker_match:
                continue

            item_no = int(marker_match.group(1))
            inline_rest = marker_match.group(2).strip()
            citation_text = inline_rest if _CITATION_RE.match(inline_rest) else ""

            probe_indices = iter_nonblank_indices(i + 1, min(len(lines), i + 15))
            if not citation_text:
                for probe_idx in probe_indices:
                    probe_text = norm(lines[probe_idx].get("text") or "")
                    if _TOP_MARKER_RE.match(probe_text) or _STOP_RE.match(probe_text):
                        break
                    if _FOOTNOTE_DIGIT_RE.match(probe_text):
                        continue
                    if _CITATION_RE.match(probe_text):
                        citation_text = probe_text
                        break

            if not citation_text:
                continue

            numbered_item_nos.append(item_no)
            numbered_headings.append((i, item_no, f"{item_no}. {citation_text}"))

        unnumbered_headings = []
        if not numbered_headings:
            for i in range(anchor_idx, len(lines)):
                text = norm(lines[i].get("text") or "")
                if not _CITATION_RE.match(text):
                    continue
                prev_indices = iter_nonblank_indices(max(0, i - 2), i)
                if prev_indices and _TOP_MARKER_RE.match(norm(lines[prev_indices[-1]].get("text") or "")):
                    continue
                if has_failure_narrative(i, len(lines)):
                    unnumbered_headings.append((i, text))

        need_item_refs = False
        if not numbered_item_nos:
            need_item_refs = True
        else:
            highest = max(numbered_item_nos)
            if min(numbered_item_nos) > 1 or len(set(numbered_item_nos)) < highest:
                need_item_refs = True

        item_ref_numbers = set()
        item_ref_texts = []
        if need_item_refs:
            for i in range(anchor_idx, len(lines)):
                text = norm(lines[i].get("text") or "")
                if not text or not _ITEM_REF_RE.search(text):
                    continue
                nums = extract_item_ref_numbers(text)
                if nums:
                    item_ref_numbers.update(nums)
                    item_ref_texts.append((i, text, nums))
                    continue
                if "item number" in text.lower():
                    trailing_digits = []
                    for probe_idx in iter_nonblank_indices(i + 1, min(len(lines), i + 8)):
                        probe_text = norm(lines[probe_idx].get("text") or "")
                        if _FOOTNOTE_DIGIT_RE.match(probe_text):
                            trailing_digits.append(probe_text)
                            continue
                        if trailing_digits:
                            break
                    if trailing_digits:
                        nums = {int(d) for d in trailing_digits}
                        item_ref_numbers.update(nums)
                        item_ref_texts.append((i, f"{text}: {' '.join(trailing_digits)}", nums))

        if numbered_headings:
            numbered_set = set(numbered_item_nos)
            all_numbers = sorted(numbered_set | item_ref_numbers)
            evidence_lines = []
            if all_numbers:
                evidence_lines.append(
                    "The alleged violation items enumerated in the Notice are numbered: "
                    + ", ".join(str(n) for n in all_numbers)
                    + f". This indicates {max(all_numbers)} distinct alleged violations."
                )
            for _, _, heading_text in numbered_headings:
                evidence_lines.append(heading_text)
            if item_ref_numbers and (not numbered_set or numbered_set != set(range(1, max(all_numbers) + 1))):
                added_ref = 0
                for _, ref_text, nums in item_ref_texts:
                    if nums - numbered_set:
                        evidence_lines.append(ref_text)
                        added_ref += 1
                    if added_ref >= 3:
                        break

            start_idx = min(
                [item[0] for item in numbered_headings] + [item[0] for item in item_ref_texts] or [anchor_idx]
            )
            span = {"text": "\n".join(evidence_lines)}
            for key in ("page_no", "line_no", "paragraph_no"):
                value = lines[start_idx].get(key)
                if value is not None:
                    span[key] = value
            return [span]

        if unnumbered_headings:
            evidence_lines = [
                f"The Notice body contains {len(unnumbered_headings)} distinct alleged violation citation headings."
            ]
            evidence_lines.extend(text for _, text in unnumbered_headings)
            start_idx = unnumbered_headings[0][0]
            span = {"text": "\n".join(evidence_lines)}
            for key in ("page_no", "line_no", "paragraph_no"):
                value = lines[start_idx].get(key)
                if value is not None:
                    span[key] = value
            return [span]

        if item_ref_numbers:
            all_numbers = sorted(item_ref_numbers)
            evidence_lines = [
                "The alleged violation items referenced in the Notice are numbered: "
                + ", ".join(str(n) for n in all_numbers)
                + f". This indicates {max(all_numbers)} distinct alleged violations."
            ]
            for _, ref_text, _ in item_ref_texts[:3]:
                evidence_lines.append(ref_text)
            start_idx = item_ref_texts[0][0]
            span = {"text": "\n".join(evidence_lines)}
            for key in ("page_no", "line_no", "paragraph_no"):
                value = lines[start_idx].get(key)
                if value is not None:
                    span[key] = value
            return [span]

        block_lines = []
        first_idx = None
        for i in range(anchor_idx, len(lines)):
            text = norm(lines[i].get("text") or "")
            if not text or is_artifact(text) or _STOP_RE.match(text):
                continue
            if first_idx is None:
                first_idx = i
            block_lines.append(text)

        if not block_lines or first_idx is None:
            return []

        span = {"text": "\n".join(block_lines)}
        for key in ("page_no", "line_no", "paragraph_no"):
            value = lines[first_idx].get(key)
            if value is not None:
                span[key] = value
        return [span]
    except Exception:
        return []
