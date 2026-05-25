import re


_COUNSEL_HEAD_RE = re.compile(
    r"(?:^|\n)\s*(?:COUNSEL|ATTORNEYS?)\s*(?:\n|$)",
    re.IGNORECASE,
)


def _as_int(value):
    try:
        return int(value)
    except Exception:
        return value


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _iter_items(doc: dict):
    lines = doc.get("lines") or []
    if lines:
        items = [item for item in lines if isinstance(item, dict)]
        items = sorted(
            items,
            key=lambda x: (_as_int(x.get("page_no", 0)), _as_int(x.get("line_no", 0))),
        )
        for item in items:
            yield item
        return

    paragraphs = doc.get("paragraphs") or []
    if paragraphs:
        items = [item for item in paragraphs if isinstance(item, dict)]
        items = sorted(
            items,
            key=lambda x: (
                _as_int(x.get("page_no", 0)),
                _as_int(x.get("paragraph_no", 0)),
            ),
        )
        for item in items:
            yield item
        return

    pages = doc.get("pages") or []
    if pages:
        items = [item for item in pages if isinstance(item, dict)]
        items = sorted(items, key=lambda x: _as_int(x.get("page_no", 0)))
        for page in items:
            page_no = _as_int(page.get("page_no", 0))
            for line_no, raw_line in enumerate((page.get("text") or "").splitlines(), start=1):
                yield {"page_no": page_no, "line_no": line_no, "text": raw_line}


def _first_attorney_from_block(block_text: str) -> str:
    text = _normalize(block_text)
    if not text:
        return ""

    # Trim off affiliations and later counsel names, leaving the first person
    # in the block.
    text = text.split(";", 1)[0].strip()
    text = re.split(r"\s+\band\b\s+", text, 1, flags=re.IGNORECASE)[0].strip()
    text = text.split(",", 1)[0].strip()
    text = re.sub(
        r"\s*\((?:argued|lead|co[- ]?counsel|on brief|oral argument)\)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text.strip(" ,;:")


def rule_first_listed_attorney_appellee(doc: dict) -> list[dict]:
    try:
        items = list(_iter_items(doc))
        if not items:
            return []

        in_counsel = False
        current_block = []

        for item in items:
            raw_text = item.get("text") or ""
            text = _normalize(raw_text)
            if not text:
                continue

            if not in_counsel:
                if _COUNSEL_HEAD_RE.search(raw_text):
                    in_counsel = True
                continue

            upper = text.upper()
            if upper.startswith("OPINION") or upper.startswith("SUMMARY"):
                break

            lower = text.lower()
            if any(
                keyword in lower
                for keyword in (
                    "appellant",
                    "appellee",
                    "amicus",
                    "amici",
                    "curiae",
                    "petitioner",
                    "respondent",
                )
            ):
                if lower.startswith("for "):
                    block_items = current_block[:]
                else:
                    current_block.append(item)
                    block_items = current_block[:]

                if "appellee" in lower:
                    block_text = "\n".join((x.get("text") or "") for x in block_items)
                    first = _first_attorney_from_block(block_text)
                    if first:
                        span = {"text": first}
                        for key in ("page_no", "line_no", "paragraph_no"):
                            if key in block_items[0]:
                                span[key] = block_items[0][key]
                        return [span]

                current_block = []
                continue

            current_block.append(item)

        return []
    except Exception:
        return []
