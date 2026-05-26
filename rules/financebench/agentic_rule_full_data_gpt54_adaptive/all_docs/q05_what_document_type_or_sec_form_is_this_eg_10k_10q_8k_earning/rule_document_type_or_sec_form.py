import re


def rule_document_type_or_sec_form(doc: dict) -> list[dict]:
    try:
        form_patterns = [
            re.compile(r"\bform\s*10\s*[- ]?\s*k\b", re.IGNORECASE),
            re.compile(r"\bform\s*10\s*[- ]?\s*q\b", re.IGNORECASE),
            re.compile(r"\bform\s*8\s*[- ]?\s*k\b", re.IGNORECASE),
        ]
        earnings_title_re = re.compile(
            r"\breports\s+(?:first|second|third|fourth)\s+quarter\b.*\bfinancial\s+results\b",
            re.IGNORECASE,
        )

        def normalize(text: str) -> str:
            text = (text or "").replace("\xa0", " ")
            return re.sub(r"\s+", " ", text).strip()

        def simplify(text: str) -> str:
            text = normalize(text).lower()
            text = re.sub(r"[^a-z0-9]+", " ", text)
            return " ".join(text.split())

        def build_span(items: list[dict], parts: list[str]) -> list[dict]:
            span = {"text": " ".join(part for part in parts if part).strip()}
            if items:
                first = items[0]
                if "page_no" in first:
                    span["page_no"] = first.get("page_no")
                if "line_no" in first:
                    span["line_no"] = first.get("line_no")
                if "paragraph_no" in first:
                    span["paragraph_no"] = first.get("paragraph_no")
            return [span] if span["text"] else []

        def collect_window(items: list[dict], start: int, max_items: int, max_chars: int) -> tuple[list[dict], list[str]]:
            chosen_items: list[dict] = []
            chosen_parts: list[str] = []
            total_chars = 0
            for item in items[start:]:
                text = normalize(item.get("text") or "")
                if not text:
                    continue
                chosen_items.append(item)
                chosen_parts.append(text)
                total_chars += len(text)
                if len(chosen_parts) >= max_items or total_chars >= max_chars:
                    break
            return chosen_items, chosen_parts

        lines = doc.get("lines") or []
        top_lines = lines[:150]
        top_clean = " ".join(simplify(item.get("text") or "") for item in top_lines[:25])

        is_earnings_release = (
            "news release" in top_clean
            or "press release" in top_clean
            or "prnewswire" in top_clean
            or bool(earnings_title_re.search(top_clean))
            or (
                "financial results" in top_clean
                and ("exhibit 99 1" in top_clean or "reported first quarter" in top_clean or "reported second quarter" in top_clean)
            )
        )

        if is_earnings_release and top_lines:
            anchor = 0
            for idx, item in enumerate(top_lines[:20]):
                simple = simplify(item.get("text") or "")
                if (
                    "exhibit 99 1" in simple
                    or "news release" in simple
                    or "press release" in simple
                    or "prnewswire" in simple
                    or "financial results" in simple
                    or earnings_title_re.search(simple)
                ):
                    anchor = idx
                    if anchor > 0 and "exhibit 99 1" in simplify(top_lines[anchor - 1].get("text") or ""):
                        anchor -= 1
                    break
            chosen_items, chosen_parts = collect_window(top_lines, anchor, max_items=6, max_chars=320)
            span = build_span(chosen_items, chosen_parts)
            if span:
                return span

        for idx, item in enumerate(top_lines):
            text = normalize(item.get("text") or "")
            if not text:
                continue
            if not any(pattern.search(text) for pattern in form_patterns):
                continue

            chosen_items, chosen_parts = collect_window(top_lines, idx, max_items=5, max_chars=320)
            joined = " ".join(simplify(part) for part in chosen_parts)

            if "form 10 k" in joined and "annual report" not in joined:
                for extra in top_lines[idx + len(chosen_parts): idx + 12]:
                    extra_text = normalize(extra.get("text") or "")
                    if not extra_text:
                        continue
                    chosen_items.append(extra)
                    chosen_parts.append(extra_text)
                    if "annual report" in simplify(" ".join(chosen_parts)):
                        break
            elif "form 10 q" in joined and "quarterly report" not in joined:
                for extra in top_lines[idx + len(chosen_parts): idx + 12]:
                    extra_text = normalize(extra.get("text") or "")
                    if not extra_text:
                        continue
                    chosen_items.append(extra)
                    chosen_parts.append(extra_text)
                    if "quarterly report" in simplify(" ".join(chosen_parts)):
                        break
            elif "form 8 k" in joined and "current report" not in joined:
                for extra in top_lines[idx + len(chosen_parts): idx + 12]:
                    extra_text = normalize(extra.get("text") or "")
                    if not extra_text:
                        continue
                    chosen_items.append(extra)
                    chosen_parts.append(extra_text)
                    if "current report" in simplify(" ".join(chosen_parts)):
                        break

            span = build_span(chosen_items, chosen_parts)
            if span:
                return span

        paragraphs = doc.get("paragraphs") or []
        for item in paragraphs[:20]:
            text = normalize(item.get("text") or "")
            simple = simplify(text)
            if not text:
                continue
            if any(pattern.search(text) for pattern in form_patterns):
                return build_span([item], [text])
            if (
                "news release" in simple
                or "press release" in simple
                or "prnewswire" in simple
                or earnings_title_re.search(simple)
            ):
                return build_span([item], [text])

        text = normalize(doc.get("text") or "")
        if text:
            head = text[:4000]
            if any(pattern.search(head) for pattern in form_patterns):
                snippet_match = re.search(
                    r"(?is)(form\s*10\s*[- ]?\s*[kq]|form\s*8\s*[- ]?\s*k.{0,250}?(?:annual report|quarterly report|current report)?)",
                    head,
                )
                if snippet_match:
                    return [{"text": normalize(snippet_match.group(0))}]
            if re.search(r"(?is)(news release|press release|prnewswire|reports\s+(?:first|second|third|fourth)\s+quarter.{0,120}financial results)", head):
                return [{"text": normalize(head[:300])}]

        return []
    except Exception:
        return []
