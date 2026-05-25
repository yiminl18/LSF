import re


def rule_operator_regulated_part(doc: dict) -> list[dict]:
    try:
        parts_re = re.compile(r"\b(?:49\s*C\.?\s*F\.?\s*R\.?\s*)?(?:Part|part)\s*(192|193|195|199)\b", re.I)
        anchor_re = re.compile(
            r"(subject to|covered by|requirements? of|compliance with|in accordance with|conforms? to|"
            r"required by this part|should have been subject|not subject to|codified in|regulated by)",
            re.I,
        )

        def iter_items():
            for key in ("paragraphs", "lines", "pages"):
                for item in doc.get(key, []) or []:
                    text = (item.get("text") or "").strip()
                    if text:
                        yield key, item, text

        best_candidate = None

        for source, item, text in iter_items():
            if not anchor_re.search(text):
                continue
            for match in parts_re.finditer(text):
                lowered = text.lower()
                specificity = 0
                for phrase in (
                    "subject to",
                    "covered by",
                    "requirements of",
                    "compliance with",
                    "in accordance with",
                    "conforms to",
                    "required by this part",
                    "should have been subject",
                    "not subject to",
                    "codified in",
                    "regulated by",
                ):
                    if phrase in lowered:
                        specificity += 1
                if source == "lines":
                    source_rank = 0
                elif source == "paragraphs":
                    source_rank = 1
                else:
                    source_rank = 2
                word_count = max(1, len(text.split()))

                span = {"text": text}
                for field in ("page_no", "paragraph_no", "line_no"):
                    if field in item:
                        span[field] = item[field]
                span["_source"] = source
                candidate = (word_count, -specificity, source_rank, span.get("page_no", 0), span.get("paragraph_no", 0), span.get("line_no", 0), span)
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate

        if best_candidate is None:
            # Fallback: capture the first nearby sentence-like snippet with a Part citation.
            full_text = doc.get("text") or ""
            for match in parts_re.finditer(full_text):
                start = max(0, match.start() - 140)
                end = min(len(full_text), match.end() + 180)
                snippet = full_text[start:end].strip()
                if anchor_re.search(snippet):
                    return [{"text": snippet}]
            return []

        span = best_candidate[-1]
        span.pop("_source", None)
        return [span]
    except Exception:
        return []
