import re


_NON_TARGET_HINTS = (
    "weighted average",
    "commercial paper",
    "debt",
    "derivative",
    "options",
    "option",
    "warrant",
    "preferred stock",
    "borrowings",
    "lease",
    "anti-dilutive",
)


def rule_cover_page_common_stock_outstanding(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "").replace("\u2019", "'").replace("\xa0", " ")).strip()

        def alpha_word_count(text: str) -> int:
            return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))

        def extract_count(text: str) -> str | None:
            numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
            if not numbers:
                return None
            return max(numbers, key=lambda token: len(re.sub(r"\D", "", token)))

        def candidate_score(text: str) -> int:
            low = norm(text).lower()
            if not low or "outstanding" not in low:
                return 0

            if "indicate the number of shares" in low or "latest practicable date" in low:
                return 0

            if any(hint in low for hint in _NON_TARGET_HINTS):
                if "common stock" not in low and "registrant" not in low and "shares" not in low:
                    return 0

            score = 1
            if "common stock" in low:
                score += 4
            if "registrant" in low and "shares" in low:
                score += 2
            if "issued and outstanding" in low and "share" in low:
                score += 3
            if "outstanding as of" in low and "share" in low:
                score += 3
            if "outstanding at" in low:
                score += 4
            if "there were" in low and "share" in low:
                score += 2
            if "the number of shares" in low and "outstanding" in low:
                score += 3
            if "as of" in low:
                score += 1
            if re.search(r"\d", low):
                score += 3
            if low.endswith(":"):
                score += 1
            if "class" in low and "common stock" in low:
                score += 1
            return score

        def is_context(text: str) -> bool:
            low = norm(text).lower()
            if not low:
                return False
            if candidate_score(low) > 0:
                return True
            if re.fullmatch(r"[\d,().$\-\s]+", low) and len(re.sub(r"\D", "", low)) >= 3:
                return True
            if re.search(r"\d", low) and "share" in low:
                return True
            if "outstanding" in low and ("common stock" in low or "share" in low or "registrant" in low):
                return True
            if "as of" in low and "common stock" in low and alpha_word_count(low) <= 24:
                return True
            return False

        def build_window(items: list[dict], idx: int) -> dict | None:
            start = max(0, idx - 1)
            end = min(len(items), idx + 4)
            snippet_lines = []
            meta = items[idx]
            for j, item in enumerate(items[start:end], start=start):
                item_text = norm(item.get("text") or "")
                if not item_text or not is_context(item_text):
                    continue

                low = item_text.lower()
                if j < idx:
                    if re.fullmatch(r"[\d,().$\-\s]+", low):
                        continue
                    if not (
                        candidate_score(item_text) > 0
                        or ("outstanding" in low and ("common stock" in low or "share" in low or "registrant" in low))
                    ):
                        continue

                count = extract_count(item_text)
                if count and (
                    re.fullmatch(r"[\d,().$\-\s]+", low)
                    or re.fullmatch(r"\d[\d,]*(?:\.\d+)?\s+shares?", low)
                    or " was " in f" {low} "
                    or " were " in f" {low} "
                    or "issued and outstanding" in low
                    or ("share" in low and "outstanding" in low and "as of" not in low)
                ):
                    keep = count
                else:
                    keep = item_text

                if keep and keep not in snippet_lines:
                    snippet_lines.append(keep)
            if not snippet_lines:
                return None
            snippet = "\n".join(snippet_lines).strip()
            if not snippet:
                return None
            return (
                {
                    "text": snippet,
                    **{k: v for k, v in meta.items() if k in ("page_no", "line_no", "paragraph_no")},
                }
            )

        for source_key in ("lines", "paragraphs"):
            items = [item for item in (doc.get(source_key) or []) if isinstance(item, dict)]
            items.sort(
                key=lambda d: (
                    int(d.get("page_no") or 0),
                    int(d.get("line_no") or d.get("paragraph_no") or 0),
                )
            )
            for idx, item in enumerate(items):
                text = norm(item.get("text") or "")
                if not text:
                    continue
                page_no = item.get("page_no")
                if page_no is not None and int(page_no) > 2:
                    continue
                if candidate_score(text) > 0:
                    span = build_window(items, idx)
                    if span:
                        return [span]

        for item in [p for p in (doc.get("pages") or []) if isinstance(p, dict)]:
            page_no = item.get("page_no")
            if page_no is not None and int(page_no) > 2:
                continue
            text = norm(item.get("text") or "")
            if not text or candidate_score(text) <= 0:
                continue

            # Page-level fallback. Keep only a local slice around the first hit.
            m = re.search(r".{0,180}(?:common stock|outstanding|shares).{0,240}", text, re.IGNORECASE)
            snippet = norm(m.group(0)) if m else text[:500]
            return [{"page_no": page_no, "text": snippet}]

        return []
    except Exception:
        return []
