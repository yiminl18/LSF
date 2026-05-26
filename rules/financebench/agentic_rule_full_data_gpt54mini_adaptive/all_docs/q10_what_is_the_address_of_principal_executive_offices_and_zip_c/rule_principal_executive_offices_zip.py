def rule_principal_executive_offices_zip(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        if not isinstance(lines, list) or not lines:
            return []

        def _norm(value: object) -> str:
            return " ".join(str(value or "").split())

        def _same_page(a: int, b: int) -> bool:
            return lines[a].get("page_no") == lines[b].get("page_no")

        def _is_boilerplate(text: str) -> bool:
            lowered = text.lower()
            return any(
                needle in lowered
                for needle in (
                    "registrant",
                    "telephone",
                    "employer identification",
                    "incorporation",
                    "principal executive offices",
                    "commission file",
                    "table of contents",
                    "exact name of registrant",
                    "state or other jurisdiction",
                )
            )

        needle = re.compile(r"principal executive offices?|address of principal executive offices?", re.I)
        candidate_indices = [
            idx
            for idx, line in enumerate(lines)
            if needle.search(_norm(line.get("text")))
        ]
        if not candidate_indices:
            return []

        candidate = min(
            candidate_indices,
            key=lambda idx: (
                lines[idx].get("page_no", 10**9),
                lines[idx].get("line_no", 10**9),
            ),
        )

        selected = [candidate]

        prev = candidate - 1
        while prev >= 0 and len(selected) < 5:
            if not _same_page(prev, candidate):
                break
            text = _norm(lines[prev].get("text"))
            if not text:
                prev -= 1
                continue
            if _is_boilerplate(text):
                break
            selected.insert(0, prev)
            prev -= 1

        # Short forward fallback for layouts where the label appears before the
        # actual address block.
        if not any(re.search(r"\d", _norm(lines[idx].get("text"))) for idx in selected):
            nxt = candidate + 1
            added = 0
            while nxt < len(lines) and added < 2:
                if not _same_page(nxt, candidate):
                    break
                text = _norm(lines[nxt].get("text"))
                if not text:
                    nxt += 1
                    continue
                selected.append(nxt)
                added += 1
                nxt += 1

        candidate_text = _norm(lines[candidate].get("text"))
        candidate_looks_like_address = bool(
            re.search(r"\d", candidate_text)
            or re.search(
                r"\b(street|st\.|road|rd\.|avenue|ave\.|boulevard|blvd\.|drive|dr\.|parkway|pkwy\.|highway|hwy\.|lane|ln\.|court|ct\.|circle|cir\.|suite|ste\.|way|plaza|center|centre|place|po box|p\.o\.)\b",
                candidate_text,
                re.I,
            )
            or ("," in candidate_text and re.search(r"[A-Za-z]", candidate_text))
        )

        texts: list[str] = []
        for idx in selected:
            text = _norm(lines[idx].get("text"))
            if not text:
                continue
            if idx == candidate and not candidate_looks_like_address:
                continue
            texts.append(text)

        if not texts:
            if candidate_text:
                texts = [candidate_text]
            else:
                return []

        start_idx = selected[0]
        end_idx = selected[-1]
        return [
            {
                "text": ", ".join(texts),
                "page_no": lines[start_idx].get("page_no"),
                "line_no": lines[start_idx].get("line_no"),
                "line_no_end": lines[end_idx].get("line_no"),
            }
        ]
    except Exception:
        return []
