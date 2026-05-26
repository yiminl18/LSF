import re


def rule_esf_total_assets(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []

        def norm(text: str) -> str:
            return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())

        def compact(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

        def has_esf1(normal_text: str, compact_text: str) -> bool:
            return bool(re.search(r"\btable\s+esf\s+[1l]\b", normal_text)) or "tableesf1" in compact_text or "tableesfl" in compact_text

        if not lines:
            return []

        normalized_lines = [norm(line.get("text", "")) for line in lines]
        compact_lines = [compact(line.get("text", "")) for line in lines]

        def join_window(start: int, end: int) -> str:
            return " ".join(normalized_lines[start:min(end, len(normalized_lines))])

        def join_compact(start: int, end: int) -> str:
            return "".join(compact_lines[start:min(end, len(compact_lines))])

        def find_phrase_idx(start: int, end: int, phrase_parts: list[str]) -> int:
            target = "".join(phrase_parts)
            upper = min(end, len(compact_lines))
            for idx in range(start, upper):
                built = ""
                for look_ahead in range(4):
                    pos = idx + look_ahead
                    if pos >= upper:
                        break
                    built += compact_lines[pos]
                    if built == target:
                        return idx
                    if not target.startswith(built):
                        break
            for idx in range(start, upper):
                if target in join_compact(idx, idx + 4):
                    return idx
            return -1

        start_candidates = []
        doc_compact = compact(doc.get("text", ""))
        for idx, text in enumerate(normalized_lines):
            compact_window = join_compact(max(0, idx - 2), idx + 14)
            if has_esf1(join_window(max(0, idx - 2), idx + 14), compact_window) and (
                "balancesasof" in compact_window
                or "balanceasof" in compact_window
                or "exchangestabilizationfund" in compact_window
                or "assetsliabilitiesandcapital" in compact_window
            ):
                start_candidates.append(idx)

        if not start_candidates and "exchangestabilizationfund" in doc_compact:
            for idx in range(len(lines)):
                compact_window = join_compact(max(0, idx - 8), idx + 40)
                if (
                    "exchangestabilizationfund" in compact_window
                    and "totalassets" in compact_window
                    and "totalliabilitiesandcapital" in compact_window
                ):
                    start_candidates.append(max(0, idx - 8))

        for start_idx in start_candidates:
            total_idx = find_phrase_idx(start_idx, min(len(lines), start_idx + 260), ["total", "assets"])
            liabilities_idx = -1

            if total_idx == -1:
                continue

            end_idx = min(len(lines), total_idx + 90)
            liabilities_idx = find_phrase_idx(
                total_idx,
                min(len(lines), total_idx + 120),
                ["total", "liabilities", "and", "capital"],
            )
            if liabilities_idx != -1:
                end_idx = min(len(lines), liabilities_idx + 60)

            begin_idx = start_idx
            snippet_lines = lines[begin_idx:end_idx]
            snippet_text = "\n".join(
                (line.get("text") or "").rstrip() for line in snippet_lines if line.get("text")
            )
            if snippet_text.strip():
                first_line = snippet_lines[0]
                last_line = snippet_lines[-1]
                main_span = {
                    "text": snippet_text,
                }
                if "page_no" in first_line:
                    main_span["page_no"] = first_line.get("page_no")
                if "line_no" in first_line:
                    main_span["line_no"] = first_line.get("line_no")
                if "page_no" in last_line:
                    main_span["end_page_no"] = last_line.get("page_no")
                if "line_no" in last_line:
                    main_span["end_line_no"] = last_line.get("line_no")

                spans = [main_span]
                if liabilities_idx != -1:
                    date_lines = []
                    for idx in range(start_idx, min(len(lines), total_idx)):
                        raw = (lines[idx].get("text") or "").rstrip()
                        compact_raw = compact(raw)
                        if not raw:
                            continue
                        if re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", raw, re.IGNORECASE) or re.search(r"\b(19|20)\d{2}\b", raw) or compact_raw in {"through"}:
                            date_lines.append(raw)

                    total_line_raw = (lines[total_idx].get("text") or "").strip()
                    if re.search(r"\d", total_line_raw):
                        inline_numbers = re.findall(r"\(?[-]?\d[\d,.\s]*\)?", total_line_raw)
                        inline_numbers = [num.strip() for num in inline_numbers if re.search(r"\d", num)]
                        if inline_numbers:
                            hint_text = "\n".join(date_lines[-4:] + ["Total assets", inline_numbers[-1]])
                            hint_span = {
                                "text": hint_text,
                            }
                            total_line = lines[total_idx]
                            if "page_no" in total_line:
                                hint_span["page_no"] = total_line.get("page_no")
                                hint_span["end_page_no"] = total_line.get("page_no")
                            if "line_no" in total_line:
                                hint_span["line_no"] = total_line.get("line_no")
                                hint_span["end_line_no"] = total_line.get("line_no")
                            spans.insert(0, hint_span)

                    numeric_entries = []
                    for idx in range(liabilities_idx + 1, min(len(lines), liabilities_idx + 80)):
                        raw = (lines[idx].get("text") or "").strip()
                        compact_raw = compact(raw)
                        if compact_raw in {"tableesf2", "glossary"} or "exchangestabilizationfund" in compact_raw:
                            break
                        if re.fullmatch(r"[\s$(),.\-0-9]+", raw) and re.search(r"\d", raw):
                            digits = re.sub(r"[^0-9]", "", raw)
                            if digits:
                                numeric_entries.append((int(digits), raw, idx))

                    if len(numeric_entries) >= 15 and not re.search(r"\d", lines[total_idx].get("text", "")):
                        _, best_raw, best_idx = max(numeric_entries, key=lambda item: item[0])
                        hint_lines = date_lines[-4:] + ["Total assets", best_raw, "Total liabilities and capital", best_raw]
                        hint_text = "\n".join(line for line in hint_lines if line)
                        if hint_text.strip():
                            hint_span = {
                                "text": hint_text,
                            }
                            best_line = lines[best_idx]
                            total_line = lines[total_idx]
                            if "page_no" in total_line:
                                hint_span["page_no"] = total_line.get("page_no")
                            if "line_no" in total_line:
                                hint_span["line_no"] = total_line.get("line_no")
                            if "page_no" in best_line:
                                hint_span["end_page_no"] = best_line.get("page_no")
                            if "line_no" in best_line:
                                hint_span["end_line_no"] = best_line.get("line_no")
                            spans.insert(0, hint_span)

                return spans

        return []
    except Exception:
        return []
