import re


def rule_first_listed_attorney_appellant(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return " ".join((text or "").split())

        def compact(text: str) -> str:
            return re.sub(r"[^a-z]+", "", (text or "").lower())

        def line_dicts() -> list[dict]:
            lines = doc.get("lines") or []
            if lines:
                return [line for line in lines if isinstance(line, dict)]
            text = doc.get("text") or ""
            return [{"page_no": None, "line_no": i + 1, "text": raw} for i, raw in enumerate(text.splitlines())]

        def make_span(text: str, src: dict | None) -> dict:
            span = {"text": text}
            if isinstance(src, dict):
                if src.get("page_no") is not None:
                    span["page_no"] = src.get("page_no")
                if src.get("line_no") is not None:
                    span["line_no"] = src.get("line_no")
            return span

        def extract_caption_roles(lines: list[dict]) -> set[str]:
            role_aliases = {
                "appellant",
                "appellants",
                "petitioner",
                "petitioners",
                "portioner",
                "portioners",
            }
            role_patterns = (
                "plaintiff",
                "plaintiffs",
                "defendant",
                "defendants",
                "intervenor",
                "intervenors",
                "petitioner",
                "petitioners",
                "portioner",
                "portioners",
                "respondent",
                "respondents",
                "claimant",
                "claimants",
                "intervenordefendant",
                "intervenordefendants",
                "intervenorplaintiff",
                "intervenorplaintiffs",
                "intervenorrespondent",
                "intervenorrespondents",
                "intervenorpetitioner",
                "intervenorpetitioners",
            )
            for item in lines[:160]:
                text = compact(item.get("text", ""))
                if "appellant" not in text:
                    continue
                for pattern in role_patterns:
                    if pattern in text:
                        role_aliases.add(pattern)
            return role_aliases

        def is_major_heading(text: str) -> bool:
            return bool(
                re.match(
                    r"(?i)^\s*(?:OPINION|ORDER|SUMMARY|JUDGMENT|MEMORANDUM|BACKGROUND|INTRODUCTION|DISCUSSION|CONCLUSION|PER CURIAM)\b",
                    text or "",
                )
            )

        def is_counsel_heading(text: str) -> bool:
            return bool(re.match(r"(?i)^\s*(?:COUNSEL|ATTORNEYS?)\s*:?\s*$", text or ""))

        def first_attorney(block_text: str) -> str | None:
            text = norm(block_text)
            if not text:
                return None

            text = re.sub(r"(?i)^\s*counsel for\s+.*?:\s*", "", text).strip()
            text = re.sub(r"(?i),?\s*for\s+(?:the\s+)?(?:appellant|appellants)\b.*$", "", text).strip()
            candidate = text.split(";", 1)[0].strip()
            candidate = candidate.split(",", 1)[0].strip()
            candidate = re.split(r"(?i)\s+\band\b\s+", candidate, 1)[0].strip()
            candidate = re.sub(r"\s*\((?:argued|submitted|on brief|lead counsel|co-counsel|pro hac vice)\)\s*$", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"\s*\([^)]*\)\s*$", "", candidate).strip(" ,;:")

            if candidate and len(candidate.split()) >= 2:
                if not re.search(
                    r"(?i)\b(?:llp|llc|pc|p\.c\.|inc\.?|corporation|company|department|office|attorney|attorneys|counsel|group|association|clinic|services|partners|foundation|trust)\b",
                    candidate,
                ):
                    return candidate

            name_match = re.search(
                r"\b([A-Z][A-Za-z.'’\-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z.'’\-]+){1,4}(?:\s+Jr\.)?)\b",
                text,
            )
            if not name_match:
                return None
            fallback = norm(name_match.group(1)).strip(" ,;:")
            if len(fallback.split()) < 2:
                return None
            return fallback

        lines = line_dicts()
        if not lines:
            return []

        appellant_roles = extract_caption_roles(lines)
        generic_roles = {
            "appellant",
            "appellants",
            "appellee",
            "appellees",
            "plaintiff",
            "plaintiffs",
            "defendant",
            "defendants",
            "intervenor",
            "intervenors",
            "petitioner",
            "petitioners",
            "portioner",
            "portioners",
            "respondent",
            "respondents",
            "claimant",
            "claimants",
            "intervenordefendant",
            "intervenordefendants",
            "intervenorplaintiff",
            "intervenorplaintiffs",
            "intervenorrespondent",
            "intervenorrespondents",
            "intervenorpetitioner",
            "intervenorpetitioners",
            "amicus",
            "amici",
        }
        explicit_roles = {
            role
            for role in appellant_roles
            if role not in {"appellant", "appellants", "petitioner", "petitioners", "portioner", "portioners"}
        }
        target_roles = explicit_roles or {"petitioner", "petitioners", "portioner", "portioners"}

        def is_any_side_marker(text: str) -> bool:
            ctext = compact(text)
            if "for" not in ctext and not ctext.startswith("counselfor"):
                return False
            return any(f"for{role}" in ctext or f"counselfor{role}" in ctext for role in generic_roles)

        def marker_priority(text: str) -> int | None:
            ctext = compact(text)
            target_hit = any(
                role in ctext and (
                    f"for{role}" in ctext
                    or f"counselfor{role}" in ctext
                    or ctext.endswith(role)
                    or f"{role}appellant" in ctext
                    or f"{role}appellee" in ctext
                )
                for role in target_roles
            )
            direct_appellant = "appellant" in ctext and "crossappellant" not in ctext
            cross_appellant = "crossappellant" in ctext
            if target_hit and direct_appellant:
                return 0
            if target_hit and not cross_appellant:
                return 1
            if direct_appellant and not explicit_roles:
                return 2
            if target_hit and cross_appellant:
                return 3
            if cross_appellant and not explicit_roles:
                return 4
            if "appellee" in ctext and "appellant" not in ctext:
                return None
            if direct_appellant:
                return 5
            return None

        def block_result(block_start: int, block_end: int) -> list[dict]:
            start = block_start
            while start <= block_end and not norm(lines[start].get("text", "")):
                start += 1
            if start > block_end:
                return []

            block_lines = []
            for idx in range(start, block_end + 1):
                raw = norm(lines[idx].get("text", ""))
                if not raw:
                    continue
                if is_major_heading(raw) or is_counsel_heading(raw):
                    continue
                if raw.startswith("The Honorable "):
                    continue
                if re.fullmatch(r"\d+", raw):
                    continue
                if re.fullmatch(r"[A-Z][A-Z0-9 .,&'\-]{8,}", raw) and " V. " in raw:
                    continue
                block_lines.append(raw)
            if not block_lines:
                return []

            name = first_attorney(" ".join(block_lines))
            if not name:
                return []
            return [make_span(name, lines[start])]

        def is_side_marker_end(section_start: int, idx: int) -> bool:
            current = norm(lines[idx].get("text", ""))
            if not current:
                return False
            current_compact = compact(current)
            if current_compact.startswith("counselfor") and is_any_side_marker(current):
                return True
            if is_any_side_marker(current):
                return True
            if (
                any(role in current_compact for role in generic_roles)
                and len(current.split()) <= 4
                and not any(ch in current for ch in ",;:@")
            ):
                return True
            if ";" in current and any(role in current_compact for role in generic_roles):
                return True
            if (
                any(role in current_compact for role in target_roles | {"appellant", "appellants"})
                and (" and " in current.lower() or current.endswith("."))
            ):
                return True
            if idx <= section_start:
                return False
            previous = norm(lines[idx - 1].get("text", ""))
            previous_compact = compact(previous)
            if not previous or "for" not in previous_compact:
                return False
            return is_any_side_marker(f"{previous} {current}") and any(role in current_compact for role in generic_roles)

        counsel_indices = [i for i, item in enumerate(lines) if is_counsel_heading(norm(item.get("text", "")))]
        for counsel_idx in counsel_indices:
            section_end = len(lines)
            for j in range(counsel_idx + 1, len(lines)):
                text = norm(lines[j].get("text", ""))
                if not text:
                    continue
                if is_counsel_heading(text) or is_major_heading(text):
                    section_end = j
                    break

            marker_indices: list[int] = []
            marker_windows: dict[int, str] = {}
            for end in range(counsel_idx + 1, section_end):
                if not is_side_marker_end(counsel_idx + 1, end):
                    continue
                pieces = [
                    norm(lines[k].get("text", ""))
                    for k in range(max(counsel_idx + 1, end - 1), end + 1)
                    if norm(lines[k].get("text", ""))
                ]
                window = " ".join(pieces)
                marker_indices.append(end)
                marker_windows[end] = window

            candidates: list[tuple[int, int, list[dict]]] = []
            prev_end = counsel_idx
            for end in marker_indices:
                priority = marker_priority(marker_windows[end])
                if priority is not None:
                    spans = block_result(prev_end + 1, end)
                    if spans:
                        candidates.append((priority, end, spans))
                prev_end = end
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]))
                return candidates[0][2]

        inline_counsel_re = re.compile(r"(?i)^\s*counsel for\s+(.+?)\s*:?\s*(.*)$")
        for i, item in enumerate(lines):
            raw = item.get("text", "") or ""
            match = inline_counsel_re.match(raw)
            if not match:
                continue
            label = compact(match.group(1))
            if marker_priority(label) is None:
                continue
            tail = norm(match.group(2))
            if tail:
                name = first_attorney(tail)
                if name:
                    return [make_span(name, item)]
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = norm(lines[j].get("text", ""))
                if not candidate or is_major_heading(candidate) or is_any_side_marker(candidate):
                    break
                name = first_attorney(candidate)
                if name:
                    return [make_span(name, lines[j])]

        caption_counsel_re = re.compile(r"(?i)\bcounsel\s+for\s+(.+?)[,.:]?\s*$")
        for i in range(min(80, len(lines))):
            raw = norm(lines[i].get("text", ""))
            next_raw = norm(lines[i + 1].get("text", "")) if i + 1 < len(lines) else ""
            if not raw and not next_raw:
                continue
            caption_window = raw
            if "counsel" in raw.lower() and next_raw.lower().startswith("for "):
                caption_window = f"{raw} {next_raw}"
            match = caption_counsel_re.search(caption_window)
            if not match:
                continue
            label = compact(match.group(1))
            nearby = " ".join(
                norm(lines[j].get("text", ""))
                for j in range(max(0, i - 4), min(len(lines), i + 5))
                if norm(lines[j].get("text", ""))
            )
            nearby_compact = compact(nearby)
            if marker_priority(f"for {label}") is None and marker_priority(nearby_compact) is None:
                continue

            start = max(0, i - 3)
            for j in range(i - 1, -1, -1):
                prev = norm(lines[j].get("text", ""))
                prev_compact = compact(prev)
                if not prev:
                    start = j + 1
                    break
                if prev_compact == "v":
                    start = j + 1
                    break
                if any(token in prev_compact for token in generic_roles):
                    start = j + 1
                    break

            block_end = i + 1 if caption_window != raw else i
            block = " ".join(
                norm(lines[j].get("text", ""))
                for j in range(start, block_end + 1)
                if norm(lines[j].get("text", ""))
            )
            name = first_attorney(block)
            if name:
                return [make_span(name, lines[start])]

        info_label_re = re.compile(r"(?i)^(?:counsel\s+for|for)\s+(.+?)(?::\s*|,\s*$)")
        for i in range(len(lines)):
            raw = norm(lines[i].get("text", ""))
            if not raw:
                continue
            combined = raw
            consume_next = False
            if re.match(r"(?i)^counsel\s+for\b", raw) and not re.search(r"[:]\s*$", raw):
                next_raw = norm(lines[i + 1].get("text", "")) if i + 1 < len(lines) else ""
                if next_raw.endswith(":"):
                    combined = f"{raw} {next_raw}"
                    consume_next = True
            match = info_label_re.match(combined)
            if not match:
                continue
            label = match.group(1)
            if marker_priority(f"for {label}") is None:
                continue
            if ":" in combined:
                tail = norm(combined.split(":", 1)[1])
                if tail:
                    name = first_attorney(tail)
                    if name:
                        return [make_span(name, lines[i])]
            start_idx = i + 1 + (1 if consume_next else 0)
            for j in range(start_idx, min(start_idx + 6, len(lines))):
                candidate = norm(lines[j].get("text", ""))
                if not candidate:
                    continue
                if info_label_re.match(candidate) or is_major_heading(candidate):
                    break
                if candidate.endswith(":") and j != start_idx:
                    break
                name = first_attorney(candidate)
                if name:
                    return [make_span(name, lines[j])]

        info_context_re = re.compile(r"(?i)(?:counsel information|names and addresses of counsel|list of counsel|included below)")
        for i in range(len(lines)):
            current = norm(lines[i].get("text", ""))
            if not current or marker_priority(current) is None:
                continue
            if not any(info_context_re.search(norm(lines[j].get("text", ""))) for j in range(max(0, i - 12), i)):
                continue
            start = i
            for j in range(i - 1, max(-1, i - 10), -1):
                prev = norm(lines[j].get("text", ""))
                if not prev:
                    start = j + 1
                    break
                if info_label_re.match(prev) or is_major_heading(prev) or info_context_re.search(prev):
                    start = j + 1
                    break
            block = " ".join(
                norm(lines[j].get("text", ""))
                for j in range(start, i + 1)
                if norm(lines[j].get("text", ""))
            )
            name = first_attorney(block)
            if name:
                return [make_span(name, lines[start])]

        return [{"text": "NOT FOUND"}]
    except Exception:
        return []
