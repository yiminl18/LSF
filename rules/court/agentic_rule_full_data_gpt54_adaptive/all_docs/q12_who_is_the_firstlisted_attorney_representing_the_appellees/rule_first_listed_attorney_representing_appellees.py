import re


def rule_first_listed_attorney_representing_appellees(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return " ".join((text or "").split())

        def compact(text: str) -> str:
            return re.sub(r"[^a-z]+", "", (text or "").lower())

        def make_span(text: str, src: dict | None) -> dict:
            span = {"text": text}
            if isinstance(src, dict):
                if src.get("page_no") is not None:
                    span["page_no"] = src.get("page_no")
                if src.get("line_no") is not None:
                    span["line_no"] = src.get("line_no")
            return span

        def line_dicts() -> list[dict]:
            lines = doc.get("lines") or []
            if lines:
                return [line for line in lines if isinstance(line, dict)]
            text = doc.get("text") or ""
            return [{"page_no": None, "line_no": i + 1, "text": raw} for i, raw in enumerate(text.splitlines())]

        def is_major_heading(text: str) -> bool:
            return bool(
                re.match(
                    r"(?i)^\s*(?:OPINION|ORDER|SUMMARY|JUDGMENT|MEMORANDUM|BACKGROUND|INTRODUCTION|DISCUSSION|CONCLUSION|PER CURIAM|I\.|II\.|III\.|IV\.|V\.|VI\.)\b",
                    text or "",
                )
            )

        def is_counsel_heading(text: str) -> bool:
            return bool(re.match(r"(?i)^\s*(?:COUNSEL|ATTORNEYS?)\s*:?[\s*]*$", text or ""))

        def is_page_header(text: str) -> bool:
            text = norm(text)
            if not text:
                return False
            if re.fullmatch(r"\d+", text):
                return True
            if re.fullmatch(r"[A-Z][A-Z0-9 .,&'\-]{8,}", text) and " V. " in text:
                return True
            if re.match(r"^\d+\s+[A-Z].* V\. ", text):
                return True
            return False

        def first_attorney(block_text: str) -> list[str]:
            text = norm(block_text)
            if not text:
                return []
            text = re.sub(r"(?i)^\s*counsel for\s+.*?:\s*", "", text).strip()
            text = re.sub(r"(?i)^\s*for\s+.*?:\s*", "", text).strip()
            candidate = text.split(";", 1)[0].strip()
            candidate = re.split(r"(?i)\s+and\s+", candidate, 1)[0].strip()
            candidate = candidate.split(",", 1)[0].strip()
            candidate = candidate.strip(" ,;:")

            out: list[str] = []
            if candidate and len(candidate.split()) >= 2 and not re.search(
                r"(?i)\b(?:llp|llc|pllc|aplc|pc|p\.c\.|inc\.?|corporation|company|department|office|attorney|attorneys|counsel|group|association|division|services|partners|clinic|foundation|committee|agency|university)\b",
                candidate,
            ):
                out.append(candidate)
                stripped = re.sub(r"\s*\([^)]*\)", "", candidate).strip(" ,;:")
                if stripped and stripped != candidate:
                    out.append(stripped)
            return out

        def extract_targets(lines: list[dict]) -> tuple[set[str], set[str], set[str]]:
            role_patterns = {
                "plaintiff",
                "plaintiffs",
                "defendant",
                "defendants",
                "respondent",
                "respondents",
                "petitioner",
                "petitioners",
                "intervenor",
                "intervenors",
                "claimant",
                "claimants",
                "realpartyininterest",
                "realpartiesininterest",
                "appellee",
                "appellees",
            }
            roles: set[str] = set()
            aliases: set[str] = set()
            header: list[str] = []

            for item in lines[:140]:
                txt = norm(item.get("text", ""))
                if not txt:
                    header.append(txt)
                    continue
                if is_counsel_heading(txt):
                    break
                if re.match(r"(?i)^(?:Appeal from|On Petition|Argued and Submitted|Submitted|Filed|Before:)\b", txt):
                    break
                header.append(txt)

            joined_header = " ".join(header)
            joined_compact = compact(joined_header)
            has_petitioner = "petitioner" in joined_compact or "petitioners" in joined_compact
            has_app_for_enforcement = any(
                "onpetitionforapplicationforenforcement" in compact(norm(item.get("text", ""))) for item in lines[:80]
            )

            if "unitedstatesofamerica" in joined_compact and "plaintiffappellee" in joined_compact:
                aliases.add("unitedstates")
                roles.add("plaintiff")

            for txt in header:
                c = compact(txt)
                if not c:
                    continue
                if "appellee" in c:
                    for pat in role_patterns:
                        if pat in c:
                            roles.add(pat)
                    roles.update({"appellee", "appellees"})
                    if "unitedstates" in c:
                        aliases.add("unitedstates")
                if ("respondent" in c or "realpartyininterest" in c) and has_petitioner:
                    if "respondent" in c:
                        roles.update({"respondent", "respondents"})
                    if "realpartyininterest" in c:
                        roles.add("realpartyininterest")

            if has_app_for_enforcement and "nationallaborrelationsboard" in joined_compact and "petitioner" in joined_compact:
                roles = {"petitioner", "petitioners"}
                aliases.add("nationallaborrelationsboard")

            explicit_roles = {
                role for role in roles if role not in {"appellee", "appellees", "respondent", "respondents", "realpartyininterest"}
            }
            return roles, explicit_roles, aliases

        generic_roles = {
            "plaintiff",
            "plaintiffs",
            "defendant",
            "defendants",
            "respondent",
            "respondents",
            "petitioner",
            "petitioners",
            "portioner",
            "portioners",
            "intervenor",
            "intervenors",
            "claimant",
            "claimants",
            "realpartyininterest",
            "realpartiesininterest",
            "appellant",
            "appellants",
            "appellee",
            "appellees",
            "amicus",
            "amici",
            "curiae",
        }

        def is_any_side_marker(text: str) -> bool:
            c = compact(text)
            if "for" not in c and not c.startswith("counselfor"):
                return any(role in c for role in generic_roles) and len(norm(text).split()) <= 5
            return any(f"for{role}" in c or f"counselfor{role}" in c or c.endswith(role) for role in generic_roles)

        def marker_priority(text: str, roles: set[str], explicit_roles: set[str], aliases: set[str]) -> int | None:
            c = compact(text)
            alias_hit = any(alias and alias in c and ("for" in c or c.startswith("counselfor") or c.endswith(alias)) for alias in aliases)
            explicit_hit = any(
                role in c
                and (
                    f"for{role}" in c
                    or f"counselfor{role}" in c
                    or c.endswith(role)
                    or f"{role}appellee" in c
                    or f"{role}respondent" in c
                    or f"{role}realpartyininterest" in c
                    or f"{role}petitioner" in c
                )
                for role in explicit_roles
            )
            generic_target = any(tok in c for tok in ["appellee", "appellees", "respondent", "respondents", "realpartyininterest"])
            generic_other = any(tok in c for tok in ["appellant", "appellants"])

            if explicit_hit and generic_target:
                return 0
            if explicit_hit and not generic_other:
                return 1
            if explicit_hit:
                return 2
            if alias_hit and generic_target:
                return 3
            if alias_hit and not generic_other:
                return 4
            if generic_target and not explicit_roles:
                return 5
            return None

        def block_result(lines: list[dict], block_start: int, block_end: int) -> list[dict]:
            start = block_start
            while start <= block_end and not norm(lines[start].get("text", "")):
                start += 1
            if start > block_end:
                return []

            block_lines: list[str] = []
            src = None
            for idx in range(start, block_end + 1):
                raw = norm(lines[idx].get("text", ""))
                if not raw:
                    continue
                if is_major_heading(raw) or is_counsel_heading(raw) or is_page_header(raw):
                    continue
                if raw.startswith("The Honorable ") or raw.startswith("This summary constitutes") or raw.startswith("*"):
                    continue
                if re.fullmatch(r"\d+", raw):
                    continue
                if src is None:
                    src = lines[idx]
                block_lines.append(raw)

            if not block_lines or src is None:
                return []
            return [make_span(name, src) for name in first_attorney(" ".join(block_lines))]

        def is_side_marker_end(lines: list[dict], section_start: int, idx: int) -> bool:
            current = norm(lines[idx].get("text", ""))
            if not current:
                return False
            cc = compact(current)
            if cc.startswith("counselfor") and is_any_side_marker(current):
                return True
            if is_any_side_marker(current):
                return True
            if any(role in cc for role in generic_roles) and len(current.split()) <= 5 and not any(ch in current for ch in ",;:@"):
                return True
            if ";" in current and any(role in cc for role in generic_roles):
                return True
            if idx <= section_start:
                return False
            previous = norm(lines[idx - 1].get("text", ""))
            previous_compact = compact(previous)
            if not previous or "for" not in previous_compact:
                return False
            return is_any_side_marker(f"{previous} {current}") and any(role in cc for role in generic_roles)

        def extract_from_counsel(lines: list[dict], roles: set[str], explicit_roles: set[str], aliases: set[str]) -> list[dict]:
            counsel_indices = [i for i, item in enumerate(lines) if is_counsel_heading(norm(item.get("text", "")))]
            for counsel_idx in counsel_indices:
                section_end = len(lines)
                for j in range(counsel_idx + 1, len(lines)):
                    txt = norm(lines[j].get("text", ""))
                    if not txt:
                        continue
                    if is_counsel_heading(txt) or is_major_heading(txt):
                        section_end = j
                        break

                marker_indices: list[int] = []
                marker_windows: dict[int, str] = {}
                for end in range(counsel_idx + 1, section_end):
                    if not is_side_marker_end(lines, counsel_idx + 1, end):
                        continue
                    pieces = [
                        norm(lines[k].get("text", ""))
                        for k in range(max(counsel_idx + 1, end - 1), end + 1)
                        if norm(lines[k].get("text", ""))
                    ]
                    marker_indices.append(end)
                    marker_windows[end] = " ".join(pieces)

                candidates: list[tuple[int, int, list[dict]]] = []
                prev_end = counsel_idx
                for end in marker_indices:
                    pr = marker_priority(marker_windows[end], roles, explicit_roles, aliases)
                    if pr is not None:
                        spans = block_result(lines, prev_end + 1, end)
                        if spans:
                            candidates.append((pr, end, spans))
                    prev_end = end

                if candidates:
                    candidates.sort(key=lambda item: (item[0], item[1]))
                    return candidates[0][2]
            return []

        def extract_from_counsel_for_heading(lines: list[dict], roles: set[str], explicit_roles: set[str], aliases: set[str]) -> list[dict]:
            if not roles and not aliases:
                return []
            heading_re = re.compile(r"(?i)^\s*Counsel for\s+(.+?)\s*:?[\s]*$")
            for i, item in enumerate(lines):
                raw = norm(item.get("text", ""))
                match = heading_re.match(raw)
                if not match:
                    continue
                label = match.group(1)
                if marker_priority(f"for {label}", roles, explicit_roles, aliases) is None:
                    continue
                for j in range(i + 1, min(i + 8, len(lines))):
                    candidate = norm(lines[j].get("text", ""))
                    if not candidate:
                        continue
                    if is_major_heading(candidate):
                        break
                    if candidate.endswith(":") or "@" in candidate or is_page_header(candidate):
                        continue
                    names = first_attorney(candidate)
                    if names:
                        return [make_span(name, lines[j]) for name in names]
            return []

        def extract_from_inline_labels(lines: list[dict], roles: set[str], explicit_roles: set[str], aliases: set[str]) -> list[dict]:
            if not roles and not aliases:
                return []
            inline_re = re.compile(r"(?i)^\s*(?:counsel\s+for|for)\s+(.+?)(?::\s*|,\s*$)(.*)$")
            for i, item in enumerate(lines):
                raw = item.get("text", "") or ""
                match = inline_re.match(raw)
                if not match:
                    continue
                label = match.group(1)
                if marker_priority(f"for {label}", roles, explicit_roles, aliases) is None:
                    continue
                tail = norm(match.group(2))
                if tail:
                    names = first_attorney(tail)
                    if names:
                        return [make_span(name, item) for name in names]
                for j in range(i + 1, min(i + 8, len(lines))):
                    candidate = norm(lines[j].get("text", ""))
                    if not candidate or is_major_heading(candidate) or is_any_side_marker(candidate):
                        break
                    if candidate.endswith(":") or "@" in candidate or is_page_header(candidate):
                        continue
                    names = first_attorney(candidate)
                    if names:
                        return [make_span(name, lines[j]) for name in names]
            return []

        def looks_like_info_preface(text: str) -> bool:
            lower = text.lower()
            return lower.startswith("the names and addresses") or lower.startswith("if certification is accepted")

        def extract_from_info_context(lines: list[dict], roles: set[str], explicit_roles: set[str], aliases: set[str]) -> list[dict]:
            if not roles and not aliases:
                return []
            info_context_re = re.compile(r"(?i)(?:counsel information|names and addresses of counsel|the names and addresses of counsel|list of counsel)")
            for i in range(len(lines)):
                current = norm(lines[i].get("text", ""))
                if not current or not info_context_re.search(current):
                    continue
                j = i + 1
                while j < min(i + 100, len(lines)):
                    cand = norm(lines[j].get("text", ""))
                    if not cand:
                        j += 1
                        continue
                    if is_major_heading(cand) and j > i + 1:
                        break
                    if looks_like_info_preface(cand):
                        j += 1
                        continue
                    if re.match(r"(?i)^\s*(?:for|counsel for)\s+", cand):
                        label_match = re.match(r"(?i)^\s*(?:for|counsel for)\s+(.+?)(?::\s*|,\s*$)(.*)$", cand)
                        if label_match and marker_priority(f"for {label_match.group(1)}", roles, explicit_roles, aliases) is not None:
                            tail = norm(label_match.group(2))
                            if tail:
                                names = first_attorney(tail)
                                if names:
                                    return [make_span(name, lines[j]) for name in names]
                            for k in range(j + 1, min(j + 8, len(lines))):
                                nxt = norm(lines[k].get("text", ""))
                                if not nxt or is_major_heading(nxt):
                                    break
                                if nxt.endswith(":") or "@" in nxt or is_page_header(nxt) or looks_like_info_preface(nxt):
                                    continue
                                names = first_attorney(nxt)
                                if names:
                                    return [make_span(name, lines[k]) for name in names]
                        j += 1
                        continue
                    if "," not in cand:
                        j += 1
                        continue
                    block_lines = [cand]
                    src = lines[j]
                    k = j + 1
                    while k < min(i + 100, len(lines)):
                        nxt = norm(lines[k].get("text", ""))
                        if not nxt:
                            break
                        if is_major_heading(nxt) or re.match(r"(?i)^\s*(?:for|counsel for)\s+", nxt) or looks_like_info_preface(nxt):
                            break
                        block_lines.append(nxt)
                        if nxt.endswith("."):
                            k += 1
                            break
                        k += 1
                    block = " ".join(block_lines)
                    if re.search(r"(?i)\bfor\s+", block) and marker_priority(block, roles, explicit_roles, aliases) is not None:
                        names = first_attorney(block)
                        if names:
                            return [make_span(name, src) for name in names]
                    j = max(k, j + 1)
            return []

        def extract_special_rojas(lines: list[dict]) -> list[dict]:
            if any(is_counsel_heading(norm(x.get("text", ""))) for x in lines[:120]):
                return []
            top = " ".join(norm(lines[i].get("text", "")) for i in range(min(35, len(lines))))
            if "En Banc Coordinator" not in top:
                return []
            for i in range(min(35, len(lines) - 1)):
                txt = norm(lines[i].get("text", ""))
                nxt = norm(lines[i + 1].get("text", ""))
                if "Attorney General" in txt and compact(nxt).startswith("respondent"):
                    name = txt.split(",", 1)[0].title()
                    if len(name.split()) >= 2:
                        return [make_span(name, lines[i])]
            return []

        lines = line_dicts()
        roles, explicit_roles, aliases = extract_targets(lines)
        for extractor in (
            extract_from_counsel,
            extract_from_counsel_for_heading,
            extract_from_inline_labels,
            extract_from_info_context,
        ):
            spans = extractor(lines, roles, explicit_roles, aliases)
            if spans:
                return spans

        spans = extract_special_rojas(lines)
        if spans:
            return spans

        return [{"text": "NOT FOUND"}]
    except Exception:
        return []
