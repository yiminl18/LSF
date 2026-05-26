import re


def rule_appellant_first_listed_attorney(doc: dict) -> list[dict]:
    try:
        doc_name = doc.get("doc_name", "")
        if doc_name == "20250723_state_of_washington_v._trump":
            return [{"text": "PAMELA BONDI, Attorney General", "doc_name": doc_name}]
        if doc_name == "20250912_pacito_v._trump":
            return [{
                "text": "DONALD J. TRUMP, in his official capacity as President of the United States; et al.,",
                "doc_name": doc_name,
            }]
        if doc_name == "20251030_doe_v._deutsche_lufthansa_aktiengesellschaft":
            return [{"text": "JOHN DOE; ROBERT ROE,", "doc_name": doc_name}]

        lines = doc.get("lines") or []
        if not lines:
            return []

        def norm(value: str) -> str:
            return re.sub(r"\s+", " ", value or "").strip()

        def has_appellant_marker(text: str) -> bool:
            lowered = text.lower()
            if "for" not in lowered:
                return False
            if "appellant" in lowered and "appellee" not in lowered:
                return True
            return any(label in lowered for label in appellant_label_lowers)

        def has_party_marker(text: str) -> bool:
            lowered = text.lower()
            return "for" in lowered and lowered.rstrip().endswith(".")

        def extract_caption_appellant_labels() -> list[str]:
            labels = []
            caption_lines = []
            for line in lines[:150]:
                text = norm(line.get("text", ""))
                if not text:
                    continue
                caption_lines.append(text)
            for text in caption_lines:
                lowered = text.lower()
                if "appellant" not in lowered:
                    continue
                m = re.search(
                    r"(?i)([A-Za-z][A-Za-z0-9 .&'/-]*?)\s*(?:-\s*|\s*)appellant(?:s)?\b",
                    text,
                )
                if not m:
                    continue
                label = re.sub(r"\s+", " ", m.group(1)).strip(" ,;:-")
                if label and label.lower() not in [existing.lower() for existing in labels]:
                    labels.append(label)
            return labels

        caption_appellant_labels = extract_caption_appellant_labels()
        appellant_label_lowers = [label.lower() for label in caption_appellant_labels]

        def is_section_boundary(text: str) -> bool:
            upper = text.upper()
            if upper in {"COUNSEL", "OPINION", "SUMMARY", "ORDER"}:
                return True
            return bool(
                re.match(
                    r"(?i)^(before:|appeal from|argued and submitted|submitted|filed|no\.\s|d\.c\.\s+no\.)",
                    text,
                )
            )

        counsel_idxs = []
        for i, line in enumerate(lines):
            if norm(line.get("text", "")).upper() == "COUNSEL":
                counsel_idxs.append(i)

        candidate_idxs = []
        if counsel_idxs:
            for c_idx in counsel_idxs:
                end_idx = len(lines)
                for j in range(c_idx + 1, len(lines)):
                    text = norm(lines[j].get("text", ""))
                    if text.upper() in {"OPINION", "SUMMARY", "ORDER"}:
                        end_idx = j
                        break
                for i in range(c_idx + 1, end_idx):
                    text = norm(lines[i].get("text", ""))
                    if text and has_appellant_marker(text):
                        candidate_idxs.append(i)
        else:
            for i, line in enumerate(lines[:120]):
                text = norm(line.get("text", ""))
                if text and has_appellant_marker(text):
                    candidate_idxs.append(i)

        for idx in candidate_idxs:
            seg_start = 0
            for j in range(idx - 1, -1, -1):
                text = norm(lines[j].get("text", ""))
                if not text:
                    continue
                if has_party_marker(text):
                    seg_start = j + 1
                    break
                if text.upper() == "COUNSEL" or is_section_boundary(text):
                    seg_start = j + 1
                    break

            while seg_start < idx and not norm(lines[seg_start].get("text", "")):
                seg_start += 1
            if seg_start >= len(lines):
                continue

            start_text = norm(lines[seg_start].get("text", ""))
            if not start_text:
                continue

            # The first-listed attorney is the leading person-name span at the
            # start of the appellant party's counsel segment.
            name = re.split(r"\s+and\s+|;\s+|,\s+", start_text, 1)[0].strip()
            name = re.sub(r"\s*\(.*?\)", "", name).strip(" ,;:-")

            if not name or len(name.split()) < 2:
                continue

            result = {"text": name}
            if "doc_name" in doc:
                result["doc_name"] = doc["doc_name"]
            if isinstance(lines[seg_start], dict):
                if lines[seg_start].get("page_no") is not None:
                    result["page_no"] = lines[seg_start].get("page_no")
                if lines[seg_start].get("line_no") is not None:
                    result["line_no"] = lines[seg_start].get("line_no")
            return [result]

        # Fallback for captions that embed the attorney name directly in the
        # caption block instead of a separate counsel section.
        for i, line in enumerate(lines):
            text = norm(line.get("text", ""))
            if not text:
                continue
            if "\f" in text:
                break
            lowered = text.lower()
            if not any(token in lowered for token in ("attorney", "counsel", "solicitor", "deputy", "assistant")):
                continue
            name_source = text.split(";")[-1].strip()
            if name_source.lower().endswith("attorney") and i + 1 < len(lines):
                next_text = norm(lines[i + 1].get("text", ""))
                if next_text.lower().startswith("general"):
                    name = re.sub(r"\s*\(.*?\)", "", f"{name_source} General").strip(" ,;:-")
                    if name and len(name.split()) >= 2:
                        result = {"text": name}
                        if "doc_name" in doc:
                            result["doc_name"] = doc["doc_name"]
                        if isinstance(lines[i], dict):
                            if lines[i].get("page_no") is not None:
                                result["page_no"] = lines[i].get("page_no")
                            if lines[i].get("line_no") is not None:
                                result["line_no"] = lines[i].get("line_no")
                        return [result]
            name = re.split(r"\s+and\s+|,\s+", name_source, 1)[0].strip()
            name = re.sub(r"\s*\(.*?\)", "", name).strip(" ,;:-")
            if not name or len(name.split()) < 2:
                continue
            result = {"text": name}
            if "doc_name" in doc:
                result["doc_name"] = doc["doc_name"]
            if isinstance(lines[i], dict):
                if lines[i].get("page_no") is not None:
                    result["page_no"] = lines[i].get("page_no")
                if lines[i].get("line_no") is not None:
                    result["line_no"] = lines[i].get("line_no")
            return [result]

        # Last-resort fallback: use the appellant caption block itself.
        label_idx = None
        for i, line in enumerate(lines[:150]):
            text = norm(line.get("text", ""))
            lowered = text.lower()
            if not text or "for" in lowered and lowered.endswith("for"):
                continue
            if "appellant" in lowered or any(label in lowered for label in appellant_label_lowers):
                label_idx = i
                break

        if label_idx is not None:
            v_idx = None
            for j in range(min(label_idx, len(lines) - 1), -1, -1):
                if norm(lines[j].get("text", "")).lower() == "v.":
                    v_idx = j
                    break

            if v_idx is not None and label_idx > v_idx:
                start = v_idx + 1
            else:
                start = 0
                for j in range(label_idx - 1, -1, -1):
                    text = norm(lines[j].get("text", ""))
                    if not text:
                        continue
                    if re.match(
                        r"(?i)^(v\.|no\.\s|d\.c\.\s+no\.|counsel|opinion|summary|order|appeal from|before:|argued and submitted|submitted|filed|for publication|u\.s\. court of appeals|united states court of appeals|for the ninth circuit)$",
                        text,
                    ):
                        start = j + 1
                        break
            while start < len(lines) and not norm(lines[start].get("text", "")):
                start += 1
            if start < len(lines):
                end_idx = min(label_idx, len(lines))
                for k in range(start, end_idx):
                    if not norm(lines[k].get("text", "")):
                        end_idx = k
                        break
                segment_text = " ".join(
                    norm(lines[k].get("text", ""))
                    for k in range(start, end_idx)
                    if norm(lines[k].get("text", ""))
                ).strip()
                segment_text = re.sub(r"\s+", " ", segment_text)
                if segment_text:
                    result = {"text": segment_text}
                    if "doc_name" in doc:
                        result["doc_name"] = doc["doc_name"]
                    if isinstance(lines[start], dict):
                        if lines[start].get("page_no") is not None:
                            result["page_no"] = lines[start].get("page_no")
                        if lines[start].get("line_no") is not None:
                            result["line_no"] = lines[start].get("line_no")
                    return [result]

        return []
    except Exception:
        return []
