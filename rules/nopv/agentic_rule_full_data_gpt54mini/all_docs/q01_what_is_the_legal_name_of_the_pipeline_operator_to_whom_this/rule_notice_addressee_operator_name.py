def rule_notice_addressee_operator_name(doc: dict) -> list[dict]:
    """Retrieve the legal operator name from the addressee block at the top of PHMSA notices."""
    try:
        import re

        lines = [s for s in (doc.get("lines") or []) if isinstance(s, dict)]
        if not lines:
            return []

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def low(text: str) -> str:
            return norm(text).lower()

        def is_blank(text: str) -> bool:
            return not norm(text)

        def is_header_or_notice(text: str) -> bool:
            t = low(text)
            return (
                "notice of probable violation" in t
                or "proposed compliance order" in t
                or "pipeline and hazardous materials safety administration" in t
                or "u.s. department of transportation" in t
                or t.startswith("via electronic mail to:")
            )

        def is_cpf(text: str) -> bool:
            return bool(re.search(r"\bCPF\b", norm(text)))

        def is_email(text: str) -> bool:
            return "@" in text

        def looks_like_person_line(text: str) -> bool:
            t = norm(text)
            if not t:
                return False
            return bool(re.match(r"^(Mr|Ms|Mrs|Dr|Hon)\.?\s+[A-Z]", t))

        def looks_like_title_line(text: str) -> bool:
            t = low(text)
            if not t:
                return False
            title_patterns = [
                r"\bpresident\b",
                r"\bvice president\b",
                r"\bchief executive\b",
                r"\bchief operating\b",
                r"\bchief financial\b",
                r"\bceo\b",
                r"\bcfo\b",
                r"\bcoo\b",
                r"\bdirector\b",
                r"\bmanager\b",
                r"\battorney\b",
                r"\bcounsel\b",
                r"\bsecretary\b",
                r"\bchair\b",
                r"\bchairman\b",
                r"\bgeneral counsel\b",
            ]
            return any(re.search(pat, t) for pat in title_patterns)

        def looks_like_address(text: str) -> bool:
            t = norm(text)
            if not t:
                return False
            low_t = t.lower()
            if re.search(r"\b\d{5}(?:-\d{4})?\b", t) and any(ch.isalpha() for ch in t):
                return True
            if re.search(r"\b(po box|p\.o\. box)\b", low_t):
                return True
            if re.search(
                r"\b(street|st\.|road|rd\.|avenue|ave\.|boulevard|blvd\.|lane|ln\.|drive|dr\.|court|ct\.|place|plaza|suite|ste\.|floor|blg|building|highway|hwy\.)\b",
                low_t,
            ):
                return True
            if re.search(r"\b[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", t):
                return True
            if sum(ch.isdigit() for ch in t) >= 3 and any(
                term in low_t for term in ["street", "road", "avenue", "suite", "drive", "way", "lane", "boulevard"]
            ):
                return True
            return False

        def looks_like_org_name(text: str) -> bool:
            t = norm(text)
            if not t:
                return False
            low_t = t.lower()
            if looks_like_address(t) or looks_like_person_line(t) or looks_like_title_line(t):
                return False
            if is_header_or_notice(t) or is_email(t) or is_cpf(t):
                return False
            if t[0].islower():
                return False
            org_markers = [
                "company",
                "corp",
                "corporation",
                "inc",
                "incorporated",
                "llc",
                "l.l.c",
                "lp",
                "l.p",
                "ltd",
                "limited",
                "pipeline",
                "pipelines",
                "energy",
                "transmission",
                "midstream",
                "terminal",
                "terminals",
                "operating",
                "resources",
                "systems",
                "gas",
                "liquid",
                "liquids",
                "petroleum",
            ]
            if any(marker in low_t for marker in org_markers):
                return True
            if len(t.split()) >= 2 and any(ch.isalpha() for ch in t):
                return True
            return False

        # Find the top address block, then select the first organization-like line
        # after the recipient name/title lines and before the postal address/CPF line.
        cpf_idx = None
        for i, span in enumerate(lines):
            if is_cpf(span.get("text") or ""):
                cpf_idx = i
                break
        if cpf_idx is None:
            cpf_idx = len(lines)

        person_idx = None
        for i, span in enumerate(lines[:cpf_idx]):
            if looks_like_person_line(span.get("text") or ""):
                person_idx = i
                break
        title_idx = None
        for i, span in enumerate(lines[:cpf_idx]):
            if looks_like_title_line(span.get("text") or ""):
                title_idx = i
                break

        anchor_idx = title_idx if title_idx is not None else person_idx
        if anchor_idx is None:
            # Fallback: scan the top block directly for the first organization-like line.
            for span in lines[:cpf_idx]:
                text = span.get("text") or ""
                if looks_like_org_name(text):
                    return [{
                        "text": norm(text),
                        **{k: span[k] for k in ("page_no", "line_no") if k in span},
                    }]
            return []

        candidates = []
        for span in lines[anchor_idx + 1:cpf_idx]:
            text = span.get("text") or ""
            if is_blank(text) or is_header_or_notice(text) or is_email(text):
                continue
            if looks_like_person_line(text) or looks_like_title_line(text):
                continue
            if looks_like_address(text):
                break
            if looks_like_org_name(text):
                candidates.append(span)
                continue

        if candidates:
            best = candidates[0]
            return [{
                "text": norm(best.get("text") or ""),
                **{k: best[k] for k in ("page_no", "line_no") if k in best},
            }]

        return []
    except Exception:
        return []
