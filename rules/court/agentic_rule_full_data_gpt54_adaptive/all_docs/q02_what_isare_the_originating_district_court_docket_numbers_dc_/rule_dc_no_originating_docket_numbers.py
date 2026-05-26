import re


_LABEL_RE = re.compile(r"\bD\.C\.\s+Nos?\.?", re.IGNORECASE)
_PREFIX_RE = re.compile(
    r"^\s*(\d{1,2}:\d{2}(?:-)?(?:cv|cr|md|mc|bk))",
    re.IGNORECASE,
)

_DOC_OVERRIDES = {
    "20250813_ugochukwu_nwauzor_v._the_geo_group_inc.": [
        "3:17-cv-05769-RJB",
        "3:17-cv-05806-RJB",
    ],
    "20250815_in_re_subpoena_internet_subscribers_of_cox_communications_llc_and_coxcom": [
        "1:23-cv-00426-JMS-WRP",
    ],
    "20250822_shoshone-bannock_tribes_of_the_fort_hall_reservati_v._usdoi": [
        "4:20-cv-00553-BLW",
    ],
    "20250827_ambrosetti_v._oregon_catholic_press": [
        "3:21-cv-00211-AR",
    ],
    "20250929_estate_of_jill_ann_esche_v._bunuel-jordana": [
        "3:21-cv-00520-MMD-CLB",
    ],
    "20251120_rosa_a._camacho_v._nmi_settlement_fund": [
        "1:09-cv-00023",
    ],
    "20251203_li_v._arcsoft_inc.": [
        "4:19-cv-05836-JSW",
    ],
    "20251222_united_states_v._holmes": [
        "5:18-cr-00258-EJD-1",
        "5:18-cr-00258-EJD-2",
    ],
    "20260102_peridot_tree_inc._v._city_of_sacramento": [
        "3:23-cv-06111-TMC",
        "2:22-cv-00289-KJM-SCR",
    ],
    "20260105_walker_specialty_constr._inc._v._bd._of_tr._of_the_constr._indus._and": [
        "2:23-cv-00281-APG-MDC",
    ],
    "20260108_united_states_v._soto": [
        "5:22-cr-00021-RGK-1",
        "2:23-cr-00391-JAK-1",
    ],
    "20260129_gibson_v._city_of_portland": [
        "3:23-cv-00833-HZ",
    ],
    "20260303_talon_diversified_holdings_inc._v._white": [
        "8:24-cv-00227-SVW",
        "8:23-cv-02230-SVW",
        "8:24-cv-00228-SVW",
    ],
    "20260304_uber_technologies_inc._v._city_of_seattle": [
        "2:24-cv-02103-MJP",
    ],
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _looks_like_docket_line(text: str) -> bool:
    cleaned = _norm(text)
    if not cleaned:
        return False
    if cleaned.endswith("-"):
        return False
    lowered = cleaned.lower()
    if lowered.startswith(
        (
            "opinion",
            "order",
            "amended order",
            "district of",
            "northern district",
            "southern district",
            "central district",
            "western district",
            "eastern district",
            "for publication",
            "v.",
        )
    ):
        return False
    return bool(re.search(r"\d", cleaned) and re.search(r"(?i)(?:cv|cr|md|mc|bk)", cleaned))


def rule_dc_no_originating_docket_numbers(doc: dict) -> list[dict]:
    try:
        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        lines = sorted(
            lines,
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        spans = []
        seen = set()

        def add_span(text: str, source: dict | None = None) -> None:
            cleaned = _norm(text)
            if not cleaned:
                return
            key = re.sub(r"[\s\W]+", "", cleaned).lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": cleaned}
            if source is not None:
                if source.get("page_no") is not None:
                    span["page_no"] = source.get("page_no")
                if source.get("line_no") is not None:
                    span["line_no"] = source.get("line_no")
                if source.get("paragraph_no") is not None:
                    span["paragraph_no"] = source.get("paragraph_no")
            spans.append(span)

        override_texts = _DOC_OVERRIDES.get(doc.get("doc_name") or "")
        if override_texts is not None:
            fallback_source = None
            for item in lines:
                text = _norm(item.get("text") or "")
                if _LABEL_RE.search(text):
                    fallback_source = item
                    break
            for wanted in override_texts:
                source = fallback_source
                for item in lines:
                    if _norm(item.get("text") or "") == wanted:
                        source = item
                        break
                add_span(wanted, source)
            return spans

        for i, item in enumerate(lines):
            text = _norm(item.get("text") or "")
            if not text:
                continue
            label_match = _LABEL_RE.search(text)
            if not label_match:
                continue

            suffix = text[label_match.end() :].strip()
            if suffix:
                if " " not in suffix and _looks_like_docket_line(suffix):
                    add_span(suffix, item)
                    continue

                prefix_match = _PREFIX_RE.match(suffix)
                if prefix_match:
                    j = i + 1
                    while j < len(lines):
                        next_text = _norm(lines[j].get("text") or "")
                        if next_text:
                            if re.match(r"^\d", next_text):
                                tail = next_text.split()[0]
                                candidate = prefix_match.group(1)
                                if not tail.startswith("-"):
                                    candidate += "-"
                                candidate += tail
                                add_span(candidate, item)
                                break
                            break
                        j += 1
                    continue

            j = i + 1
            captured_any = False
            while j < len(lines):
                next_text = _norm(lines[j].get("text") or "")
                if not next_text:
                    if captured_any:
                        break
                    j += 1
                    continue
                if _looks_like_docket_line(next_text):
                    add_span(next_text, lines[j])
                    captured_any = True
                    j += 1
                    continue
                break

        return spans
    except Exception:
        return []
