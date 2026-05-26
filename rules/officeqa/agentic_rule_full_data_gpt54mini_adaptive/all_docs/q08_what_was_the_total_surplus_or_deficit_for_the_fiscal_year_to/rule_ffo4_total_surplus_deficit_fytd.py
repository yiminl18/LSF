import re


def rule_ffo4_total_surplus_deficit_fytd(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        if not lines:
            return []

        line_texts = [str(line.get("text", "")) for line in lines]
        normalized = [re.sub(r"\s+", " ", text).strip() for text in line_texts]
        lowered = [text.lower() for text in normalized]

        table_start = None
        for i, text in enumerate(normalized):
            if "total on- and off-budget results and financing of the u.s. government" in lowered[i]:
                table_start = i
                break

        if table_start is None:
            return []

        # This article table is the stable source for the question across the
        # later bulletin layouts. The fiscal-year-to-date figure is the second
        # numeric entry on the total surplus/deficit row.
        row_idx = None
        search_end = min(len(normalized), table_start + 120)
        for i in range(table_start, search_end):
            text = lowered[i]
            if "surplus or deficit" not in text:
                continue
            if "on-budget" in text or "off-budget" in text:
                continue
            row_idx = i
            break

        if row_idx is None:
            return []

        numeric_lines = []
        numeric_values = []
        for j in range(row_idx, min(len(normalized), row_idx + 8)):
            text = normalized[j]
            if not text:
                continue
            # Rows in this table are usually split so that each numeric cell is
            # on its own line. Collect any stand-alone numbers in order.
            matches = re.findall(r"(?<!\w)-?\d[\d,]*", text)
            if not matches:
                continue
            for match in matches:
                numeric_values.append(match)
                numeric_lines.append(j)
            if len(numeric_values) >= 3:
                break

        if len(numeric_values) < 2:
            return []

        value = numeric_values[1]
        value_line = numeric_lines[1]
        return [
            {
                "text": value,
                "page_no": lines[value_line].get("page_no"),
                "line_no": lines[value_line].get("line_no"),
            }
        ]
    except Exception:
        return []
