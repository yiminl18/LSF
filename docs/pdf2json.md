# `pdf2json.py` — PDF → JSON via Databricks `ai_parse_document`

This document specifies how to implement `pdf2json.py`, a small Python utility
that takes a PDF file as input and returns a structured JSON representation of
its contents using Databricks' `ai_parse_document` function.

`ai_parse_document` is a SQL function in Databricks' **Agent Bricks** suite
(Public Preview, announced November 2025). It parses complex documents — text,
tables, figures, diagrams — into structured output including layout
information, bounding boxes, and AI-generated captions for figures. Because it
is a Databricks-side function, the Python script does not "run" the model
locally; it orchestrates an upload + SQL call against a Databricks workspace.

## What `ai_parse_document` returns

A single call to `ai_parse_document(<binary_pdf>)` produces a structured result
that contains, per the Databricks blog announcement:

- The document's text content
- Layout / reading-order information
- Parsed tables (merged cells and nested structures preserved)
- Figures and diagrams, each with an AI-generated caption
- Spatial metadata (bounding boxes) for citations and validation
- Optional rendered image outputs

`pdf2json.py` will serialize that result as JSON.

## High-level flow

1. Read the local PDF into bytes.
2. Upload the bytes to a Unity Catalog **Volume** (or a workspace path) so the
   Databricks SQL warehouse can read it.
3. Execute a SQL query of the form
   `SELECT ai_parse_document(content) FROM read_files('<volume-path>')`
   against a Databricks SQL warehouse.
4. Fetch the row, normalize the returned struct into plain Python dicts/lists,
   and write the result to `output.json`.
5. (Optional) Clean up the uploaded file from the Volume.

## Requirements

- A Databricks workspace with `ai_parse_document` enabled (Public Preview as of
  the November 11, 2025 launch).
- A **SQL warehouse** (serverless recommended) that the user can query.
- A **Unity Catalog Volume** the user has `WRITE VOLUME` permission on, used as
  a staging location for the PDF.
- A **personal access token** (or OAuth M2M credentials) for authentication.
- Python 3.10+.

### Python dependencies

```
databricks-sql-connector>=3.0.0
databricks-sdk>=0.30.0
```

Install:

```bash
pip install "databricks-sql-connector>=3.0.0" "databricks-sdk>=0.30.0"
```

## Configuration

`pdf2json.py` reads configuration from environment variables so no secrets are
hard-coded:

| Variable | Meaning |
| --- | --- |
| `DATABRICKS_HOST` | e.g. `https://<workspace>.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | Personal access token |
| `DATABRICKS_HTTP_PATH` | SQL warehouse HTTP path, e.g. `/sql/1.0/warehouses/abc123` |
| `DATABRICKS_VOLUME` | Staging volume, e.g. `/Volumes/main/default/pdf_staging` |

## Reference implementation

Save the following as `pdf2json.py` at the repo root.

```python
"""pdf2json.py

Parse a PDF into structured JSON using Databricks' ai_parse_document.

Usage:
    python pdf2json.py <input.pdf> [-o output.json]

The script:
  1. Uploads the local PDF to a Unity Catalog Volume.
  2. Calls ai_parse_document on it via a Databricks SQL warehouse.
  3. Writes the structured parse result to JSON.
  4. Removes the staged file from the Volume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Any

from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient


# ---------- Config ----------

REQUIRED_ENV = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_HTTP_PATH",
    "DATABRICKS_VOLUME",
)


def _load_config() -> dict[str, str]:
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    return {v: os.environ[v] for v in REQUIRED_ENV}


# ---------- Upload ----------

def upload_to_volume(
    client: WorkspaceClient, local_path: Path, volume: str
) -> str:
    """Upload local PDF to a Unity Catalog volume; return its volume path."""
    remote_name = f"{uuid.uuid4().hex}_{local_path.name}"
    remote_path = f"{volume.rstrip('/')}/{remote_name}"
    with local_path.open("rb") as f:
        client.files.upload(remote_path, f, overwrite=True)
    return remote_path


def delete_from_volume(client: WorkspaceClient, remote_path: str) -> None:
    try:
        client.files.delete(remote_path)
    except Exception as e:  # noqa: BLE001
        print(f"warning: failed to delete staged file {remote_path}: {e}",
              file=sys.stderr)


# ---------- Parse ----------

def parse_pdf(
    cfg: dict[str, str], remote_path: str
) -> dict[str, Any]:
    """Run ai_parse_document on the staged file and return the parsed result."""
    query = """
        SELECT ai_parse_document(content) AS parsed
        FROM read_files(:path, format => 'binaryFile')
    """
    with dbsql.connect(
        server_hostname=cfg["DATABRICKS_HOST"].replace("https://", ""),
        http_path=cfg["DATABRICKS_HTTP_PATH"],
        access_token=cfg["DATABRICKS_TOKEN"],
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"path": remote_path})
            row = cur.fetchone()
    if row is None:
        raise RuntimeError("ai_parse_document returned no rows")
    return _to_jsonable(row.parsed)


# ---------- JSON normalization ----------

def _to_jsonable(obj: Any) -> Any:
    """Convert Databricks SQL result types into plain JSON-friendly values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    # Row / struct objects expose asDict() or _asdict()
    if hasattr(obj, "asDict"):
        return _to_jsonable(obj.asDict(recursive=True))
    if hasattr(obj, "_asdict"):
        return _to_jsonable(obj._asdict())
    return str(obj)


# ---------- Entry point ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Parse PDF to JSON via ai_parse_document.")
    ap.add_argument("pdf", type=Path, help="Path to the input PDF.")
    ap.add_argument("-o", "--output", type=Path,
                    help="Output JSON path (default: <pdf>.json).")
    args = ap.parse_args()

    if not args.pdf.is_file():
        raise SystemExit(f"Not a file: {args.pdf}")
    if args.pdf.suffix.lower() != ".pdf":
        print(f"warning: {args.pdf} does not have a .pdf extension",
              file=sys.stderr)

    cfg = _load_config()
    out_path = args.output or args.pdf.with_suffix(".json")

    client = WorkspaceClient(
        host=cfg["DATABRICKS_HOST"], token=cfg["DATABRICKS_TOKEN"]
    )

    remote_path = upload_to_volume(client, args.pdf, cfg["DATABRICKS_VOLUME"])
    try:
        parsed = parse_pdf(cfg, remote_path)
    finally:
        delete_from_volume(client, remote_path)

    out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
```

## Example session

```bash
export DATABRICKS_HOST="https://<workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi..."
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/abc123"
export DATABRICKS_VOLUME="/Volumes/main/default/pdf_staging"

python pdf2json.py treasury_bulletin_2024Q4.pdf -o bulletin.json
```

`bulletin.json` will contain a JSON document roughly shaped like:

```json
{
  "document_metadata": { "num_pages": 42, "...": "..." },
  "pages": [
    {
      "page_number": 1,
      "elements": [
        {
          "type": "text",
          "content": "Treasury Bulletin — December 2024",
          "bbox": [72.0, 90.5, 540.0, 110.2]
        },
        {
          "type": "table",
          "rows": [["...", "..."], ["...", "..."]],
          "bbox": [72.0, 200.0, 540.0, 380.0]
        },
        {
          "type": "figure",
          "caption": "Federal receipts by source, FY2024.",
          "bbox": [72.0, 400.0, 540.0, 620.0]
        }
      ]
    }
  ]
}
```

The exact field names follow whatever `ai_parse_document` returns at the time
of the call — the script preserves the structure verbatim and only converts
SQL types into JSON-friendly Python types.

## Notes and caveats

- The exact SQL signature and result schema for `ai_parse_document` may
  evolve during Public Preview. If a future signature requires named arguments
  (e.g. `ai_parse_document(content, options => ...)`), update the `query`
  string in `parse_pdf`. Consult Databricks' documentation for the current
  schema: <https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document>.
- For batch processing, prefer the Databricks-native pattern from the
  [reference solution](https://github.com/databricks/bundle-examples/tree/main/contrib/job_with_ai_parse_document):
  ingest PDFs into a Delta table with `read_files`, then materialize parsed
  output with a Spark Declarative Pipeline. The Python script above is meant
  for one-off local conversions, not millions-of-documents workloads.
- Costs accrue on the SQL warehouse running the query and on `ai_parse_document`
  itself (priced per page of input).
- The script deletes the staged copy from the Volume on completion. The
  Databricks-side parse result is not persisted anywhere unless you write it
  to a table.

## Sources

- [Introducing the OfficeQA Benchmark](https://www.databricks.com/blog/introducing-officeqa-benchmark-end-to-end-grounded-reasoning)
- [PDFs to Production: Announcing state-of-the-art document intelligence on Databricks](https://www.databricks.com/blog/pdfs-production-announcing-state-art-document-intelligence-databricks)
- [Reference solution: `job_with_ai_parse_document`](https://github.com/databricks/bundle-examples/tree/main/contrib/job_with_ai_parse_document)
