"""Batch convert PDFs to reconstructed JSON using pdf2json_docling.py.

Usage:
    python src/batch_pdf2json.py --docs data/court/docs --out data/court/json
    python src/batch_pdf2json.py --docs data/nopv/docs --out data/nopv/json
    python src/batch_pdf2json.py --docs data/officeqa/docs --out data/officeqa/docling_json \
        --sample data/officeqa/sample_docs.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PDF2JSON = _ROOT / "src" / "pdf2json_docling.py"
_PYTHON = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs",   required=True, help="Directory containing PDFs")
    ap.add_argument("--out",    required=True, help="Output directory for JSON files")
    ap.add_argument("--sample", default=None,  help="Optional .txt file listing doc stems to process")
    ap.add_argument("--work-dir", default="/tmp/pdf2json_work")
    args = ap.parse_args()

    docs_dir = Path(args.docs)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        lines = Path(args.sample).read_text().splitlines()
        stems = {Path(l.strip()).stem for l in lines if l.strip()}
        pdfs  = sorted(p for p in docs_dir.glob("*.pdf") if p.stem in stems)
    else:
        pdfs = sorted(docs_dir.glob("*.pdf"))

    total = len(pdfs)
    print(f"docs={docs_dir}  out={out_dir}  total={total}", flush=True)

    ok = skip = err = 0
    t0 = time.time()
    for i, pdf in enumerate(pdfs, 1):
        out_file = out_dir / f"{pdf.stem}.json"
        if out_file.exists():
            skip += 1
            print(f"[{i}/{total}] SKIP  {pdf.stem}", flush=True)
            continue

        print(f"[{i}/{total}] {pdf.stem} ...", end=" ", flush=True)
        t1 = time.time()
        result = subprocess.run(
            [_PYTHON, str(_PDF2JSON), str(pdf), "-o", str(out_file),
             "--work-dir", args.work_dir],
            capture_output=True, text=True,
        )
        lat = round(time.time() - t1, 1)
        if result.returncode == 0:
            ok += 1
            print(f"OK  {lat}s", flush=True)
        else:
            err += 1
            msg = (result.stderr or result.stdout or "")[-200:].strip().replace("\n", " ")
            print(f"ERR {lat}s  {msg}", flush=True)

    elapsed = round(time.time() - t0, 0)
    print(f"\nDone: ok={ok}  skip={skip}  err={err}  total_time={elapsed}s", flush=True)


if __name__ == "__main__":
    main()
