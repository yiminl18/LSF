"""Driver: Farthest-Point document sampling on FinanceBench single-cluster.

Pipeline:
  1. Merge sampled + unsampled label files → pool of 59 unique docs.
  2. Build 50 span-binned chunks per doc.
  3. Embed all chunks + all 10 questions (cached to embeddings.npz).
  4. For each (doc, query): cosine-similarity curve, contrast-normalised → 50-dim vec.
  5. Concatenate across 10 queries → 500-dim feature v_d per doc.
  6. Run FPS on v_d's with elbow stopping (stop_ratio=0.5, max_K=None).
  7. Write sample_doc_labels.json + unsampled_doc_labels.json + fps_run.json.

See docs/approach/sampling.md for the algorithm spec.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

from models.embedding3small import embed, AZURE_DEPLOYMENT      # noqa: E402
from sampling.fps          import farthest_point_sampling       # noqa: E402

# ── Config (FinanceBench defaults; --dataset overrides) ──────────────────────
SAMPLED_LABELS_FILE   = "data/financebench/sample/single_cluster/random/sample_doc_labels.json"
UNSAMPLED_LABELS_FILE = "data/financebench/sample/single_cluster/random/unsampled_doc_labels.json"
PROCESSING_DIR        = "data/financebench/processing"
OUT_DIR               = Path("data/financebench/sample/single_cluster/fps")

L_BINS     = 50      # per-doc chunks (FPS spec §2 step 3)
STOP_RATIO = 0.5     # elbow rule (FPS spec §5) — won't fire on a single-cluster pool
MAX_K      = 10      # hard cap: top-K most-diverse picks (overridden by --max-K)
SEED       = 0

# Doc-JSON filename patterns we try (matches pipeline._DOC_JSON_CANDIDATES).
_DOC_JSON_CANDIDATES = ("{stem}_reconstructed.json", "{stem}.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_doc(doc_name: str) -> dict:
    for tmpl in _DOC_JSON_CANDIDATES:
        path = Path(PROCESSING_DIR) / tmpl.format(stem=doc_name)
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(
        f"No JSON for {doc_name!r} in {PROCESSING_DIR} (tried "
        + ", ".join(t.format(stem=doc_name) for t in _DOC_JSON_CANDIDATES) + ")"
    )


def _doc_exists(doc_name: str) -> bool:
    for tmpl in _DOC_JSON_CANDIDATES:
        if (Path(PROCESSING_DIR) / tmpl.format(stem=doc_name)).exists():
            return True
    return False


def bin_doc_into_chunks(doc: dict, L: int) -> list[str]:
    """Split doc['texts'] into L contiguous bins (by reading order),
    concatenating each bin's text. Returns list of L strings (may be empty).
    """
    texts = doc.get("texts", [])
    n = len(texts)
    if n == 0:
        return [""] * L
    # Boundary indices for L equal-sized bins (last bin absorbs remainder)
    edges = [int(round(i * n / L)) for i in range(L + 1)]
    bins: list[str] = []
    for a, b in zip(edges[:-1], edges[1:]):
        chunk = " ".join((s.get("text") or "") for s in texts[a:b])
        # Trim — embedding API limit is generous but very long chunks waste tokens
        if len(chunk) > 8000:
            chunk = chunk[:8000]
        bins.append(chunk.strip())
    return bins


def load_or_build_embeddings(
    pool_docs:        list[str],
    questions:        list[str],
    cache_path:       Path,
    embedding_model:  str,
    bin_count:        int,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, np.ndarray]]:
    """Return ({(doc, bin) -> vector}, {question -> vector}).
    Caches to cache_path and only embeds the missing keys on re-run.
    """
    chunk_emb: dict[tuple[str, int], np.ndarray] = {}
    query_emb: dict[str, np.ndarray] = {}

    # Load existing cache if compatible
    if cache_path.exists():
        try:
            cache = np.load(cache_path, allow_pickle=True)
            cached_model = str(cache["embedding_model"])
            cached_bins  = int(cache["bin_count"])
            if cached_model == embedding_model and cached_bins == bin_count:
                ck = cache["chunk_keys"]
                ce = cache["chunk_emb"]
                for k, v in zip(ck, ce):
                    doc, idx = str(k).rsplit("::", 1)
                    chunk_emb[(doc, int(idx))] = v
                qk = cache["query_keys"]
                qe = cache["query_emb"]
                for k, v in zip(qk, qe):
                    query_emb[str(k)] = v
                print(f"  [cache] loaded {len(chunk_emb)} chunks + {len(query_emb)} queries from {cache_path}")
            else:
                print(f"  [cache] model/bin_count mismatch; rebuilding from scratch")
        except Exception as e:
            print(f"  [cache] failed to load ({e}); rebuilding")

    # ── Embed missing chunks ─────────────────────────────────────────────────
    missing_chunks: list[tuple[str, int, str]] = []  # (doc, idx, text)
    for doc_name in pool_docs:
        doc_json = load_doc(doc_name)
        chunks   = bin_doc_into_chunks(doc_json, bin_count)
        for i, chunk_text in enumerate(chunks):
            if (doc_name, i) not in chunk_emb:
                # Fall back to a single space if empty (avoid API errors on empty input)
                missing_chunks.append((doc_name, i, chunk_text or " "))

    print(f"  [embed] {len(missing_chunks)} chunks to embed")
    if missing_chunks:
        t0 = time.time()
        # Batch in groups of 64 to avoid hitting per-request limits
        BATCH = 64
        for off in range(0, len(missing_chunks), BATCH):
            batch = missing_chunks[off:off + BATCH]
            texts = [t for _, _, t in batch]
            vecs  = embed(texts)
            for (doc, idx, _), v in zip(batch, vecs):
                chunk_emb[(doc, idx)] = np.asarray(v, dtype=np.float32)
            if off % (BATCH * 4) == 0:
                print(f"    {off + len(batch)}/{len(missing_chunks)} chunks "
                      f"({time.time() - t0:.0f}s)")
        print(f"  [embed] chunks done in {time.time() - t0:.0f}s")

    # ── Embed missing queries ────────────────────────────────────────────────
    missing_queries = [q for q in questions if q not in query_emb]
    if missing_queries:
        print(f"  [embed] {len(missing_queries)} queries to embed")
        vecs = embed(missing_queries)
        for q, v in zip(missing_queries, vecs):
            query_emb[q] = np.asarray(v, dtype=np.float32)

    # ── Save cache ───────────────────────────────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_keys = np.array([f"{d}::{i}" for (d, i) in chunk_emb])
    chunk_vals = np.stack(list(chunk_emb.values())).astype(np.float32)
    query_keys = np.array(list(query_emb.keys()))
    query_vals = np.stack(list(query_emb.values())).astype(np.float32)
    np.savez_compressed(
        cache_path,
        chunk_keys      = chunk_keys,
        chunk_emb       = chunk_vals,
        query_keys      = query_keys,
        query_emb       = query_vals,
        embedding_model = np.array(embedding_model),
        bin_count       = np.array(bin_count),
    )
    print(f"  [cache] saved → {cache_path}  ({chunk_vals.nbytes / 1e6:.1f} MB chunks)")

    return chunk_emb, query_emb


def build_v_d(
    chunk_emb: dict[tuple[str, int], np.ndarray],
    query_emb: dict[str, np.ndarray],
    pool_docs: list[str],
    questions: list[str],
    L:         int,
) -> np.ndarray:
    """Build (N_docs, L * N_queries) feature matrix per FPS spec §2.

    For each (doc, query):
      1. cosine-similarity sequence of length L
      2. position-normalised (already L-long since we binned)
      3. contrast-normalised (subtract median, divide by IQR)
    Concatenate the per-query vectors → per-doc feature of length L * N_queries.
    """
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        an = np.linalg.norm(a); bn = np.linalg.norm(b)
        if an == 0 or bn == 0: return 0.0
        return float(np.dot(a, b) / (an * bn))

    N, Q = len(pool_docs), len(questions)
    out = np.zeros((N, L * Q), dtype=np.float32)

    for di, doc_name in enumerate(pool_docs):
        for qi, q in enumerate(questions):
            q_vec = query_emb[q]
            s = np.array([
                cos(chunk_emb[(doc_name, i)], q_vec) for i in range(L)
            ], dtype=np.float32)
            # Contrast-normalise: subtract median, divide by IQR
            med = float(np.median(s))
            q25 = float(np.percentile(s, 25))
            q75 = float(np.percentile(s, 75))
            iqr = max(q75 - q25, 1e-6)
            v = (s - med) / iqr
            out[di, qi * L:(qi + 1) * L] = v
    return out


def estimate_separation(V: np.ndarray, pool_docs: list[str]) -> dict:
    """Heuristic δ_intra / δ_inter from the pool's company × form-type clusters."""
    from sklearn.metrics.pairwise import cosine_distances
    # Cluster id from doc name prefix (company) + presence of 10K/10Q/8K/EARNINGS token
    def cluster_id(name: str) -> str:
        parts = name.split("_")
        company = parts[0]
        form = "OTHER"
        for tok in ("10K", "10Q", "8K", "EARNINGS"):
            if any(tok in p for p in parts[1:]):
                form = tok
                break
        return f"{company}::{form}"

    cl = [cluster_id(d) for d in pool_docs]
    D = cosine_distances(V)
    same, diff = [], []
    for i in range(len(pool_docs)):
        for j in range(i + 1, len(pool_docs)):
            (same if cl[i] == cl[j] else diff).append(D[i, j])
    if not same:
        delta_intra = None
    else:
        delta_intra = float(max(same))
    delta_inter = float(min(diff)) if diff else None
    ratio = (delta_inter / delta_intra) if (delta_intra and delta_intra > 0) else None
    return {
        "delta_intra_observed":         delta_intra,
        "delta_inter_observed":         delta_inter,
        "ratio":                        ratio,
        "deterministic_guarantee_holds": ratio is not None and ratio > 2.0,
        "n_same_cluster_pairs":         len(same),
        "n_diff_cluster_pairs":         len(diff),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_dataset_args():
    """Optional CLI: --dataset <name> overrides the FinanceBench defaults.

    Supported datasets:
      financebench (default) - single_cluster legacy paths
      court / nopv / officeqa - read from data/<ds>/all_labels.json + json/ and
                                write the split into --output-dir (else data/<ds>/sample/all_docs/fps).
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="financebench")
    ap.add_argument("--max-K",   type=int, default=None,
                    help="hard cap on |R| (overrides module default)")
    ap.add_argument("--output-dir", default=None,
                    help="where to write sample/unsampled label files (overrides default)")
    ap.add_argument("--labels-file", default=None,
                    help="explicit labels file (only used when dataset has no pre-split)")
    ap.add_argument("--queries-file", default=None)
    ap.add_argument("--doc-dir", default=None,
                    help="explicit dir of reconstructed (texts-schema) doc JSONs; "
                         "overrides the processing/->json/ probe (e.g. data/officeqa/normalized_json)")
    return ap.parse_args()


def main():
    global SAMPLED_LABELS_FILE, UNSAMPLED_LABELS_FILE, PROCESSING_DIR, OUT_DIR, MAX_K

    args = _parse_dataset_args()
    if args.max_K:
        MAX_K = args.max_K

    if args.dataset != "financebench":
        # Generic single-pool flow: read flat all_labels.json + queries.json
        ds = args.dataset
        labels_file  = args.labels_file or f"data/{ds}/all_labels.json"
        queries_file = args.queries_file or f"data/{ds}/queries.json"
        # Explicit --doc-dir wins (same texts-schema docs the rest of the pipeline
        # uses, e.g. officeqa's normalized_json); otherwise probe processing/ then json/.
        if args.doc_dir:
            PROCESSING_DIR = args.doc_dir
        else:
            for sub in ("processing", "json"):
                cand = Path(f"data/{ds}/{sub}")
                if cand.is_dir():
                    PROCESSING_DIR = str(cand); break
            else:
                PROCESSING_DIR = f"data/{ds}/processing"
        OUT_DIR = Path(args.output_dir or f"data/{ds}/sample/all_docs/fps")
        SAMPLED_LABELS_FILE = labels_file
        UNSAMPLED_LABELS_FILE = labels_file  # single source; loader dedupes
        print(f"  [fps] dataset={ds}  labels={labels_file}  processing={PROCESSING_DIR}  out={OUT_DIR}")
    elif args.output_dir:
        OUT_DIR = Path(args.output_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Merge sampled + unsampled label files (single-file datasets resolve to same path → no-op merge)
    print("[1/5] Building pool ...")
    sampled = json.loads(Path(SAMPLED_LABELS_FILE).read_text())
    unsampled = json.loads(Path(UNSAMPLED_LABELS_FILE).read_text()) if UNSAMPLED_LABELS_FILE != SAMPLED_LABELS_FILE else {}
    all_labels: dict[str, dict] = {**unsampled, **sampled}  # sampled overrides on overlap
    pool_docs = sorted(d.replace(".pdf", "").replace(".PDF","") for d in all_labels.keys())
    pool_docs = [d for d in pool_docs if _doc_exists(d)]
    print(f"  pool size: {len(pool_docs)} unique docs (after dedup, with processing JSON)")

    # Question set is the union of question keys across docs (in practice identical)
    qset: list[str] = []
    seen = set()
    for d, qa in all_labels.items():
        for q in qa.keys():
            if q not in seen:
                seen.add(q); qset.append(q)
    print(f"  questions: {len(qset)}")
    for q in qset:
        print(f"    - {q[:90]}")

    # 2. Embed (with cache)
    print(f"[2/5] Embedding (cache={OUT_DIR / 'embeddings.npz'}) ...")
    chunk_emb, query_emb = load_or_build_embeddings(
        pool_docs       = pool_docs,
        questions       = qset,
        cache_path      = OUT_DIR / "embeddings.npz",
        embedding_model = AZURE_DEPLOYMENT,
        bin_count       = L_BINS,
    )

    # 3. Build feature vectors v_d
    print(f"[3/5] Building v_d feature matrix (L={L_BINS} per query × {len(qset)} queries) ...")
    V = build_v_d(chunk_emb, query_emb, pool_docs, qset, L_BINS)
    print(f"  shape: {V.shape}")

    # 4. Validation: δ_intra / δ_inter heuristic
    print("[4/5] Estimating separation (heuristic on company×form clusters) ...")
    sep = estimate_separation(V, pool_docs)
    print(f"  δ_intra={sep['delta_intra_observed']}  δ_inter={sep['delta_inter_observed']}  "
          f"ratio={sep['ratio']}  guarantee_holds={sep['deterministic_guarantee_holds']}")

    # 5. Run FPS
    print(f"[5/5] FPS (stop_ratio={STOP_RATIO}, max_K={MAX_K}, seed={SEED}) ...")
    indices, gaps = farthest_point_sampling(V, stop_ratio=STOP_RATIO, max_K=MAX_K, seed=SEED)
    K = len(indices)
    elbow_at = K - 1   # last picked index (0-based) was where the elbow fired
    gap_ratio_at_elbow = (gaps[-1] / gaps[-2]) if (len(gaps) >= 2 and gaps[-1] and gaps[-2]) else None

    sampled_names    = [pool_docs[i] for i in indices]
    sampled_pdfs     = set(f"{n}.pdf" for n in sampled_names)
    unsampled_names  = [d for d in pool_docs if f"{d}.pdf" not in sampled_pdfs]

    print(f"  K = {K} picked")
    print(f"  gap sequence: {[round(g, 4) if g is not None else None for g in gaps]}")
    print(f"  sampled docs:")
    for n in sampled_names:
        print(f"    - {n}")

    # ── Write outputs ────────────────────────────────────────────────────────
    new_sampled   = {pdf: all_labels[pdf] for pdf in sorted(sampled_pdfs) if pdf in all_labels}
    new_unsampled = {pdf: all_labels[pdf]
                     for pdf in sorted(f"{n}.pdf" for n in unsampled_names)
                     if pdf in all_labels}
    (OUT_DIR / "sample_doc_labels.json").write_text(
        json.dumps(new_sampled, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / "unsampled_doc_labels.json").write_text(
        json.dumps(new_unsampled, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fps_run = {
        "query_set_size":    len(qset),
        "queries":           qset,
        "feature_dim":       int(V.shape[1]),
        "L_bins":            L_BINS,
        "distance":          "cosine_on_contrast_normalised",
        "embedding_model":   AZURE_DEPLOYMENT,
        "pool_size":         len(pool_docs),
        "K_picked":          K,
        "stop_ratio":        STOP_RATIO,
        "max_K":             MAX_K,
        "seed":              SEED,
        "indices":           indices,
        "doc_names":         sampled_names,
        "gaps":              [None if g is None else round(g, 6) for g in gaps],
        "elbow_at":          elbow_at,
        "gap_ratio_at_elbow":(None if gap_ratio_at_elbow is None else round(gap_ratio_at_elbow, 4)),
        "separation_check":  sep,
    }
    (OUT_DIR / "fps_run.json").write_text(
        json.dumps(fps_run, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'sample_doc_labels.json'}    ({K} docs)")
    print(f"  {OUT_DIR / 'unsampled_doc_labels.json'}  ({len(new_unsampled)} docs)")
    print(f"  {OUT_DIR / 'fps_run.json'}")
    print(f"  {OUT_DIR / 'embeddings.npz'}            (cache, reusable)")


if __name__ == "__main__":
    main()
