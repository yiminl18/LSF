#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Processing JSON Module

Builds the document hierarchy tree from raw LSF and Docling data,
and writes a processing JSON into the SHARED processing directory.

Usage:
    python -m core.pipeline.build_processing_json --dataset pdfs --limit 10 --parser docling
"""

import orjson
import argparse
import sys
import gc
import random
from pathlib import Path
import torch
from core.utils.paths import PathManager, PROJECT_ROOT
from core.utils.suffix import strip_required_suffix

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from core.utils.progress import rich_tqdm as tqdm
from core.doc.tree_reconstructor import PatternKnowledgeBase, Node


def _should_preserve_leaf_header(node: Node) -> bool:
    """Preserve MinerU raw header hints / aside_text leaf headers; avoid amplifying Docling header mis-labels."""
    return bool(
        node.metadata.get("raw_header_hint", False)
        or node.metadata.get("is_aside_text", False)
    )


def flatten_tree_to_items(root: Node) -> list:
    """
    Traverse the tree (DFS) to produce a linear list of items, then apply
    linear text span aggregation (Header owns text until next Header).
    """
    items = []

    # 1. Pre-calculate global stats
    total_h1 = 0
    for c in root.children:
        if c.style.get("size", 0) >= 14.0:
            total_h1 += 1

    # 2. DFS Traversal (Stack-based) to build the flat list first
    stack = []
    for i in range(len(root.children) - 1, -1, -1):
        stack.append(
            (root.children[i], -1, i, 1)
        )  # node, parent_id, sibling_idx, depth

    current_h1_count = 0

    while stack:
        node, parent_id, sibling_idx, depth = stack.pop()

        is_h1 = depth == 1
        if is_h1:
            current_h1_count += 1

        total_siblings = (
            len(node.parent.children) if node.parent else len(root.children)
        )

        structure = {
            "level": "Body",
            "level_index": depth,
            "parent_id": parent_id if parent_id != -1 else None,
            "path_text": node.text,
            "depth": depth,
            "h1_index_norm": (current_h1_count - 1) / total_h1 if total_h1 else 0,
            "sibling_index_norm": sibling_idx / total_siblings if total_siblings else 0,
            "is_first_child": (sibling_idx == 0),
            "is_last_child": (sibling_idx == total_siblings - 1),
        }

        # Nodes with children in the tree, or leaf headers preserved from
        # MinerU raw header hints, should be serialized as headers to stay
        # consistent with the tree linkage.
        is_header_in_tree = len(node.children) > 0
        preserve_leaf_header = _should_preserve_leaf_header(node)
        if is_header_in_tree or preserve_leaf_header:
            structure["level"] = f"H{depth}"

        # Set label based on the node's actual role in the tree structure,
        # consistent with structure.level.
        # If the node acts as a header in the tree (has children), label = section_header.
        # Otherwise keep the original label (could be text, table, etc.).
        # If the original label is section_header but the node acts as text in the tree, correct it to text.
        if is_header_in_tree or preserve_leaf_header:
            final_label = "section_header"
        elif node.label == "section_header":
            # Docling labeled as header, but not a header in the tree structure; correct to text
            final_label = "text"
        else:
            # Keep original label (text, table, etc.)
            final_label = node.label

        # Set path_text: header nodes include their own text; text nodes only include the parent path
        if parent_id >= 0:
            parent_path = items[parent_id]["structure"]["path_text"]
            if final_label == "section_header":
                # Header node: includes its own text
                structure["path_text"] = f"{parent_path} | {node.text}"
            else:
                # Text/table node: does not include its own text, only uses parent path
                structure["path_text"] = parent_path
        else:
            # Root level: text nodes without a parent also exclude their own text
            if final_label == "section_header":
                structure["path_text"] = node.text
            else:
                structure["path_text"] = ""

        # Construct item with full metadata
        s = node.style
        current_id = len(items)
        page_no = node.page
        item = {
            "text": node.text,
            "text_span": "",
            "size": s.get("size"),
            "bold": int(s.get("bold", 0)),
            "font": s.get("font"),
            "all_cap": s.get("all_cap", 0),
            "num_st": s.get("num_st", 0),
            "is_center": s.get("is_center", 0),
            "is_underline": s.get("is_underline", 0),
            "label": final_label,
            "page_no": page_no,
            "structure": structure,
        }
        # Record the header's page explicitly for downstream page-based filtering
        if final_label == "section_header":
            item["header_page"] = page_no
        # Table nodes: output additional structural metadata
        if node.metadata.get("table_data"):
            item["table_data"] = node.metadata["table_data"]

        items.append(item)

        # Push children (Reverse)
        for i in range(len(node.children) - 1, -1, -1):
            stack.append((node.children[i], current_id, i, depth + 1))

    # 3. Tree-aware direct-child span aggregation
    # Build a tree from parent_id; each section_header aggregates text from
    # its direct non-header children only.
    # The tree structure naturally bounds the number of direct children per header,
    # so no character limit is needed.
    from collections import defaultdict

    children_map: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(items)):
        pid = items[idx]["structure"].get("parent_id")
        if pid is not None:
            children_map[pid].append(idx)

    for idx in range(len(items)):
        if items[idx]["label"] != "section_header":
            continue
        span_texts: list[str] = []
        for child_idx in children_map.get(idx, []):
            child = items[child_idx]
            if child["label"] == "section_header":
                continue
            text = child.get("text", "")
            if text:
                span_texts.append(text)
        items[idx]["text_span"] = " ".join(span_texts)

    return items


def break_node_references(node: Node):
    """Deeply break circular references to assist Python GC."""
    for child in node.children:
        break_node_references(child)
    node.parent = None
    node.children = []


def _process_single_doc_reconstruct(
    paths: PathManager,
    doc_name: str,
    kb: PatternKnowledgeBase,
    dataset: str,
    llm_provider: str = "azure",
    parser: str = "docling",
) -> bool:
    """Worker function to process a single document."""
    try:
        from core.doc.tree_reconstructor import prepare_initial_tree
        from core.doc.cross_encoder import (
            collect_parent_verification_tasks,
            score_header_content_pairs,
            apply_parent_fixes,
        )

        intermediate_dir = paths.get_intermediate_dir(dataset)
        lsf_path = intermediate_dir / f"{doc_name}_lsf.json"

        output_path = paths.get_reconstructed_json_path(dataset, doc_name)

        if not lsf_path.exists():
            return False

        if parser == "mineru":
            from core.doc.mineru_adapter import convert_mineru_to_docling_format

            mineru_path = intermediate_dir / f"{doc_name}_mineru.json"
            pdf_path = paths.get_data_dir(dataset) / f"{doc_name}.pdf"
            if not mineru_path.exists():
                return False
            mineru_data = orjson.loads(mineru_path.read_bytes())
            docling_data = convert_mineru_to_docling_format(mineru_data, pdf_path)
        else:
            docling_path = intermediate_dir / f"{doc_name}_docling.json"
            if not docling_path.exists():
                return False
            docling_data = orjson.loads(docling_path.read_bytes())

        lsf_words = orjson.loads(lsf_path.read_bytes())

        root, body_style = prepare_initial_tree(
            lsf_words,
            docling_data,
            kb,
            dataset,
            llm_provider,
            parser=parser,
        )
        pairs = collect_parent_verification_tasks(root, body_style)

        if pairs:
            text_pairs = [(p.text, c.text) for p, c in pairs]
            scores = score_header_content_pairs(text_pairs)
            fix_count = apply_parent_fixes(pairs, scores)
            if fix_count > 0:
                print(f"  [CE] Verified {len(pairs)} pairs, fixed {fix_count}")

        # 3. Flatten & Save
        new_texts = flatten_tree_to_items(root)
        output_data = {
            "doc_name": doc_name,
            "origin": docling_data.get("origin", {}),
            "texts": new_texts,
        }
        output_path.write_bytes(
            orjson.dumps(
                output_data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
            )
        )

        # 4. Memory Offload
        break_node_references(root)
        return True
    except Exception as e:
        import traceback

        print(f"Error processing {doc_name}: {e}")
        traceback.print_exc()
        return False


def _resolve_doc_name_from_parser_output(file_path: Path, parser: str) -> str | None:
    """Strictly extract the doc_name from a parser output filename."""
    suffix = "_mineru" if parser == "mineru" else "_docling"
    return strip_required_suffix(file_path.stem, suffix)


def _collect_valid_doc_names(
    parser_files: list[Path],
    gt_dir: Path,
    parser: str,
    include_no_gt: bool,
) -> list[str]:
    """Collect valid doc_names, avoiding incorrect stripping of parser keywords from mid-stem."""
    valid_docs: list[str] = []
    expected_suffix = "_mineru" if parser == "mineru" else "_docling"
    for file_path in parser_files:
        doc_name = _resolve_doc_name_from_parser_output(file_path, parser)
        if doc_name is None:
            print(
                f"WARNING: Skip unexpected parser output filename: "
                f"{file_path.name} (expected stem suffix {expected_suffix})"
            )
            continue
        if include_no_gt or (gt_dir / f"{doc_name}.txt_answers.json").exists():
            valid_docs.append(doc_name)
    return valid_docs


def reconstruct_documents(
    dataset: str = "pdfs",
    limit: int = None,
    experiment: str = "default",
    workers: int = 1,
    seed: int = 42,
    llm_provider: str = "azure",
    include_no_gt: bool = False,
    parser: str = "docling",
):
    """
    Serial Reconstruction with memory safety and orjson speedups.
    """
    variant = parser if parser != "docling" else None
    paths = PathManager(experiment=experiment, processing_variant=variant)
    random.seed(seed)
    intermediate_dir = paths.get_intermediate_dir(dataset)
    processing_dir = paths.get_processing_dir(dataset)
    gt_dir = paths.get_ground_truth_dir(dataset)
    processing_dir.mkdir(parents=True, exist_ok=True)

    kb_path = paths.get_knowledge_base_path(dataset)
    kb = PatternKnowledgeBase(kb_path)

    # Select the glob suffix based on parser
    suffix = "_mineru.json" if parser == "mineru" else "_docling.json"
    all_files = sorted(list(intermediate_dir.glob(f"*{suffix}")))
    valid_docs = _collect_valid_doc_names(
        parser_files=all_files,
        gt_dir=gt_dir,
        parser=parser,
        include_no_gt=include_no_gt,
    )

    selected_docs = valid_docs[:limit] if limit else valid_docs

    print("=== Build Processing JSON (Performance Optimized) [Problem 1 Step 2] ===")
    print(f"Dataset:    {dataset}")
    print(f"Parser:     {parser}")
    print(f"Output:     {processing_dir}")
    print(f"To Process: {len(selected_docs)} docs")
    print(f"GT Filter:  {'OFF' if include_no_gt else 'ON'}")
    print(f"Workers:    {workers} (Serial + Periodic Cache Clear)")
    print(f"Seed:       {seed}")
    print(f"LLM:        {llm_provider}")
    print()

    success_count = 0
    failed_docs = []
    for doc_name in tqdm(selected_docs, desc="Building processing JSON"):
        if _process_single_doc_reconstruct(
            paths, doc_name, kb, dataset, llm_provider, parser=parser
        ):
            success_count += 1
        else:
            failed_docs.append(doc_name)

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n=== Build Processing JSON Complete ===")
    print(f"Success: {success_count}")
    print(f"Failed:  {len(failed_docs)}")
    if failed_docs:
        print(f"Failed docs: {', '.join(failed_docs)}")


def main():
    parser = argparse.ArgumentParser(description="Build Processing JSON")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--limit", type=int, help="Max documents")
    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment ID (unused for output)",
    )
    parser.add_argument("--workers", type=int, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="azure",
        choices=sorted(["azure", "openai", "openrouter"]),
        help="LLM provider for header validation",
    )
    parser.add_argument(
        "--include-no-gt",
        action="store_true",
        help="Include documents without ground truth",
    )
    parser.add_argument(
        "--parser",
        type=str,
        required=True,
        choices=["docling", "mineru"],
        help="Structural parser (docling/mineru)",
    )

    args = parser.parse_args()

    reconstruct_documents(
        dataset=args.dataset,
        limit=args.limit,
        experiment=args.experiment,
        workers=args.workers,
        seed=args.seed,
        llm_provider=args.llm_provider,
        include_no_gt=args.include_no_gt,
        parser=args.parser,
    )


if __name__ == "__main__":
    main()
