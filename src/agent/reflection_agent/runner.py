"""runner for bundle-based packaging modes."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from agent.rule_runtime.data import (
    QueryPackage,
    _get_label_dir,
    _get_label_filename,
    build_query_package,
    estimate_tokens,
)
from agent.rule_runtime.context import (
    _estimate_tokens_for_model_context,
    _resolve_model_context_limit,
)
from agent.rule_runtime.prompts import fill_prompt, load_prompt_template
from agent.rules.code_rule_json import (
    CodeRule,
    CodeRuleBundle,
    inspect_code_rule_candidates,
    parse_code_rule_bundle,
)
from agent.rules.code_rule_sandbox import execute_locate_region
from agent.rules.range_rule_exec import execute_range_rule
from agent.rules.range_rule_json import (
    RangeRule,
    RangeRuleBundle,
    RetrievalSpec,
    build_range_rule_response_schema,
    parse_range_rule_bundle,
)
from agent.rules.range_rule_scorer import JUDGE_METHOD_NAME, score_retrieved_subset
from core.pipeline.e2e_utils.cache import CachedLLMCaller, DEFAULT_CACHE_DB_PATH
from core.llm.cost import compute_cost

SUPPORTED_PACKAGING_MODES: tuple[str, ...] = (
    "full_bundle_reference",
    "grouped_433",
)
DEFAULT_SWEEP_PACKAGING_MODES: tuple[str, ...] = (
    "full_bundle_reference",
)
SUPPORTED_RULE_MODES: tuple[str, ...] = (
    "json_spec",
    "json_spec_reflect",
    "python_code",
)
_GROUPED_433_SEED = 42
_GROUPED_433_SIZES: tuple[int, ...] = (4, 3, 3)
_DEFAULT_PROJECTED_GENERATION_COST_THRESHOLD_USD = 5.0
_DEFAULT_CONFIG_PATH = "src/agent/config.yaml"
_DEFAULT_OUTPUT_ROOT = "output/agent/financial_baseline_runner"
_DEFAULT_CACHE_DB_PATH = DEFAULT_CACHE_DB_PATH
_RULE_GENERATION_RESPONSE_SCHEMA = build_range_rule_response_schema()
_DEFAULT_REFLECTION_ROUNDS = 2
_DEFAULT_REFLECTION_TOP_FAILURE_EXAMPLES = 2
_DEFAULT_RETRIEVAL_TOO_LARGE_TOKEN_THRESHOLD = 5000
_INFO_NOT_FOUND = "information not found."

# anchor_source -> high-level classification map
_ANCHOR_SOURCE_TO_TYPE: dict[str, str] = {
    "section_line": "section_heading",
    "markdown_heading": "section_heading",
    "substring": "keyword",
    "normalized_line": "keyword",
}

@dataclass(slots=True, frozen=True)
class BundleSpec:
    bundle_index: int
    doc_ids: tuple[str, ...]
    prompt: str
    prompt_tokens: int
    projected_generation_cost_usd: float


def _ensure_request_within_model_context(
    *,
    prompt_text: str,
    max_output_tokens: int,
    llm_provider: str,
    llm_model: str,
    stage_label: str,
) -> None:
    """Runner-local wrapper kept patchable for baseline tests."""
    model_identity, context_limit = _resolve_model_context_limit(
        llm_provider,
        llm_model,
    )
    if context_limit is None:
        return

    prompt_tokens = _estimate_tokens_for_model_context(
        prompt_text,
        llm_provider,
        llm_model,
    )
    total_request_tokens = prompt_tokens + max_output_tokens
    if total_request_tokens > context_limit:
        raise RuntimeError(
            f"{stage_label} request exceeds model context limit: "
            f"model={model_identity} prompt_tokens={prompt_tokens} "
            f"max_output_tokens={max_output_tokens} total={total_request_tokens} > {context_limit}"
        )


@dataclass(slots=True, frozen=True)
class BundleRunResult:
    bundle_index: int
    doc_ids: tuple[str, ...]
    prompt_path: str
    prompt_tokens: int
    projected_generation_cost_usd: float
    actual_generation_cost_usd: float
    cache_hit: bool
    raw_response: str
    parsed_bundle: RangeRuleBundle


@dataclass(slots=True, frozen=True)
class CodeBundleRunResult:
    bundle_index: int
    doc_ids: tuple[str, ...]
    prompt_path: str
    prompt_tokens: int
    projected_generation_cost_usd: float
    actual_generation_cost_usd: float
    cache_hit: bool
    raw_response: str
    parsed_rules: tuple[CodeRule, ...]
    validation_records: tuple[dict[str, Any], ...]


@dataclass(slots=True, frozen=True)
class MergedRule:
    rule: RangeRule
    primary_doc_ids: tuple[str, ...]
    source_bundle_doc_ids_list: tuple[tuple[str, ...], ...]
    source_bundle_prompt_tokens_list: tuple[int, ...]
    source_bundle_projected_costs_usd: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_text": self.rule.rule_text,
            "evidence_basis": self.rule.evidence_basis,
            "retrieval_spec": self.rule.retrieval_spec.to_dict(),
            "primary_doc_ids": list(self.primary_doc_ids),
            "source_bundle_doc_ids_list": [
                list(doc_ids) for doc_ids in self.source_bundle_doc_ids_list
            ],
            "source_bundle_prompt_tokens_list": list(
                self.source_bundle_prompt_tokens_list
            ),
            "source_bundle_projected_costs_usd": list(
                self.source_bundle_projected_costs_usd
            ),
        }


@dataclass(slots=True, frozen=True)
class CodeMergedRule:
    rule: CodeRule
    primary_doc_ids: tuple[str, ...]
    source_bundle_doc_ids_list: tuple[tuple[str, ...], ...]
    source_bundle_prompt_tokens_list: tuple[int, ...]
    source_bundle_projected_costs_usd: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_kind": "code",
            "rule_text": self.rule.rule_text,
            "evidence_basis": self.rule.evidence_basis,
            "code": self.rule.code,
            "primary_doc_ids": list(self.primary_doc_ids),
            "source_bundle_doc_ids_list": [
                list(doc_ids) for doc_ids in self.source_bundle_doc_ids_list
            ],
            "source_bundle_prompt_tokens_list": list(
                self.source_bundle_prompt_tokens_list
            ),
            "source_bundle_projected_costs_usd": list(
                self.source_bundle_projected_costs_usd
            ),
            "sandbox_validation_status": "valid",
        }


@dataclass(slots=True, frozen=True)
class BaselineRunSummary:
    status: str
    query_idx: int
    query_text: str
    packaging_mode: str
    selected_doc_ids: tuple[str, ...]
    doc_count: int
    bundle_count: int
    total_prompt_tokens: int
    projected_generation_cost_usd: float
    actual_generation_cost_usd: float
    rule_count: int
    row_count: int
    avg_accuracy: float | None
    rule_mode: str = "json_spec"
    text_format: str = "markdown"
    best_iteration: int | None = None
    best_single_rule_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "selected_doc_ids": list(self.selected_doc_ids),
        }


@dataclass(slots=True, frozen=True)
class BaselineSweepSummary:
    status: str
    packaging_modes: tuple[str, ...]
    query_indices: tuple[int, ...]
    run_count: int
    total_projected_generation_cost_usd: float
    total_actual_generation_cost_usd: float
    runs: tuple[BaselineRunSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "packaging_modes": list(self.packaging_modes),
            "query_indices": list(self.query_indices),
            "run_count": self.run_count,
            "total_projected_generation_cost_usd": self.total_projected_generation_cost_usd,
            "total_actual_generation_cost_usd": self.total_actual_generation_cost_usd,
            "runs": [run.to_dict() for run in self.runs],
        }


@dataclass(slots=True, frozen=True)
class ReflectionIterationResult:
    iteration_idx: int
    bundle_specs: tuple[BundleSpec, ...]
    bundle_results: tuple[BundleRunResult, ...]
    merged_rules: tuple[MergedRule, ...]
    rows: tuple[dict[str, Any], ...]
    rule_summary: tuple[dict[str, Any], ...]
    bundle_diagnostics: dict[str, Any]
    summary: BaselineRunSummary


def classify_rule_anchor_type(mode: str, anchor_source: str | None) -> str:
    """Map retrieval mode + anchor_source to a high-level anchor type.

    Returns: "page_marker" | "section_heading" | "keyword" | "other"
    """
    if mode == "page":
        return "page_marker"
    if anchor_source is not None:
        return _ANCHOR_SOURCE_TO_TYPE.get(anchor_source, "other")
    return "other"


def _extract_anchor_source(subset: dict[str, Any]) -> str | None:
    """Extract anchor_source from a RetrievedSubset's metadata."""
    metadata = subset.get("metadata", {})
    return metadata.get("anchor_source") or metadata.get("anchor_a_source")


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_llm_provider(config: dict[str, Any], llm_provider: str | None) -> str:
    return llm_provider or str(config.get("llm_provider", "azure"))


def _resolve_llm_model(config: dict[str, Any], llm_model: str | None) -> str:
    if llm_model is not None and llm_model.strip():
        return llm_model
    configured_model = config.get("llm_model")
    if configured_model is not None and str(configured_model).strip():
        return str(configured_model)
    raise ValueError("llm_model must be specified in config or function arguments")


def _resolve_rule_text_format(config: dict[str, Any], rule_mode: str) -> str:
    configured_format = config.get("rule_text_format")
    if configured_format is not None:
        text_format = str(configured_format)
    elif rule_mode in {"python_code", "json_spec_reflect"}:
        text_format = "normalized"
    else:
        text_format = "markdown"

    if text_format not in {"markdown", "normalized"}:
        raise ValueError(f"unsupported rule_text_format={text_format!r}")
    return text_format


def _resolve_rule_mode(config: dict[str, Any]) -> str:
    rule_mode = str(config.get("rule_mode", "json_spec"))
    if rule_mode not in SUPPORTED_RULE_MODES:
        raise ValueError(
            f"unsupported rule_mode={rule_mode!r}; supported={list(SUPPORTED_RULE_MODES)}"
        )
    return rule_mode


def _ensure_bundle_specs_within_model_context(
    bundle_specs: Sequence[BundleSpec],
    *,
    max_output_tokens: int,
    llm_provider: str,
    llm_model: str,
    stage_label: str,
) -> None:
    for spec in bundle_specs:
        _ensure_request_within_model_context(
            prompt_text=spec.prompt,
            max_output_tokens=max_output_tokens,
            llm_provider=llm_provider,
            llm_model=llm_model,
            stage_label=f"{stage_label} bundle={spec.bundle_index}",
        )


def _get_query_indices_from_config(config: dict[str, Any]) -> list[int]:
    return [int(query["query_idx"]) for query in config["queries"]]


def _get_query_doc_ids(config: dict[str, Any], query_idx: int) -> list[str]:
    for query in config["queries"]:
        if int(query["query_idx"]) == query_idx:
            return list(query["documents"])
    raise ValueError(f"q{query_idx} not found in config")


def _load_label_map(
    config: dict[str, Any], query_idx: int
) -> dict[str, dict[str, Any]]:
    label_dir = _get_label_dir(config)
    label_filename = _get_label_filename(config, query_idx)
    label_path = label_dir / label_filename
    data = json.loads(label_path.read_text(encoding="utf-8"))
    return {entry["doc_name"]: entry for entry in data["labels"]}


def _validate_doc_eligibility(
    config: dict[str, Any], query_idx: int, doc_ids: Sequence[str]
) -> None:
    labels = _load_label_map(config, query_idx)
    for doc_id in doc_ids:
        entry = labels.get(doc_id)
        if entry is None:
            raise ValueError(f"{doc_id} missing from q{query_idx} labels")
        if entry.get("absence_label", False):
            raise ValueError(f"{doc_id} has absence_label=True")
        if not str(entry.get("ground_truth", "")).strip():
            raise ValueError(f"{doc_id} has empty ground truth")
        if not entry.get("possible_provenance_nodes"):
            raise ValueError(f"{doc_id} has no provenance node")


def _slice_query_package(
    query_package: QueryPackage, doc_ids: Sequence[str]
) -> QueryPackage:
    doc_map = {document.doc_id: document for document in query_package.documents}
    documents = [doc_map[doc_id] for doc_id in doc_ids]
    return QueryPackage(
        query_idx=query_package.query_idx,
        query_text=query_package.query_text,
        documents=documents,
    )


def _projected_generation_cost(
    prompt_tokens: int,
    max_output_tokens: int,
    llm_provider: str,
    llm_model: str,
) -> float:
    return compute_cost(
        input_tokens=prompt_tokens,
        output_tokens=max_output_tokens,
        llm_provider=llm_provider,
        model=llm_model,
    )


def _build_bundle_doc_lists(
    packaging_mode: str,
    doc_ids: Sequence[str],
) -> list[tuple[str, ...]]:
    ordered_doc_ids = tuple(doc_ids)
    if packaging_mode == "full_bundle_reference":
        return [ordered_doc_ids]
    if packaging_mode == "grouped_433":
        if len(ordered_doc_ids) != sum(_GROUPED_433_SIZES):
            raise ValueError(
                f"grouped_433 baseline requires exactly {sum(_GROUPED_433_SIZES)} documents"
            )
        shuffled_doc_ids = list(ordered_doc_ids)
        random.Random(_GROUPED_433_SEED).shuffle(shuffled_doc_ids)
        bundle_doc_lists: list[tuple[str, ...]] = []
        offset = 0
        for size in _GROUPED_433_SIZES:
            bundle_doc_lists.append(tuple(shuffled_doc_ids[offset : offset + size]))
            offset += size
        return bundle_doc_lists
    raise NotImplementedError(
        f"unsupported packaging_mode={packaging_mode!r}; "
        f"supported={list(SUPPORTED_PACKAGING_MODES)}"
    )


def _build_bundle_specs(
    config: dict[str, Any],
    query_idx: int,
    packaging_mode: str,
    llm_provider: str,
    llm_model: str,
    rule_mode: str | None = None,
) -> tuple[QueryPackage, list[BundleSpec]]:
    selected_doc_ids = _get_query_doc_ids(config, query_idx)
    _validate_doc_eligibility(config, query_idx, selected_doc_ids)

    resolved_rule_mode = rule_mode if rule_mode is not None else _resolve_rule_mode(config)
    text_format = _resolve_rule_text_format(config, resolved_rule_mode)
    full_query_package = build_query_package(
        config, query_idx, selected_doc_ids, text_format=text_format
    )
    bundle_doc_lists = _build_bundle_doc_lists(packaging_mode, selected_doc_ids)
    template = load_prompt_template("range_v1")
    max_output_tokens = int(config.get("max_tokens_output", 4096))

    bundle_specs: list[BundleSpec] = []
    for bundle_index, bundle_doc_ids in enumerate(bundle_doc_lists):
        bundle_query_package = _slice_query_package(full_query_package, bundle_doc_ids)
        prompt = fill_prompt(
            template,
            bundle_query_package,
            anonymize=config.get("anonymize_doc_ids", False),
        )
        prompt_tokens = estimate_tokens(prompt)
        bundle_specs.append(
            BundleSpec(
                bundle_index=bundle_index,
                doc_ids=bundle_doc_ids,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                projected_generation_cost_usd=_projected_generation_cost(
                    prompt_tokens,
                    max_output_tokens,
                    llm_provider,
                    llm_model,
                ),
            )
        )

    return full_query_package, bundle_specs


def _ensure_projected_cost_within_threshold(
    config: dict[str, Any],
    bundle_specs: Sequence[BundleSpec],
) -> None:
    total_projected_cost = sum(
        spec.projected_generation_cost_usd for spec in bundle_specs
    )
    _ensure_projected_cost_value_within_threshold(
        config,
        total_projected_cost,
        label="projected generation cost",
    )


def _ensure_projected_cost_value_within_threshold(
    config: dict[str, Any],
    projected_cost_usd: float,
    *,
    label: str,
) -> None:
    threshold_raw = config.get(
        "projected_generation_cost_threshold_usd",
        _DEFAULT_PROJECTED_GENERATION_COST_THRESHOLD_USD,
    )
    if threshold_raw is None:
        return

    threshold = float(threshold_raw)
    if projected_cost_usd > threshold:
        raise RuntimeError(
            f"{label} exceeds notification threshold: "
            f"${projected_cost_usd:.4f} > ${threshold:.2f}"
        )


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "prompt_dir": output_dir / "bundle_prompts",
        "generation": output_dir / "generation.json",
        "parsed": output_dir / "parsed_rules.json",
        "code_rules": output_dir / "code_rules.json",
        "rows": output_dir / "eval_rows.jsonl",
        "best_iteration": output_dir / "best_iteration.json",
        "best_rules": output_dir / "best_rules.json",
        "summary": output_dir / "summary.json",
    }


def _canonical_retrieval_spec(rule: RangeRule) -> str:
    return json.dumps(rule.retrieval_spec.to_dict(), sort_keys=True, ensure_ascii=False)


def _merge_rules(bundle_results: Sequence[BundleRunResult]) -> list[MergedRule]:
    merged: dict[str, MergedRule] = {}
    for bundle in bundle_results:
        for rule in bundle.parsed_bundle.rules:
            key = _canonical_retrieval_spec(rule)
            existing = merged.get(key)
            if existing is None:
                merged[key] = MergedRule(
                    rule=rule,
                    primary_doc_ids=bundle.doc_ids,
                    source_bundle_doc_ids_list=(bundle.doc_ids,),
                    source_bundle_prompt_tokens_list=(bundle.prompt_tokens,),
                    source_bundle_projected_costs_usd=(
                        bundle.projected_generation_cost_usd,
                    ),
                )
                continue

            if bundle.doc_ids in existing.source_bundle_doc_ids_list:
                continue

            merged[key] = MergedRule(
                rule=existing.rule,
                primary_doc_ids=existing.primary_doc_ids,
                source_bundle_doc_ids_list=existing.source_bundle_doc_ids_list
                + (bundle.doc_ids,),
                source_bundle_prompt_tokens_list=existing.source_bundle_prompt_tokens_list
                + (bundle.prompt_tokens,),
                source_bundle_projected_costs_usd=(
                    existing.source_bundle_projected_costs_usd
                    + (bundle.projected_generation_cost_usd,)
                ),
            )
    return list(merged.values())


def _write_bundle_prompts(
    prompt_dir: Path,
    bundle_specs: Sequence[BundleSpec],
    filename_prefix: str = "bundle",
) -> dict[int, str]:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_paths: dict[int, str] = {}
    for spec in bundle_specs:
        prompt_path = prompt_dir / f"{filename_prefix}_{spec.bundle_index:03d}.txt"
        prompt_path.write_text(spec.prompt, encoding="utf-8")
        prompt_paths[spec.bundle_index] = str(prompt_path)
    return prompt_paths


def _run_bundle_generations(
    bundle_specs: Sequence[BundleSpec],
    prompt_paths: dict[int, str],
    caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    max_output_tokens: int,
) -> list[BundleRunResult]:
    results: list[BundleRunResult] = []
    for spec in bundle_specs:
        call_kwargs: dict[str, Any] = {
            "prompt": spec.prompt,
            "llm_provider": llm_provider,
            "max_tokens": max_output_tokens,
            "response_schema": _RULE_GENERATION_RESPONSE_SCHEMA,
            "model": llm_model,
        }
        generation = caller.call(**call_kwargs)
        parsed_bundle = parse_range_rule_bundle(generation.response)
        results.append(
            BundleRunResult(
                bundle_index=spec.bundle_index,
                doc_ids=spec.doc_ids,
                prompt_path=prompt_paths[spec.bundle_index],
                prompt_tokens=spec.prompt_tokens,
                projected_generation_cost_usd=spec.projected_generation_cost_usd,
                actual_generation_cost_usd=compute_cost(
                    generation.input_tokens,
                    generation.output_tokens,
                    llm_provider,
                    model=llm_model,
                ),
                cache_hit=generation.cache_hit,
                raw_response=generation.response,
                parsed_bundle=parsed_bundle,
            )
        )
    return results


def _build_code_bundle_specs(
    config: dict[str, Any],
    query_package: QueryPackage,
    source_specs: Sequence[BundleSpec],
    llm_provider: str,
    llm_model: str,
) -> list[BundleSpec]:
    template = load_prompt_template("code_scope_v1")
    max_output_tokens = int(config.get("max_tokens_output", 4096))

    code_specs: list[BundleSpec] = []
    for source_spec in source_specs:
        bundle_query_package = _slice_query_package(query_package, source_spec.doc_ids)
        prompt = fill_prompt(
            template,
            bundle_query_package,
            anonymize=config.get("anonymize_doc_ids", False),
        )
        prompt_tokens = estimate_tokens(prompt)
        code_specs.append(
            BundleSpec(
                bundle_index=source_spec.bundle_index,
                doc_ids=source_spec.doc_ids,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                projected_generation_cost_usd=_projected_generation_cost(
                    prompt_tokens,
                    max_output_tokens,
                    llm_provider,
                    llm_model,
                ),
            )
        )
    return code_specs


def _parse_code_rules_safely(raw_response: str, query_idx: int) -> tuple[CodeRule, ...]:
    try:
        return parse_code_rule_bundle(raw_response, query_idx).rules
    except ValueError:
        return ()


def _run_code_bundle_generations(
    query_idx: int,
    bundle_specs: Sequence[BundleSpec],
    prompt_paths: dict[int, str],
    caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    max_output_tokens: int,
) -> list[CodeBundleRunResult]:
    results: list[CodeBundleRunResult] = []
    for spec in bundle_specs:
        call_kwargs: dict[str, Any] = {
            "prompt": spec.prompt,
            "llm_provider": llm_provider,
            "max_tokens": max_output_tokens,
            "model": llm_model,
        }
        generation = caller.call(**call_kwargs)
        validation_records = tuple(inspect_code_rule_candidates(generation.response))
        parsed_rules = _parse_code_rules_safely(generation.response, query_idx)
        results.append(
            CodeBundleRunResult(
                bundle_index=spec.bundle_index,
                doc_ids=spec.doc_ids,
                prompt_path=prompt_paths[spec.bundle_index],
                prompt_tokens=spec.prompt_tokens,
                projected_generation_cost_usd=spec.projected_generation_cost_usd,
                actual_generation_cost_usd=compute_cost(
                    generation.input_tokens,
                    generation.output_tokens,
                    llm_provider,
                    model=llm_model,
                ),
                cache_hit=generation.cache_hit,
                raw_response=generation.response,
                parsed_rules=parsed_rules,
                validation_records=validation_records,
            )
        )
    return results


def _canonical_code_rule(rule: CodeRule) -> str:
    return json.dumps(
        {"rule_kind": "code", "code": rule.code},
        sort_keys=True,
        ensure_ascii=False,
    )


def _merge_code_rules(
    code_results: Sequence[CodeBundleRunResult],
) -> list[CodeMergedRule]:
    merged: dict[str, CodeMergedRule] = {}
    for bundle in code_results:
        for rule in bundle.parsed_rules:
            key = _canonical_code_rule(rule)
            existing = merged.get(key)
            if existing is None:
                merged[key] = CodeMergedRule(
                    rule=rule,
                    primary_doc_ids=bundle.doc_ids,
                    source_bundle_doc_ids_list=(bundle.doc_ids,),
                    source_bundle_prompt_tokens_list=(bundle.prompt_tokens,),
                    source_bundle_projected_costs_usd=(
                        bundle.projected_generation_cost_usd,
                    ),
                )
                continue

            if bundle.doc_ids in existing.source_bundle_doc_ids_list:
                continue

            merged[key] = CodeMergedRule(
                rule=existing.rule,
                primary_doc_ids=existing.primary_doc_ids,
                source_bundle_doc_ids_list=existing.source_bundle_doc_ids_list
                + (bundle.doc_ids,),
                source_bundle_prompt_tokens_list=existing.source_bundle_prompt_tokens_list
                + (bundle.prompt_tokens,),
                source_bundle_projected_costs_usd=(
                    existing.source_bundle_projected_costs_usd
                    + (bundle.projected_generation_cost_usd,)
                ),
            )
    return list(merged.values())


def _validate_bundle_query_indices(
    bundle_results: Sequence[BundleRunResult], query_idx: int
) -> None:
    for bundle in bundle_results:
        parsed_query_idx = bundle.parsed_bundle.query_idx
        if parsed_query_idx != query_idx:
            raise ValueError(
                f"parsed query_idx={parsed_query_idx} != expected q{query_idx} "
                f"for bundle {bundle.bundle_index}"
            )


def _generation_payload(
    query_package: QueryPackage,
    packaging_mode: str,
    bundle_results: Sequence[BundleRunResult],
) -> dict[str, Any]:
    return {
        "query_idx": query_package.query_idx,
        "query_text": query_package.query_text,
        "packaging_mode": packaging_mode,
        "bundle_generations": [
            {
                "bundle_index": bundle.bundle_index,
                "doc_ids": list(bundle.doc_ids),
                "prompt_path": bundle.prompt_path,
                "prompt_tokens": bundle.prompt_tokens,
                "projected_generation_cost_usd": bundle.projected_generation_cost_usd,
                "actual_generation_cost_usd": bundle.actual_generation_cost_usd,
                "cache_hit": bundle.cache_hit,
                "raw_response": bundle.raw_response,
            }
            for bundle in bundle_results
        ],
    }


def _parsed_payload(
    query_idx: int,
    packaging_mode: str,
    bundle_results: Sequence[BundleRunResult],
    merged_rules: Sequence[MergedRule],
) -> dict[str, Any]:
    return {
        "query_idx": query_idx,
        "packaging_mode": packaging_mode,
        "bundle_rules": [
            {
                "bundle_index": bundle.bundle_index,
                "doc_ids": list(bundle.doc_ids),
                "parsed_bundle": bundle.parsed_bundle.to_dict(),
            }
            for bundle in bundle_results
        ],
        "merged_rules": [rule.to_dict() for rule in merged_rules],
    }


def _code_generation_payload(
    query_package: QueryPackage,
    packaging_mode: str,
    code_results: Sequence[CodeBundleRunResult],
) -> dict[str, Any]:
    return {
        "query_idx": query_package.query_idx,
        "query_text": query_package.query_text,
        "packaging_mode": packaging_mode,
        "rule_mode": "python_code",
        "bundle_generations": [
            {
                "bundle_index": bundle.bundle_index,
                "doc_ids": list(bundle.doc_ids),
                "prompt_path": bundle.prompt_path,
                "prompt_tokens": bundle.prompt_tokens,
                "projected_generation_cost_usd": bundle.projected_generation_cost_usd,
                "actual_generation_cost_usd": bundle.actual_generation_cost_usd,
                "cache_hit": bundle.cache_hit,
                "raw_response": bundle.raw_response,
            }
            for bundle in code_results
        ],
    }


def _code_parsed_payload(
    query_idx: int,
    packaging_mode: str,
    code_results: Sequence[CodeBundleRunResult],
    merged_rules: Sequence[CodeMergedRule],
) -> dict[str, Any]:
    return {
        "query_idx": query_idx,
        "packaging_mode": packaging_mode,
        "rule_mode": "python_code",
        "bundle_rules": [
            {
                "bundle_index": bundle.bundle_index,
                "doc_ids": list(bundle.doc_ids),
                "parsed_bundle": {
                    "query_idx": query_idx,
                    "rules": [
                        {**rule.to_dict(), "sandbox_validation_status": "valid"}
                        for rule in bundle.parsed_rules
                    ],
                },
                "validation_records": list(bundle.validation_records),
            }
            for bundle in code_results
        ],
        "merged_rules": [rule.to_dict() for rule in merged_rules],
    }


def _rule_key_from_payload(retrieval_spec: dict[str, Any]) -> str:
    return json.dumps(retrieval_spec, sort_keys=True, ensure_ascii=False)


def _subset_text_from_payload(subset: dict[str, Any]) -> str:
    return "\n\n".join(span["text"] for span in subset.get("spans", []))


def _truncate_text(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _normalize_answer_for_vote(answer: str) -> str:
    normalized = " ".join(answer.split()).strip().rstrip(".;:,")
    return normalized.casefold()


def _is_information_not_found(answer: str | None) -> bool:
    if answer is None:
        return False
    return answer.strip().casefold() == _INFO_NOT_FOUND


def _classify_rule_blocker(
    subset: dict[str, Any],
    generated_answer: str | None,
    judge_result: Any,
    subset_tokens: int,
    retrieval_too_large_token_threshold: int,
) -> str | None:
    if not subset.get("matched", False):
        return str(subset.get("metadata", {}).get("reason", "retrieval_unmatched"))
    if subset_tokens <= 0:
        return "empty_span"
    if judge_result is True:
        return None
    if subset_tokens >= retrieval_too_large_token_threshold:
        return "retrieval_too_large"
    if _is_information_not_found(generated_answer):
        return "gen_not_found"
    return "judge_false"


def _rule_failure_breakdown(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for row in rows:
        blocker = row.get("blocker")
        if blocker is None:
            continue
        breakdown[blocker] = breakdown.get(blocker, 0) + 1
    return breakdown


def _derive_rule_diagnostics(summary: dict[str, Any]) -> list[str]:
    diagnostics: list[str] = []
    failures = summary["failure_breakdown"]
    if failures.get("anchor_not_found", 0) or failures.get("anchor_pair_not_found", 0):
        diagnostics.append(
            "anchor may be unstable or the wrong section boundary is selected"
        )
    if failures.get("page_not_found", 0):
        diagnostics.append("page restriction looks too brittle across documents")
    if failures.get("retrieval_too_large", 0):
        diagnostics.append(
            "retrieved regions are too large; tighten boundaries, reduce max_chars, or replace a broad fallback with a more precise rule"
        )
    if failures.get("gen_not_found", 0):
        diagnostics.append(
            "rule often matches but generation still returns Information not found; the window may be broad, shifted, or miss a boundary-adjacent key span"
        )
    if failures.get("judge_false", 0):
        diagnostics.append(
            "rule matches but the extracted answer is wrong; direction or boundary choice may be off, or a broad rule needs a tighter companion"
        )
    if summary["doc_coverage"] < 1.0 and not diagnostics:
        diagnostics.append("coverage is incomplete on the sample documents")
    if not diagnostics and summary["on_sample_accuracy"] < 1.0:
        diagnostics.append("rule is partially effective but still misses some samples")
    return diagnostics


def _summarize_rules(
    merged_rules: Sequence[MergedRule],
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rule_key = _rule_key_from_payload(row["retrieval_spec"])
        rows_by_key.setdefault(rule_key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for rule_index, merged_rule in enumerate(merged_rules):
        rule_key = _canonical_retrieval_spec(merged_rule.rule)
        rule_rows = rows_by_key.get(rule_key, [])
        matched_rows = [row for row in rule_rows if row["retrieved_subset"]["matched"]]
        judged_rows = [
            row for row in rule_rows if isinstance(row["judge_result"], bool)
        ]
        accuracy_sum = sum(
            1.0 if row["judge_result"] is True else 0.0 for row in rule_rows
        )
        matched_accuracy_sum = sum(
            1.0 if row["judge_result"] is True else 0.0 for row in judged_rows
        )
        subset_tokens = [row["retrieved_subset_tokens"] for row in matched_rows]
        subset_chars = [row["retrieved_subset_chars"] for row in matched_rows]

        retrieval_mode = merged_rule.rule.retrieval_spec.mode
        # Use the most common anchor_source among matched rows as the rule's representative value
        anchor_sources = [
            row.get("anchor_source") for row in matched_rows
            if row.get("anchor_source") is not None
        ]
        dominant_anchor_source = (
            max(set(anchor_sources), key=anchor_sources.count)
            if anchor_sources
            else None
        )
        anchor_type = classify_rule_anchor_type(retrieval_mode, dominant_anchor_source)

        summary = {
            "rule_index": rule_index,
            "rule_key": rule_key,
            "rule_text": merged_rule.rule.rule_text,
            "evidence_basis": merged_rule.rule.evidence_basis,
            "retrieval_spec": merged_rule.rule.retrieval_spec.to_dict(),
            "retrieval_mode": retrieval_mode,
            "anchor_type": anchor_type,
            "doc_coverage": (len(matched_rows) / len(rule_rows) if rule_rows else 0.0),
            "on_sample_accuracy": (accuracy_sum / len(rule_rows) if rule_rows else 0.0),
            "judge_accuracy_on_matched": (
                matched_accuracy_sum / len(judged_rows) if judged_rows else None
            ),
            "avg_subset_tokens": (
                sum(subset_tokens) / len(subset_tokens) if subset_tokens else None
            ),
            "max_subset_tokens": max(subset_tokens) if subset_tokens else None,
            "avg_subset_chars": (
                sum(subset_chars) / len(subset_chars) if subset_chars else None
            ),
            "max_subset_chars": max(subset_chars) if subset_chars else None,
            "failure_breakdown": _rule_failure_breakdown(rule_rows),
            "matched_doc_ids": [row["doc_id"] for row in matched_rows],
            "successful_doc_ids": [
                row["doc_id"] for row in rule_rows if row["judge_result"] is True
            ],
        }
        summary["diagnostics"] = _derive_rule_diagnostics(summary)
        summaries.append(summary)
    return summaries


def _summarize_code_rules(
    merged_rules: Sequence[CodeMergedRule],
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_index.setdefault(int(row["rule_index"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for rule_index, merged_rule in enumerate(merged_rules):
        rule_rows = rows_by_index.get(rule_index, [])
        matched_rows = [row for row in rule_rows if row["retrieved_subset"]["matched"]]
        judged_rows = [
            row for row in rule_rows if isinstance(row["judge_result"], bool)
        ]
        accuracy_sum = sum(
            1.0 if row["judge_result"] is True else 0.0 for row in rule_rows
        )
        matched_accuracy_sum = sum(
            1.0 if row["judge_result"] is True else 0.0 for row in judged_rows
        )
        subset_tokens = [row["retrieved_subset_tokens"] for row in matched_rows]
        subset_chars = [row["retrieved_subset_chars"] for row in matched_rows]
        retrieval_spec = {"rule_kind": "code", "code": merged_rule.rule.code}

        summary = {
            "rule_index": rule_index,
            "rule_kind": "code",
            "rule_key": _canonical_code_rule(merged_rule.rule),
            "rule_text": merged_rule.rule.rule_text,
            "evidence_basis": merged_rule.rule.evidence_basis,
            "code": merged_rule.rule.code,
            "retrieval_spec": retrieval_spec,
            "retrieval_mode": "python_code",
            "anchor_type": "code",
            "sandbox_validation_status": "valid",
            "doc_coverage": (len(matched_rows) / len(rule_rows) if rule_rows else 0.0),
            "on_sample_accuracy": (accuracy_sum / len(rule_rows) if rule_rows else 0.0),
            "judge_accuracy_on_matched": (
                matched_accuracy_sum / len(judged_rows) if judged_rows else None
            ),
            "avg_subset_tokens": (
                sum(subset_tokens) / len(subset_tokens) if subset_tokens else None
            ),
            "max_subset_tokens": max(subset_tokens) if subset_tokens else None,
            "avg_subset_chars": (
                sum(subset_chars) / len(subset_chars) if subset_chars else None
            ),
            "max_subset_chars": max(subset_chars) if subset_chars else None,
            "failure_breakdown": _rule_failure_breakdown(rule_rows),
            "matched_doc_ids": [row["doc_id"] for row in matched_rows],
            "successful_doc_ids": [
                row["doc_id"] for row in rule_rows if row["judge_result"] is True
            ],
        }
        summary["diagnostics"] = _derive_rule_diagnostics(summary)
        summaries.append(summary)
    return summaries


def _build_code_cross_doc_eval(
    merged_rules: Sequence[CodeMergedRule],
    rule_summary: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_index.setdefault(int(row["rule_index"]), []).append(row)

    cross_doc_eval: list[dict[str, Any]] = []
    for summary in rule_summary:
        rule_index = int(summary["rule_index"])
        rule_rows = rows_by_index.get(rule_index, [])
        accuracy = float(summary["on_sample_accuracy"])
        coverage = float(summary["doc_coverage"])
        eval_cost = sum(float(row.get("actual_cost_usd") or 0.0) for row in rule_rows)
        matched_doc_ids = list(summary["matched_doc_ids"])
        success_doc_ids = list(summary["successful_doc_ids"])
        cross_doc_eval.append(
            {
                "rule_index": rule_index,
                "rule_kind": "code",
                "rule_text": summary["rule_text"],
                "code": merged_rules[rule_index].rule.code,
                "coverage": coverage,
                "accuracy": accuracy,
                "score": round(coverage * accuracy, 6),
                "matched_count": len(matched_doc_ids),
                "success_count": len(success_doc_ids),
                "evaluated_docs": len(rule_rows),
                "total_docs": len(rule_rows),
                "budget_truncated": False,
                "skipped_due_to_budget": False,
                "eval_cost_usd": round(eval_cost, 6),
                "success_doc_ids": success_doc_ids,
                "matched_doc_ids": matched_doc_ids,
                "sandbox_validation_status": "valid",
            }
        )
    return cross_doc_eval


def _rejected_code_validation_records(
    code_results: Sequence[CodeBundleRunResult],
) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    for bundle in code_results:
        for record in bundle.validation_records:
            if record.get("sandbox_validation_status") != "rejected_ast":
                continue
            rejected.append(
                {
                    "bundle_index": bundle.bundle_index,
                    "doc_ids": list(bundle.doc_ids),
                    **record,
                }
            )
    return rejected


def _summarize_code_run(
    query_package: QueryPackage,
    packaging_mode: str,
    bundle_specs: Sequence[BundleSpec],
    code_results: Sequence[CodeBundleRunResult],
    merged_rules: Sequence[CodeMergedRule],
    rows: Sequence[dict[str, Any]],
) -> BaselineRunSummary:
    avg_accuracy = (
        sum(1.0 if row["judge_result"] is True else 0.0 for row in rows) / len(rows)
        if rows
        else None
    )
    return BaselineRunSummary(
        status="ok",
        query_idx=query_package.query_idx,
        query_text=query_package.query_text,
        packaging_mode=packaging_mode,
        selected_doc_ids=tuple(document.doc_id for document in query_package.documents),
        doc_count=len(query_package.documents),
        bundle_count=len(bundle_specs),
        total_prompt_tokens=sum(spec.prompt_tokens for spec in bundle_specs),
        projected_generation_cost_usd=sum(
            spec.projected_generation_cost_usd for spec in bundle_specs
        ),
        actual_generation_cost_usd=sum(
            result.actual_generation_cost_usd for result in code_results
        ),
        rule_count=len(merged_rules),
        row_count=len(rows),
        avg_accuracy=avg_accuracy,
    )


def _code_best_rules_payload(
    query_package: QueryPackage,
    packaging_mode: str,
    merged_rules: Sequence[CodeMergedRule],
    rule_summary: Sequence[dict[str, Any]],
    cross_doc_eval: Sequence[dict[str, Any]],
    bundle_diagnostics: dict[str, Any],
    rejected_rules: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "query_idx": query_package.query_idx,
        "packaging_mode": packaging_mode,
        "rule_mode": "python_code",
        "merged_rules": [rule.to_dict() for rule in merged_rules],
        "rule_summary": list(rule_summary),
        "cross_doc_eval": list(cross_doc_eval),
        "bundle_diagnostics": bundle_diagnostics,
        "rejected_rules": list(rejected_rules),
    }


def _build_rule_examples(
    rows: Sequence[dict[str, Any]],
    top_failure_examples: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    examples_by_key: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        rule_key = _rule_key_from_payload(row["retrieval_spec"])
        examples = examples_by_key.setdefault(
            rule_key, {"failures": [], "successes": []}
        )
        subset_text = _subset_text_from_payload(row["retrieved_subset"])
        example = {
            "doc_id": row["doc_id"],
            "matched": row["retrieved_subset"]["matched"],
            "blocker": row["blocker"],
            "retrieved_subset_chars": row["retrieved_subset_chars"],
            "retrieved_subset_tokens": row["retrieved_subset_tokens"],
            "generated_answer": row["generated_answer"],
            "judge_result": row["judge_result"],
            "ground_truth": row["ground_truth"],
            "retrieved_subset_preview": _truncate_text(subset_text),
        }
        if (
            row["blocker"] is not None
            and len(examples["failures"]) < top_failure_examples
        ):
            examples["failures"].append(example)
        elif row["judge_result"] is True and len(examples["successes"]) < 1:
            examples["successes"].append(example)
    return examples_by_key


def _best_single_rule(rule_summary: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rule_summary:
        return None

    return _rank_rule_summaries(rule_summary)[0]


def _rank_rule_summaries(
    rule_summary: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    def _sort_key(summary: dict[str, Any]) -> tuple[float, float, int]:
        avg_subset_tokens = summary["avg_subset_tokens"]
        return (
            -float(summary["on_sample_accuracy"]),
            avg_subset_tokens if avg_subset_tokens is not None else float("inf"),
            int(summary["rule_index"]),
        )

    return sorted(rule_summary, key=_sort_key)


def _build_bundle_diagnostics(
    rule_summary: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute bundle-level diagnostics consumed downstream.

    Only the top rule's on-sample accuracy is used: holdout reports read it as
    `best_single_rule_accuracy`. Previous iterations computed an oracle/union
    upper bound and majority-vote accuracy, but those fields were never read
    outside this file.
    """
    ranked_rule_summary = _rank_rule_summaries(rule_summary)
    best_single = ranked_rule_summary[0] if ranked_rule_summary else None
    best_single_rule_accuracy = (
        float(best_single["on_sample_accuracy"]) if best_single is not None else None
    )
    return {"best_single_rule_accuracy": best_single_rule_accuracy}


def _with_bundle_diagnostics(
    summary: BaselineRunSummary,
    bundle_diagnostics: dict[str, Any],
) -> BaselineRunSummary:
    return replace(
        summary,
        best_single_rule_accuracy=bundle_diagnostics.get("best_single_rule_accuracy"),
    )


def _iteration_selection_metrics(
    iteration: ReflectionIterationResult,
) -> dict[str, Any]:
    matched_tokens = [
        row["retrieved_subset_tokens"]
        for row in iteration.rows
        if row["retrieved_subset"]["matched"]
    ]
    return {
        "iteration_idx": iteration.iteration_idx,
        "on_sample_accuracy": iteration.summary.avg_accuracy or 0.0,
        "avg_subset_tokens": (
            sum(matched_tokens) / len(matched_tokens) if matched_tokens else None
        ),
        "rule_count": len(iteration.merged_rules),
    }


def _is_better_iteration(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    candidate_accuracy = float(candidate["on_sample_accuracy"])
    incumbent_accuracy = float(incumbent["on_sample_accuracy"])
    if candidate_accuracy != incumbent_accuracy:
        return bool(candidate_accuracy > incumbent_accuracy)

    candidate_tokens = candidate["avg_subset_tokens"]
    incumbent_tokens = incumbent["avg_subset_tokens"]
    candidate_token_value = (
        candidate_tokens if candidate_tokens is not None else float("inf")
    )
    incumbent_token_value = (
        incumbent_tokens if incumbent_tokens is not None else float("inf")
    )
    if candidate_token_value != incumbent_token_value:
        return bool(candidate_token_value < incumbent_token_value)

    return bool(candidate["rule_count"] < incumbent["rule_count"])


def _select_best_iteration(
    iterations: Sequence[ReflectionIterationResult],
) -> ReflectionIterationResult:
    best_iteration = iterations[0]
    best_metrics = _iteration_selection_metrics(best_iteration)
    for iteration in iterations[1:]:
        metrics = _iteration_selection_metrics(iteration)
        if _is_better_iteration(metrics, best_metrics):
            best_iteration = iteration
            best_metrics = metrics
    return best_iteration


def _filter_rule_summaries(
    bundle: BundleRunResult,
    rule_summary: Sequence[dict[str, Any]],
    examples_by_key: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bundle_rule_keys = {
        _canonical_retrieval_spec(rule) for rule in bundle.parsed_bundle.rules
    }
    filtered_summary = [
        summary for summary in rule_summary if summary["rule_key"] in bundle_rule_keys
    ]
    failure_examples: list[dict[str, Any]] = []
    success_examples: list[dict[str, Any]] = []
    for summary in filtered_summary:
        examples = examples_by_key.get(summary["rule_key"], {})
        failure_examples.extend(examples.get("failures", []))
        success_examples.extend(examples.get("successes", []))
    return filtered_summary, failure_examples, success_examples


def _combine_bundle_prompts(bundle_specs: Sequence[BundleSpec]) -> str:
    sections: list[str] = []
    for spec in bundle_specs:
        sections.append(
            f"===== bundle_{spec.bundle_index:03d} ({', '.join(spec.doc_ids)}) =====\n{spec.prompt}"
        )
    return "\n\n".join(sections) + "\n"


def _display_doc_id_map(documents: Sequence[Any], anonymize: bool) -> dict[str, str]:
    if not anonymize:
        return {str(document.doc_id): str(document.doc_id) for document in documents}
    return {
        str(document.doc_id): f"Document {chr(ord('A') + idx)}"
        for idx, document in enumerate(documents)
    }


def _apply_doc_id_display_map(value: Any, display_map: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _apply_doc_id_display_map(item, display_map)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_apply_doc_id_display_map(item, display_map) for item in value]
    if isinstance(value, tuple):
        return [_apply_doc_id_display_map(item, display_map) for item in value]
    if isinstance(value, str):
        return display_map.get(value, value)
    return value


def _with_summary_context(
    summary: BaselineRunSummary,
    rule_mode: str,
    text_format: str,
    best_iteration: int | None = None,
    total_prompt_tokens: int | None = None,
    total_projected_generation_cost_usd: float | None = None,
    total_actual_generation_cost_usd: float | None = None,
) -> BaselineRunSummary:
    return replace(
        summary,
        rule_mode=rule_mode,
        text_format=text_format,
        best_iteration=best_iteration,
        total_prompt_tokens=(
            summary.total_prompt_tokens
            if total_prompt_tokens is None
            else total_prompt_tokens
        ),
        projected_generation_cost_usd=(
            summary.projected_generation_cost_usd
            if total_projected_generation_cost_usd is None
            else total_projected_generation_cost_usd
        ),
        actual_generation_cost_usd=(
            summary.actual_generation_cost_usd
            if total_actual_generation_cost_usd is None
            else total_actual_generation_cost_usd
        ),
    )


def _evaluate_rules(
    query_package: QueryPackage,
    packaging_mode: str,
    merged_rules: Sequence[MergedRule],
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    retrieval_too_large_token_threshold: int,
    llm_model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for merged_rule in merged_rules:
        primary_doc_ids = list(merged_rule.primary_doc_ids)
        source_bundle_doc_ids_list = [
            list(doc_ids) for doc_ids in merged_rule.source_bundle_doc_ids_list
        ]
        source_bundle_prompt_tokens_list = list(
            merged_rule.source_bundle_prompt_tokens_list
        )
        source_bundle_projected_costs_usd = list(
            merged_rule.source_bundle_projected_costs_usd
        )

        for document in query_package.documents:
            subset = execute_range_rule(
                merged_rule.rule, document.markdown_text
            ).to_dict()
            subset_text = _subset_text_from_payload(subset)
            subset_chars = len(subset_text)
            subset_tokens = estimate_tokens(subset_text) if subset_text else 0
            anchor_source = _extract_anchor_source(subset)
            retrieval_mode = merged_rule.rule.retrieval_spec.mode
            anchor_type = classify_rule_anchor_type(retrieval_mode, anchor_source)
            row: dict[str, Any] = {
                "dataset": "pdfs",
                "query_idx": query_package.query_idx,
                "query_id": f"q{query_package.query_idx}",
                "query_text": query_package.query_text,
                "packaging_mode": packaging_mode,
                "doc_ids": primary_doc_ids,
                "source_bundle_doc_ids_list": source_bundle_doc_ids_list,
                "source_bundle_prompt_tokens_list": source_bundle_prompt_tokens_list,
                "source_bundle_projected_costs_usd": source_bundle_projected_costs_usd,
                "evaluated_doc_id": document.doc_id,
                "doc_id": document.doc_id,
                "matched": subset["matched"],
                "rule_text": merged_rule.rule.rule_text,
                "evidence_basis": merged_rule.rule.evidence_basis,
                "retrieval_spec": merged_rule.rule.retrieval_spec.to_dict(),
                "rule_key": _canonical_retrieval_spec(merged_rule.rule),
                "retrieved_subset": subset,
                "retrieved_subset_chars": subset_chars,
                "retrieved_subset_tokens": subset_tokens,
                "token_count_retrieved_subset": subset_tokens,
                "ground_truth": document.ground_truth_answer,
                "judge_method": JUDGE_METHOD_NAME,
                "token_count_input": source_bundle_prompt_tokens_list[0],
                "projected_cost_usd": source_bundle_projected_costs_usd[0],
                "anchor_source": anchor_source,
                "anchor_type": anchor_type,
                "retrieval_mode": retrieval_mode,
            }

            if not subset["matched"]:
                row.update(
                    {
                        "generated_answer": None,
                        "judge_result": "NOT_RUN",
                        "actual_cost_usd": None,
                        "blocker": _classify_rule_blocker(
                            subset,
                            generated_answer=None,
                            judge_result="NOT_RUN",
                            subset_tokens=0,
                            retrieval_too_large_token_threshold=retrieval_too_large_token_threshold,
                        ),
                    }
                )
                rows.append(row)
                continue

            if subset_tokens >= retrieval_too_large_token_threshold:
                row.update(
                    {
                        "generated_answer": None,
                        "judge_result": "NOT_RUN",
                        "actual_cost_usd": None,
                        "blocker": "retrieval_too_large",
                    }
                )
                rows.append(row)
                continue

            if subset_tokens <= 0:
                row.update(
                    {
                        "generated_answer": None,
                        "judge_result": "NOT_RUN",
                        "actual_cost_usd": None,
                        "blocker": "empty_span",
                    }
                )
                rows.append(row)
                continue

            score_kwargs: dict[str, Any] = {
                "question": query_package.query_text,
                "retrieved_text": subset_text,
                "ground_truth": document.ground_truth_answer,
                "cached_caller": cached_caller,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            }
            score = score_retrieved_subset(**score_kwargs)
            row.update(
                {
                    "generated_answer": score.generated_answer,
                    "judge_result": score.judge_result,
                    "actual_cost_usd": (
                        score.metadata["generation"]["cost_usd"]
                        + score.metadata["judge"]["cost_usd"]
                    ),
                    "blocker": _classify_rule_blocker(
                        subset,
                        generated_answer=score.generated_answer,
                        judge_result=score.judge_result,
                        subset_tokens=subset_tokens,
                        retrieval_too_large_token_threshold=retrieval_too_large_token_threshold,
                    ),
                }
            )
            rows.append(row)

    return rows


def _classify_code_unmatched_reason(
    *,
    success: bool,
    error: str | None,
    region_text: str,
) -> str:
    err = error or ""
    if err.startswith("AST violations:"):
        return "sandbox_rejected_ast"
    if err.startswith("TimeoutError:"):
        return "sandbox_timeout"
    if success and not region_text:
        return "sandbox_empty_region"
    if "expected non-empty str" in err:
        return "sandbox_empty_region"
    if err:
        return "sandbox_runtime_error"
    return "sandbox_runtime_error"


def _code_subset_payload(
    code_rule: CodeRule,
    document_text: str,
) -> tuple[dict[str, Any], int]:
    exec_result = execute_locate_region(code_rule.code, document_text)
    region_text = exec_result.returned_region or ""
    matched = bool(exec_result.success) and bool(region_text)
    start = document_text.find(region_text) if matched else -1
    if start < 0:
        start = 0
    metadata: dict[str, Any] = {
        "rule_kind": "code",
        "exec_time_ms": float(exec_result.exec_time_ms),
        "error": exec_result.error,
    }
    if not matched:
        metadata["reason"] = _classify_code_unmatched_reason(
            success=exec_result.success,
            error=exec_result.error,
            region_text=region_text,
        )
    subset = {
        "matched": matched,
        "spans": (
            [
                {
                    "start": start,
                    "end": start + len(region_text),
                    "text": region_text,
                }
            ]
            if matched
            else []
        ),
        "metadata": metadata,
    }
    return subset, int(exec_result.success)


def _evaluate_code_rules(
    query_package: QueryPackage,
    packaging_mode: str,
    merged_rules: Sequence[CodeMergedRule],
    cached_caller: CachedLLMCaller,
    llm_provider: str,
    retrieval_too_large_token_threshold: int,
    llm_model: str,
) -> list[dict[str, Any]]:
    """Evaluate code-based locate_region rules via gen->judge."""
    rows: list[dict[str, Any]] = []

    for rule_index, merged_rule in enumerate(merged_rules):
        code_rule = merged_rule.rule
        primary_doc_ids = list(merged_rule.primary_doc_ids)
        source_bundle_doc_ids_list = [
            list(doc_ids) for doc_ids in merged_rule.source_bundle_doc_ids_list
        ]
        source_bundle_prompt_tokens_list = list(
            merged_rule.source_bundle_prompt_tokens_list
        )
        source_bundle_projected_costs_usd = list(
            merged_rule.source_bundle_projected_costs_usd
        )
        retrieval_spec = {"rule_kind": "code", "code": code_rule.code}
        rule_key = _canonical_code_rule(code_rule)
        for document in query_package.documents:
            subset, code_exec_success = _code_subset_payload(
                code_rule,
                document.markdown_text,
            )
            region_text = _subset_text_from_payload(subset)
            region_tokens = estimate_tokens(region_text) if region_text else 0
            metadata = subset["metadata"]

            row: dict[str, Any] = {
                "dataset": "pdfs",
                "query_idx": query_package.query_idx,
                "query_id": f"q{query_package.query_idx}",
                "query_text": query_package.query_text,
                "packaging_mode": packaging_mode,
                "doc_ids": primary_doc_ids,
                "source_bundle_doc_ids_list": source_bundle_doc_ids_list,
                "source_bundle_prompt_tokens_list": source_bundle_prompt_tokens_list,
                "source_bundle_projected_costs_usd": source_bundle_projected_costs_usd,
                "rule_index": rule_index,
                "rule_kind": "code",
                "retrieval_method": "python_code",
                "evaluated_doc_id": document.doc_id,
                "doc_id": document.doc_id,
                "matched": subset["matched"],
                "rule_text": code_rule.rule_text,
                "evidence_basis": code_rule.evidence_basis,
                "code": code_rule.code,
                "code_exec_success": bool(code_exec_success),
                "code_exec_error": metadata["error"],
                "code_exec_time_ms": metadata["exec_time_ms"],
                "sandbox_validation_status": "valid",
                "retrieval_spec": retrieval_spec,
                "rule_key": rule_key,
                "retrieved_subset": subset,
                "retrieved_region_size": len(region_text),
                "retrieved_subset_chars": len(region_text),
                "retrieved_subset_tokens": region_tokens,
                "token_count_retrieved_subset": region_tokens,
                "ground_truth": document.ground_truth_answer,
                "judge_method": JUDGE_METHOD_NAME,
                "token_count_input": source_bundle_prompt_tokens_list[0],
                "projected_cost_usd": source_bundle_projected_costs_usd[0],
                "anchor_source": None,
                "anchor_type": "code",
                "retrieval_mode": "python_code",
            }

            if not region_text.strip():
                row.update(
                    {
                        "generated_answer": None,
                        "judge_result": "NOT_RUN",
                        "actual_cost_usd": None,
                        "blocker": _classify_rule_blocker(
                            subset,
                            generated_answer=None,
                            judge_result="NOT_RUN",
                            subset_tokens=0,
                            retrieval_too_large_token_threshold=(
                                retrieval_too_large_token_threshold
                            ),
                        ),
                    }
                )
                rows.append(row)
                continue

            if region_tokens >= retrieval_too_large_token_threshold:
                row.update(
                    {
                        "generated_answer": None,
                        "judge_result": "NOT_RUN",
                        "actual_cost_usd": None,
                        "blocker": "retrieval_too_large",
                    }
                )
                rows.append(row)
                continue

            score_kwargs: dict[str, Any] = {
                "question": query_package.query_text,
                "retrieved_text": region_text,
                "ground_truth": document.ground_truth_answer,
                "cached_caller": cached_caller,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
            }
            score = score_retrieved_subset(**score_kwargs)
            row.update(
                {
                    "generated_answer": score.generated_answer,
                    "judge_result": score.judge_result,
                    "actual_cost_usd": (
                        score.metadata["generation"]["cost_usd"]
                        + score.metadata["judge"]["cost_usd"]
                    ),
                    "blocker": _classify_rule_blocker(
                        subset,
                        generated_answer=score.generated_answer,
                        judge_result=score.judge_result,
                        subset_tokens=region_tokens,
                        retrieval_too_large_token_threshold=(
                            retrieval_too_large_token_threshold
                        ),
                    ),
                }
            )
            rows.append(row)

    return rows


def _build_reflection_bundle_specs(
    config: dict[str, Any],
    query_package: QueryPackage,
    packaging_mode: str,
    bundle_results: Sequence[BundleRunResult],
    rule_summary: Sequence[dict[str, Any]],
    examples_by_key: dict[str, dict[str, list[dict[str, Any]]]],
    llm_provider: str,
    llm_model: str,
) -> list[BundleSpec]:
    template = load_prompt_template("range_reflect_v1")
    max_output_tokens = int(config.get("max_tokens_output", 4096))
    anonymize = config.get("anonymize_doc_ids", False)
    display_map = _display_doc_id_map(query_package.documents, anonymize)

    bundle_specs: list[BundleSpec] = []
    for bundle in bundle_results:
        bundle_query_package = _slice_query_package(query_package, bundle.doc_ids)
        prompt = fill_prompt(template, bundle_query_package, anonymize=anonymize)
        filtered_summary, failure_examples, success_examples = _filter_rule_summaries(
            bundle,
            rule_summary,
            examples_by_key,
        )
        prompt_summary = _apply_doc_id_display_map(filtered_summary, display_map)
        prompt_failures = _apply_doc_id_display_map(failure_examples, display_map)
        prompt_successes = _apply_doc_id_display_map(success_examples, display_map)
        prompt = (
            prompt.replace(
                "{current_rules_json}",
                json.dumps(
                    bundle.parsed_bundle.to_dict(), ensure_ascii=False, indent=2
                ),
            )
            .replace(
                "{rule_summary_json}",
                json.dumps(prompt_summary, ensure_ascii=False, indent=2),
            )
            .replace(
                "{failure_examples_json}",
                json.dumps(prompt_failures, ensure_ascii=False, indent=2),
            )
            .replace(
                "{success_examples_json}",
                json.dumps(prompt_successes, ensure_ascii=False, indent=2),
            )
        )
        prompt_tokens = estimate_tokens(prompt)
        bundle_specs.append(
            BundleSpec(
                bundle_index=bundle.bundle_index,
                doc_ids=bundle.doc_ids,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                projected_generation_cost_usd=_projected_generation_cost(
                    prompt_tokens,
                    max_output_tokens,
                    llm_provider,
                    llm_model,
                ),
            )
    )
    return bundle_specs


def _placeholder_merged_rules_for_code_bundle(
    code_bundle: CodeRuleBundle,
    query_package: QueryPackage,
    bundle_spec: BundleSpec,
) -> list[MergedRule]:
    selected_doc_ids = tuple(document.doc_id for document in query_package.documents)
    merged_rules: list[MergedRule] = []
    for rule_idx, code_rule in enumerate(code_bundle.rules, start=1):
        merged_rules.append(
            MergedRule(
                rule=RangeRule(
                    rule_text=code_rule.rule_text,
                    evidence_basis=code_rule.evidence_basis,
                    retrieval_spec=RetrievalSpec(
                        mode="page",
                        anchor=None,
                        anchor_b=None,
                        page_idx=rule_idx,
                        max_chars=9999,
                    ),
                ),
                primary_doc_ids=selected_doc_ids,
                source_bundle_doc_ids_list=(bundle_spec.doc_ids,),
                source_bundle_prompt_tokens_list=(bundle_spec.prompt_tokens,),
                source_bundle_projected_costs_usd=(
                    bundle_spec.projected_generation_cost_usd,
                ),
            )
        )
    return merged_rules


def _write_iteration_artifacts(
    output_dir: Path,
    query_package: QueryPackage,
    packaging_mode: str,
    iteration: ReflectionIterationResult,
) -> None:
    generation_path = output_dir / f"iter_{iteration.iteration_idx:02d}_generation.json"
    parsed_path = output_dir / f"iter_{iteration.iteration_idx:02d}_parsed_rules.json"
    rows_path = output_dir / f"iter_{iteration.iteration_idx:02d}_eval_rows.jsonl"
    summary_path = output_dir / f"iter_{iteration.iteration_idx:02d}_rule_summary.json"

    generation_path.write_text(
        json.dumps(
            {
                "iteration_idx": iteration.iteration_idx,
                **_generation_payload(
                    query_package, packaging_mode, iteration.bundle_results
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    parsed_path.write_text(
        json.dumps(
            {
                "iteration_idx": iteration.iteration_idx,
                **_parsed_payload(
                    query_package.query_idx,
                    packaging_mode,
                    iteration.bundle_results,
                    iteration.merged_rules,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rows(rows_path, iteration.rows)
    summary_path.write_text(
        json.dumps(
            {
                "iteration_idx": iteration.iteration_idx,
                "summary": iteration.summary.to_dict(),
                "best_single_rule": _best_single_rule(iteration.rule_summary),
                "rule_quality_table": list(iteration.rule_summary),
                "bundle_diagnostics": iteration.bundle_diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_best_iteration_artifacts(
    paths: dict[str, Path],
    query_package: QueryPackage,
    packaging_mode: str,
    iterations: Sequence[ReflectionIterationResult],
    best_iteration: ReflectionIterationResult,
) -> None:
    selection_metrics = [
        _iteration_selection_metrics(iteration) for iteration in iterations
    ]
    paths["best_iteration"].write_text(
        json.dumps(
            {
                "best_iteration": best_iteration.iteration_idx,
                "selection_objective": [
                    "maximize on_sample_accuracy",
                    "minimize avg_subset_tokens",
                    "minimize rule_count",
                ],
                "iterations": selection_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["best_rules"].write_text(
        json.dumps(
            {
                "query_idx": query_package.query_idx,
                "packaging_mode": packaging_mode,
                "best_iteration": best_iteration.iteration_idx,
                "merged_rules": [
                    rule.to_dict() for rule in best_iteration.merged_rules
                ],
                "rule_summary": list(best_iteration.rule_summary),
                "bundle_diagnostics": best_iteration.bundle_diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["generation"].write_text(
        json.dumps(
            _generation_payload(
                query_package, packaging_mode, best_iteration.bundle_results
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["parsed"].write_text(
        json.dumps(
            _parsed_payload(
                query_package.query_idx,
                packaging_mode,
                best_iteration.bundle_results,
                best_iteration.merged_rules,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rows(paths["rows"], best_iteration.rows)


def _summarize_run(
    query_package: QueryPackage,
    packaging_mode: str,
    bundle_specs: Sequence[BundleSpec],
    bundle_results: Sequence[BundleRunResult],
    merged_rules: Sequence[MergedRule],
    rows: Sequence[dict[str, Any]],
) -> BaselineRunSummary:
    avg_accuracy = (
        sum(1.0 if row["judge_result"] is True else 0.0 for row in rows) / len(rows)
        if rows
        else None
    )
    return BaselineRunSummary(
        status="ok",
        query_idx=query_package.query_idx,
        query_text=query_package.query_text,
        packaging_mode=packaging_mode,
        selected_doc_ids=tuple(document.doc_id for document in query_package.documents),
        doc_count=len(query_package.documents),
        bundle_count=len(bundle_specs),
        total_prompt_tokens=sum(spec.prompt_tokens for spec in bundle_specs),
        projected_generation_cost_usd=sum(
            spec.projected_generation_cost_usd for spec in bundle_specs
        ),
        actual_generation_cost_usd=sum(
            bundle.actual_generation_cost_usd for bundle in bundle_results
        ),
        rule_count=len(merged_rules),
        row_count=len(rows),
        avg_accuracy=avg_accuracy,
    )


def _build_structured_summary_payload(
    summary: BaselineRunSummary,
    rule_summary: Sequence[dict[str, Any]],
    bundle_diagnostics: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **summary.to_dict(),
        "best_single_rule": _best_single_rule(rule_summary),
        "rule_quality_table": list(rule_summary),
        "bundle_diagnostics": bundle_diagnostics,
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def _write_rows(rows_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_failure_summary(
    summary_path: Path,
    query_idx: int,
    packaging_mode: str,
    error: Exception,
) -> None:
    payload = {
        "status": "failed",
        "query_idx": query_idx,
        "packaging_mode": packaging_mode,
        "error": str(error),
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _default_run_output_dir(
    output_root: str | Path, query_idx: int, packaging_mode: str
) -> Path:
    return Path(output_root) / f"q{query_idx}" / packaging_mode


def _run_json_spec_reflect(
    config: dict[str, Any],
    query_package: QueryPackage,
    bundle_specs: Sequence[BundleSpec],
    packaging_mode: str,
    paths: dict[str, Path],
    caller: CachedLLMCaller,
    llm_provider: str,
    llm_model: str,
    text_format: str,
) -> BaselineRunSummary:
    if not bundle_specs:
        raise ValueError("json_spec_reflect requires at least one bundle")

    max_output_tokens = int(config.get("max_tokens_output", 4096))
    reflection_rounds = int(config.get("reflection_rounds", _DEFAULT_REFLECTION_ROUNDS))
    top_failure_examples = int(
        config.get(
            "reflection_top_failure_examples",
            _DEFAULT_REFLECTION_TOP_FAILURE_EXAMPLES,
        )
    )
    retrieval_too_large_token_threshold = int(
        config.get(
            "retrieval_too_large_token_threshold",
            _DEFAULT_RETRIEVAL_TOO_LARGE_TOKEN_THRESHOLD,
        )
    )

    iterations: list[ReflectionIterationResult] = []
    current_bundle_specs = list(bundle_specs)
    previous_metrics: dict[str, Any] | None = None
    cumulative_projected_generation_cost_usd = 0.0

    for iteration_idx in range(reflection_rounds + 1):
        cumulative_projected_generation_cost_usd += sum(
            spec.projected_generation_cost_usd for spec in current_bundle_specs
        )
        _ensure_projected_cost_value_within_threshold(
            config,
            cumulative_projected_generation_cost_usd,
            label="cumulative projected generation cost",
        )
        _ensure_bundle_specs_within_model_context(
            current_bundle_specs,
            max_output_tokens=max_output_tokens,
            llm_provider=llm_provider,
            llm_model=llm_model,
            stage_label=f"json_spec_reflect iter_{iteration_idx:02d}",
        )
        prompt_paths = _write_bundle_prompts(
            paths["prompt_dir"],
            current_bundle_specs,
            filename_prefix=f"iter_{iteration_idx:02d}_bundle",
        )
        bundle_results = _run_bundle_generations(
            current_bundle_specs,
            prompt_paths,
            caller,
            llm_provider,
            llm_model,
            max_output_tokens,
        )
        _validate_bundle_query_indices(bundle_results, query_package.query_idx)
        merged_rules = _merge_rules(bundle_results)
        rows = _evaluate_rules(
            query_package,
            packaging_mode,
            merged_rules,
            caller,
            llm_provider,
            retrieval_too_large_token_threshold,
            llm_model,
        )
        summary = _with_summary_context(
            _summarize_run(
                query_package,
                packaging_mode,
                current_bundle_specs,
                bundle_results,
                merged_rules,
                rows,
            ),
            rule_mode="json_spec_reflect",
            text_format=text_format,
        )
        rule_summary = _summarize_rules(merged_rules, rows)
        bundle_diagnostics = _build_bundle_diagnostics(rule_summary)
        summary = _with_bundle_diagnostics(summary, bundle_diagnostics)
        iteration = ReflectionIterationResult(
            iteration_idx=iteration_idx,
            bundle_specs=tuple(current_bundle_specs),
            bundle_results=tuple(bundle_results),
            merged_rules=tuple(merged_rules),
            rows=tuple(rows),
            rule_summary=tuple(rule_summary),
            bundle_diagnostics=bundle_diagnostics,
            summary=summary,
        )
        iterations.append(iteration)
        _write_iteration_artifacts(
            paths["summary"].parent,
            query_package,
            packaging_mode,
            iteration,
        )

        current_metrics = _iteration_selection_metrics(iteration)
        if iteration_idx >= reflection_rounds:
            break
        if previous_metrics is not None and not _is_better_iteration(
            current_metrics, previous_metrics
        ):
            break

        examples_by_key = _build_rule_examples(rows, top_failure_examples)
        current_bundle_specs = _build_reflection_bundle_specs(
            config,
            query_package,
            packaging_mode,
            bundle_results,
            rule_summary,
            examples_by_key,
            llm_provider,
            llm_model,
        )
        (
            paths["summary"].parent
            / f"iter_{iteration_idx + 1:02d}_reflection_prompt.txt"
        ).write_text(
            _combine_bundle_prompts(current_bundle_specs),
            encoding="utf-8",
        )
        previous_metrics = current_metrics

    best_iteration = _select_best_iteration(iterations)
    total_prompt_tokens = sum(
        iteration.summary.total_prompt_tokens for iteration in iterations
    )
    total_projected_generation_cost_usd = sum(
        iteration.summary.projected_generation_cost_usd for iteration in iterations
    )
    total_actual_generation_cost_usd = sum(
        iteration.summary.actual_generation_cost_usd for iteration in iterations
    )
    best_summary = _with_summary_context(
        best_iteration.summary,
        rule_mode="json_spec_reflect",
        text_format=text_format,
        best_iteration=best_iteration.iteration_idx,
        total_prompt_tokens=total_prompt_tokens,
        total_projected_generation_cost_usd=total_projected_generation_cost_usd,
        total_actual_generation_cost_usd=total_actual_generation_cost_usd,
    )
    _write_best_iteration_artifacts(
        paths,
        query_package,
        packaging_mode,
        iterations,
        best_iteration,
    )
    paths["summary"].write_text(
        json.dumps(
            _build_structured_summary_payload(
                best_summary,
                best_iteration.rule_summary,
                best_iteration.bundle_diagnostics,
                extra_fields={
                    "reflection_rounds_requested": reflection_rounds,
                    "reflection_rounds_completed": len(iterations) - 1,
                    "iterations": [
                        _iteration_selection_metrics(iteration)
                        for iteration in iterations
                    ],
                },
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return best_summary


def _run_baseline_with_config(
    config: dict[str, Any],
    query_idx: int,
    packaging_mode: str,
    output_dir: str | Path | None,
    llm_provider: str | None,
    llm_model: str | None,
    cache_db_path: str,
    rule_mode: str | None = None,
) -> BaselineRunSummary:
    if packaging_mode not in SUPPORTED_PACKAGING_MODES:
        raise NotImplementedError(
            f"unsupported packaging_mode={packaging_mode!r}; "
            f"supported={list(SUPPORTED_PACKAGING_MODES)}"
        )

    target_output_dir = (
        _default_run_output_dir(_DEFAULT_OUTPUT_ROOT, query_idx, packaging_mode)
        if output_dir is None
        else Path(output_dir)
    )
    target_output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(target_output_dir)
    bundle_results: list[BundleRunResult] = []
    resolved_llm_provider = _resolve_llm_provider(config, llm_provider)
    resolved_llm_model = _resolve_llm_model(config, llm_model)
    query_package: QueryPackage | None = None
    try:
        resolved_rule_mode = rule_mode if rule_mode is not None else _resolve_rule_mode(config)
        if resolved_rule_mode not in SUPPORTED_RULE_MODES:
            raise ValueError(
                f"unsupported rule_mode={resolved_rule_mode!r}; "
                f"supported={list(SUPPORTED_RULE_MODES)}"
            )
        query_package, bundle_specs = _build_bundle_specs(
            config,
            query_idx,
            packaging_mode,
            resolved_llm_provider,
            resolved_llm_model,
            rule_mode=resolved_rule_mode,
        )
        max_output_tokens = int(config.get("max_tokens_output", 4096))
        _ensure_bundle_specs_within_model_context(
            bundle_specs,
            max_output_tokens=max_output_tokens,
            llm_provider=resolved_llm_provider,
            llm_model=resolved_llm_model,
            stage_label="initial_generation",
        )
        if resolved_rule_mode in {"json_spec", "json_spec_reflect"}:
            _ensure_projected_cost_within_threshold(config, bundle_specs)

        caller = CachedLLMCaller(db_path=cache_db_path)

        text_format = _resolve_rule_text_format(config, resolved_rule_mode)
        retrieval_too_large_token_threshold = int(
            config.get(
                "retrieval_too_large_token_threshold",
                _DEFAULT_RETRIEVAL_TOO_LARGE_TOKEN_THRESHOLD,
            )
        )

        if resolved_rule_mode == "python_code":
            code_bundle_specs = _build_code_bundle_specs(
                config,
                query_package,
                bundle_specs,
                resolved_llm_provider,
                resolved_llm_model,
            )
            _ensure_bundle_specs_within_model_context(
                code_bundle_specs,
                max_output_tokens=max_output_tokens,
                llm_provider=resolved_llm_provider,
                llm_model=resolved_llm_model,
                stage_label="python_code_generation",
            )
            _ensure_projected_cost_within_threshold(config, code_bundle_specs)
            code_prompt_paths = _write_bundle_prompts(
                paths["prompt_dir"],
                code_bundle_specs,
                filename_prefix="code_bundle",
            )
            code_results = _run_code_bundle_generations(
                query_idx,
                code_bundle_specs,
                code_prompt_paths,
                caller,
                resolved_llm_provider,
                resolved_llm_model,
                max_output_tokens,
            )
            merged_code_rules = _merge_code_rules(code_results)
            rejected_rules = _rejected_code_validation_records(code_results)

            paths["generation"].write_text(
                json.dumps(
                    _code_generation_payload(
                        query_package,
                        packaging_mode,
                        code_results,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            paths["parsed"].write_text(
                json.dumps(
                    _code_parsed_payload(
                        query_idx,
                        packaging_mode,
                        code_results,
                        merged_code_rules,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            paths["code_rules"].write_text(
                json.dumps(
                    _code_parsed_payload(
                        query_idx,
                        packaging_mode,
                        code_results,
                        merged_code_rules,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            rows = _evaluate_code_rules(
                query_package,
                packaging_mode,
                merged_code_rules,
                caller,
                resolved_llm_provider,
                retrieval_too_large_token_threshold,
                resolved_llm_model,
            )
            _write_rows(paths["rows"], rows)

            rule_summary = _summarize_code_rules(merged_code_rules, rows)
            cross_doc_eval = _build_code_cross_doc_eval(
                merged_code_rules,
                rule_summary,
                rows,
            )
            bundle_diagnostics = _build_bundle_diagnostics(rule_summary)
            paths["best_rules"].write_text(
                json.dumps(
                    _code_best_rules_payload(
                        query_package,
                        packaging_mode,
                        merged_code_rules,
                        rule_summary,
                        cross_doc_eval,
                        bundle_diagnostics,
                        rejected_rules,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            summary = _summarize_code_run(
                query_package,
                packaging_mode,
                code_bundle_specs,
                code_results,
                merged_code_rules,
                rows,
            )
            summary = _with_summary_context(
                _with_bundle_diagnostics(summary, bundle_diagnostics),
                rule_mode="python_code",
                text_format=text_format,
            )
        elif resolved_rule_mode == "json_spec_reflect":
            summary = _run_json_spec_reflect(
                config,
                query_package,
                bundle_specs,
                packaging_mode,
                paths,
                caller,
                resolved_llm_provider,
                resolved_llm_model,
                text_format,
            )
        elif resolved_rule_mode == "json_spec":
            # JSON spec path (original)
            prompt_paths = _write_bundle_prompts(paths["prompt_dir"], bundle_specs)
            bundle_results = _run_bundle_generations(
                bundle_specs,
                prompt_paths,
                caller,
                resolved_llm_provider,
                resolved_llm_model,
                max_output_tokens,
            )
            _validate_bundle_query_indices(bundle_results, query_idx)
            paths["generation"].write_text(
                json.dumps(
                    _generation_payload(query_package, packaging_mode, bundle_results),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            merged_rules = _merge_rules(bundle_results)
            paths["parsed"].write_text(
                json.dumps(
                    _parsed_payload(
                        query_idx, packaging_mode, bundle_results, merged_rules
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            rows = _evaluate_rules(
                query_package,
                packaging_mode,
                merged_rules,
                caller,
                resolved_llm_provider,
                retrieval_too_large_token_threshold,
                resolved_llm_model,
            )
            _write_rows(paths["rows"], rows)

            summary = _summarize_run(
                query_package,
                packaging_mode,
                bundle_specs,
                bundle_results,
                merged_rules,
                rows,
            )
            rule_summary = _summarize_rules(merged_rules, rows)
            bundle_diagnostics = _build_bundle_diagnostics(rule_summary)
            summary = _with_summary_context(
                _with_bundle_diagnostics(summary, bundle_diagnostics),
                rule_mode="json_spec",
                text_format=text_format,
            )
        else:
            raise AssertionError(
                f"unexpected validated rule_mode={resolved_rule_mode!r}"
            )
        if resolved_rule_mode != "json_spec_reflect":
            summary_payload = summary.to_dict()
            if resolved_rule_mode == "json_spec":
                summary_payload = _build_structured_summary_payload(
                    summary,
                    rule_summary,
                    bundle_diagnostics,
                )
            paths["summary"].write_text(
                json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return summary
    except Exception as exc:
        if bundle_results and query_package is not None:
            paths["generation"].write_text(
                json.dumps(
                    _generation_payload(query_package, packaging_mode, bundle_results),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        paths["parsed"].write_text(
            json.dumps(
                {"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        _write_failure_summary(paths["summary"], query_idx, packaging_mode, exc)
        raise


def run_baseline(
    query_idx: int,
    packaging_mode: str,
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    cache_db_path: str = _DEFAULT_CACHE_DB_PATH,
    rule_mode: str | None = None,
) -> BaselineRunSummary:
    config = _load_config(config_path)
    return _run_baseline_with_config(
        config=config,
        query_idx=query_idx,
        packaging_mode=packaging_mode,
        output_dir=output_dir,
        llm_provider=llm_provider,
        llm_model=llm_model,
        cache_db_path=cache_db_path,
        rule_mode=rule_mode,
    )


def run_baseline_sweep(
    packaging_modes: Sequence[str] = DEFAULT_SWEEP_PACKAGING_MODES,
    query_indices: Sequence[int] | None = None,
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    output_root: str | Path = _DEFAULT_OUTPUT_ROOT,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    cache_db_path: str = _DEFAULT_CACHE_DB_PATH,
    rule_mode: str | None = None,
) -> BaselineSweepSummary:
    config = _load_config(config_path)
    resolved_query_indices = (
        _get_query_indices_from_config(config)
        if query_indices is None
        else [int(idx) for idx in query_indices]
    )

    summaries: list[BaselineRunSummary] = []
    for query_idx in resolved_query_indices:
        for packaging_mode in packaging_modes:
            summaries.append(
                _run_baseline_with_config(
                    config=config,
                    query_idx=query_idx,
                    packaging_mode=packaging_mode,
                    output_dir=_default_run_output_dir(
                        output_root, query_idx, packaging_mode
                    ),
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    cache_db_path=cache_db_path,
                    rule_mode=rule_mode,
                )
            )

    sweep_summary = BaselineSweepSummary(
        status="ok",
        packaging_modes=tuple(packaging_modes),
        query_indices=tuple(resolved_query_indices),
        run_count=len(summaries),
        total_projected_generation_cost_usd=sum(
            run.projected_generation_cost_usd for run in summaries
        ),
        total_actual_generation_cost_usd=sum(
            run.actual_generation_cost_usd for run in summaries
        ),
        runs=tuple(summaries),
    )

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    (output_root_path / "aggregate_summary.json").write_text(
        json.dumps(sweep_summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sweep_summary


__all__ = [
    "DEFAULT_SWEEP_PACKAGING_MODES",
    "BaselineRunSummary",
    "BaselineSweepSummary",
    "SUPPORTED_PACKAGING_MODES",
    "run_baseline",
    "run_baseline_sweep",
]
