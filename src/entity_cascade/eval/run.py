"""Entry point for `make index`, `make eval`, `make record`."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

from entity_cascade.calibration import Calibration
from entity_cascade.config import PROJECT_ROOT, AppConfig, load_config, resolve_path
from entity_cascade.data.loader import Dataset, load_dataset
from entity_cascade.data.split import Splits, make_splits
from entity_cascade.data.stats import dataset_stats
from entity_cascade.eval.llm_metrics import apply_arbiter, arbiter_metrics
from entity_cascade.eval.llm_tuning import TuningResult, tune
from entity_cascade.eval.metrics import candidate_recall, top1_accuracy
from entity_cascade.eval.modes import (
    MODES,
    CurvePoint,
    mode_row,
    precision_coverage_curve,
    scored_for_mode,
)
from entity_cascade.eval.plots import make_all
from entity_cascade.eval.readme import update_readme
from entity_cascade.eval.report import (
    EvalResults,
    LLMReport,
    RecallRow,
    SplitInfo,
    SystemRow,
    Timings,
    Top1Row,
    TransferRow,
    ZoneBlock,
    write_reports,
)
from entity_cascade.eval.tradeoff import tradeoff_rows
from entity_cascade.eval.zone_metrics import Decision, system_metrics, zone_decisions, zone_table
from entity_cascade.llm.arbiter import (
    PROMPT_VERSIONS,
    ArbiterItem,
    ArbiterResult,
    arbitrate,
    build_prompt,
)
from entity_cascade.llm.base import CassetteMiss, LLMClient
from entity_cascade.llm.fake import HeuristicFakeLLM
from entity_cascade.llm.openai_compat import OpenAICompatibleLLM
from entity_cascade.llm.recorded import RecordedLLM, RecordingLLM
from entity_cascade.pipeline import (
    RankingInputs,
    benchmark_latency,
    build_generator,
    compute_ranking_inputs,
    fit_on_val,
)
from entity_cascade.rerank.cross_encoder import CachedReranker, CrossEncoderReranker, Reranker
from entity_cascade.rerank.features import FeatureWeights
from entity_cascade.rerank.ranker import RowRanking, rank_row
from entity_cascade.retrieval.dense import Embedder, SentenceTransformerEmbedder, make_client
from entity_cascade.schemas import Record, Zone
from entity_cascade.zones.thresholds import ARBITER_SHOWN

BENCH_ROWS = 50
SPLITS = ("val", "test")
# (target precision, Wilson z) pairs for the trade-off table. Reporting only: the default
# target stays the one in the config.
TRADEOFF_SETTINGS = (
    (0.90, 0.0),
    (0.95, 0.0),
    (0.98, 0.0),
    (0.99, 0.0),
    (0.995, 0.0),
    (0.98, 1.96),
)


@dataclass(frozen=True)
class Models:
    embedder: Embedder
    reranker: Reranker  # may be cached: never used for latency
    bench_reranker: Reranker  # always the raw model


def build_models(cfg: AppConfig) -> Models:
    embedder = SentenceTransformerEmbedder(cfg.models.embedding, cfg.models.query_instruction)
    raw = CrossEncoderReranker(cfg.models.reranker, cfg.rerank.batch_size)
    reranker: Reranker = raw
    if cfg.rerank.cache_path is not None:
        reranker = CachedReranker(raw, cfg.models.reranker, resolve_path(cfg.rerank.cache_path))
    return Models(embedder=embedder, reranker=reranker, bench_reranker=raw)


@dataclass(frozen=True)
class CascadeRun:
    """Retrieval, reranking, calibration and zone decisions for val and test."""

    cfg: AppConfig
    ds: Dataset
    splits: Splits
    inputs: dict[str, RankingInputs]
    cal: Calibration
    rankings: dict[str, dict[str, RowRanking]]
    decisions: dict[str, dict[str, Decision]]  # zone decisions, before the LLM
    index_s: float
    retrieval_ms: float
    rerank_ms: float
    bench_rows: int


def run_cascade(cfg: AppConfig, models: Models) -> CascadeRun:
    ds = load_dataset(cfg.dataset)
    splits = make_splits(ds, cfg.val_fraction, cfg.seed, cfg.dataset.new_product_fraction)
    gen, index_s = build_generator(
        cfg, splits.catalog, models.embedder, make_client(cfg.retrieval.qdrant_url)
    )
    # Latency is measured on an uncached sample, never on cache hits.
    sample = splits.val.rows[:BENCH_ROWS]
    retrieval_ms, rerank_ms = benchmark_latency(
        cfg, gen, splits.catalog, sample, models.bench_reranker
    )
    by_split = {"val": splits.val, "test": splits.test}
    inputs = {
        name: compute_ranking_inputs(cfg, gen, splits.catalog, sp.rows, models.reranker)
        for name, sp in by_split.items()
    }
    cal = fit_on_val(splits, inputs["val"].raws, cfg)  # val only
    rankings = {
        name: {r.id: rank_row(inputs[name].raws[r.id], cal.weights) for r in sp.rows}
        for name, sp in by_split.items()
    }
    decisions = {
        name: zone_decisions([r.id for r in sp.rows], rankings[name], cal.thresholds)
        for name, sp in by_split.items()
    }
    return CascadeRun(
        cfg=cfg,
        ds=ds,
        splits=splits,
        inputs=inputs,
        cal=cal,
        rankings=rankings,
        decisions=decisions,
        index_s=index_s,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        bench_rows=len(sample),
    )


# ---- LLM stage ----


def make_llm(cfg: AppConfig, kind: str | None = None) -> LLMClient:
    kind = kind or cfg.llm.provider
    if kind == "fake":
        return HeuristicFakeLLM()
    if kind == "recorded":
        return RecordedLLM(cfg.llm.model, resolve_path(cfg.llm.cassette_dir))
    if kind == "openai":
        return OpenAICompatibleLLM(cfg.llm)
    raise ValueError(f"unknown LLM provider {kind!r}")


def llm_items(run: CascadeRun, split: str = "test") -> list[ArbiterItem]:
    catalog = {r.id: r for r in run.splits.catalog}
    rows = {r.id: r for r in (run.splits.val if split == "val" else run.splits.test).rows}
    items: list[ArbiterItem] = []
    for rid, d in run.decisions[split].items():
        if d.zone is Zone.LLM_ARBITER:
            top = run.rankings[split][rid].candidates[:ARBITER_SHOWN]
            top = top[: run.cfg.llm.top_candidates]
            items.append(ArbiterItem(rows[rid], tuple(catalog[c.catalog_id] for c in top)))
    return items


async def _run_arbiter(
    llm: LLMClient, items: list[ArbiterItem], cfg: AppConfig
) -> list[ArbiterResult]:
    try:
        return await arbitrate(llm, items, cfg.llm)
    finally:
        closer = getattr(llm, "aclose", None)
        if closer is not None:
            await closer()


# ---- evaluation ----


def _recall_rows(
    split: str,
    rows: tuple[Record, ...],
    truth: dict[str, frozenset[str]],
    inputs: RankingInputs,
) -> list[RecallRow]:
    n_pair = sum(1 for r in rows if truth[r.id])
    out: list[RecallRow] = []
    for mode in ("bm25", "dense", "hybrid"):
        rankings = {r.id: getattr(inputs.cands[r.id], mode) for r in rows}
        out.append(
            RecallRow(
                mode=mode, split=split, n_rows=n_pair, recall=candidate_recall(rankings, truth)
            )
        )
    return out


def _top1_rows(
    split: str,
    rows: tuple[Record, ...],
    truth: dict[str, frozenset[str]],
    inputs: RankingInputs,
    weights: FeatureWeights,
) -> list[Top1Row]:
    modes: dict[str, dict[str, str | None]] = {
        "hybrid (RRF order)": {r.id: inputs.cands[r.id].hybrid[0] for r in rows},
        "hybrid + reranker": {
            r.id: rank_row(inputs.raws[r.id], FeatureWeights()).top1_id for r in rows
        },
        "hybrid + reranker + features": {
            r.id: rank_row(inputs.raws[r.id], weights).top1_id for r in rows
        },
    }
    out: list[Top1Row] = []
    for mode, top1 in modes.items():
        acc, n = top1_accuracy(top1, truth)
        out.append(Top1Row(mode=mode, split=split, n_rows=n, accuracy=acc))
    return out


def evaluate(
    run: CascadeRun,
    llm: LLMClient | None,
    config_path: str,
    tuning: TuningResult | None = None,
) -> EvalResults:
    cfg, splits = run.cfg, run.splits
    by_split = {"val": splits.val, "test": splits.test}
    recall_rows: list[RecallRow] = []
    top1_rows: list[Top1Row] = []
    split_info: list[SplitInfo] = []
    zone_blocks: list[ZoneBlock] = []
    systems: list[SystemRow] = []
    for name, sp in by_split.items():
        truth = dict(sp.truth)
        ids = [r.id for r in sp.rows]
        rankings, decisions = run.rankings[name], run.decisions[name]
        recall_rows += _recall_rows(name, sp.rows, truth, run.inputs[name])
        top1_rows += _top1_rows(name, sp.rows, truth, run.inputs[name], run.cal.weights)
        zones = {i: d.zone for i, d in decisions.items()}
        zone_blocks.append(ZoneBlock(split=name, zones=zone_table(zones, rankings, truth)))
        systems += [
            SystemRow(
                split=name,
                name="accept every top-1 (no cascade)",
                metrics=system_metrics(
                    {i: Decision(i, Zone.AUTO_MATCH, rankings[i].top1_id, "auto") for i in ids},
                    truth,
                ),
            ),
            SystemRow(
                split=name,
                name="cascade, llm_arbiter rows go to a human",
                metrics=system_metrics(decisions, truth),
            ),
            SystemRow(
                split=name,
                name="everything to a human",
                metrics=system_metrics(
                    {i: Decision(i, Zone.HUMAN_REVIEW, None, "human") for i in ids}, truth
                ),
            ),
        ]
        split_info.append(
            SplitInfo(
                split=name,
                rows=len(sp.rows),
                new_products=sum(1 for r in sp.rows if r.id in splits.new_product_ids),
                rows_with_pair=sum(1 for r in sp.rows if sp.truth[r.id]),
            )
        )

    val_ids = [r.id for r in splits.val.rows]
    test_ids = [r.id for r in splits.test.rows]
    mode_rows = [
        mode_row(
            mode,
            splits.val,
            splits.test,
            scored_for_mode(
                mode, val_ids, run.inputs["val"].cands, run.inputs["val"].raws, run.cal.weights
            ),
            scored_for_mode(
                mode, test_ids, run.inputs["test"].cands, run.inputs["test"].raws, run.cal.weights
            ),
            cfg.thresholds,
        )
        for mode in MODES
    ]
    trade = tradeoff_rows(
        splits.val,
        splits.test,
        run.rankings["val"],
        run.rankings["test"],
        cfg.thresholds,
        TRADEOFF_SETTINGS,
    )

    # The arbiter runs on test only: nothing is tuned with it, so val calls would be wasted.
    llm_report: LLMReport | None = None
    llm_note: str | None = None
    if llm is not None:
        items = llm_items(run, "test")
        try:
            results = asyncio.run(_run_arbiter(llm, items, cfg))
        except CassetteMiss:
            results = None
            llm_note = f"no cassette for model {llm.model!r} in {cfg.llm.cassette_dir}"
        if results is not None:
            test_truth = dict(splits.test.truth)
            metrics = arbiter_metrics(
                results, test_truth, run.rankings["test"], cfg.llm, len(splits.test.rows)
            )
            llm_report = LLMReport(
                model=llm.model,
                stand_in=llm.model.startswith("fake"),
                replayed=isinstance(llm, RecordedLLM),
                metrics=metrics,
            )
            with_llm = apply_arbiter(run.decisions["test"], results, cfg.llm.min_confidence)
            systems.append(
                SystemRow(
                    split="test",
                    name="cascade + LLM arbiter",
                    metrics=system_metrics(with_llm, test_truth),
                )
            )
    elif cfg.llm.provider == "recorded":
        llm_note = "LLM arbiter was not run."

    tm = Timings(
        index_build_s=run.index_s,
        retrieval_ms_per_row=run.retrieval_ms,
        rerank_ms_per_row=run.rerank_ms,
        bench_rows=run.bench_rows,
    )
    return EvalResults(
        config_path=config_path.replace("\\", "/"),
        dataset_name=cfg.dataset.name,
        seed=cfg.seed,
        embedding_model=cfg.models.embedding,
        reranker_model=cfg.models.reranker,
        dataset_stats=dataset_stats(run.ds),
        catalog_size=len(splits.catalog),
        new_product_fraction=cfg.dataset.new_product_fraction,
        splits=split_info,
        candidate_recall=recall_rows,
        modes=mode_rows,
        top1=top1_rows,
        calibration=run.cal,
        target_precision=cfg.thresholds.target_precision,
        zone_blocks=zone_blocks,
        systems=systems,
        tradeoff=trade,
        llm=llm_report,
        llm_note=llm_note,
        llm_tuning=tuning,
        timings=tm,
    )


# ---- commands ----


def cmd_index(cfg: AppConfig) -> int:
    if cfg.retrieval.qdrant_url == ":memory:":
        print("retrieval.qdrant_url is ':memory:'; nothing persists. Set a Qdrant URL (make up).")
        return 1
    ds = load_dataset(cfg.dataset)
    splits = make_splits(ds, cfg.val_fraction, cfg.seed, cfg.dataset.new_product_fraction)
    embedder = SentenceTransformerEmbedder(cfg.models.embedding, cfg.models.query_instruction)
    build_generator(cfg, splits.catalog, embedder, make_client(cfg.retrieval.qdrant_url))
    print(f"indexed {len(splits.catalog)} catalog items into {cfg.retrieval.qdrant_url}")
    return 0


def recalibrate(run: CascadeRun, cal: Calibration) -> CascadeRun:
    """The same retrieval/reranker output, ranked and zoned with another calibration."""
    rankings = {
        name: {rid: rank_row(raw, cal.weights) for rid, raw in run.inputs[name].raws.items()}
        for name in SPLITS
    }
    decisions = {
        name: zone_decisions(list(rankings[name]), rankings[name], cal.thresholds)
        for name in SPLITS
    }
    return replace(run, cal=cal, rankings=rankings, decisions=decisions)


def transfer_row(name: str, run: CascadeRun) -> TransferRow:
    truth = dict(run.splits.test.truth)
    rankings, decisions = run.rankings["test"], run.decisions["test"]
    auto = [i for i, d in decisions.items() if d.zone is Zone.AUTO_MATCH]
    sysm = system_metrics(decisions, truth)
    t = run.cal.thresholds
    return TransferRow(
        name=name,
        high=t.high,
        margin_min=t.margin_min,
        auto_share=len(auto) / len(decisions) if decisions else float("nan"),
        auto_precision=(
            sum(rankings[i].top1_id in truth[i] for i in auto) / len(auto) if auto else float("nan")
        ),
        human_share=sysm.human_share,
        system_precision=sysm.precision,
        system_recall=sysm.recall,
    )


def curves_for(run: CascadeRun) -> dict[str, dict[str, list[CurvePoint]]]:
    out: dict[str, dict[str, list[CurvePoint]]] = {}
    for mode in MODES:
        out[mode] = {}
        for name, sp in (("val", run.splits.val), ("test", run.splits.test)):
            scored = scored_for_mode(
                mode,
                [r.id for r in sp.rows],
                run.inputs[name].cands,
                run.inputs[name].raws,
                run.cal.weights,
            )
            out[mode][name] = precision_coverage_curve(scored, sp.truth)
    return out


def cmd_eval(
    cfg: AppConfig,
    config_path: str,
    llm_kind: str | None,
    secondary_path: str | None,
    out_dir: Path,
    readme: Path | None,
) -> int:
    models = build_models(cfg)
    run = run_cascade(cfg, models)
    llm = make_llm(cfg, llm_kind)
    tuning = tune_from_cassettes(run) if isinstance(llm, RecordedLLM) else None
    results = evaluate(run, llm, config_path, tuning)

    if secondary_path:
        cfg2 = load_config(secondary_path)
        run2 = run_cascade(cfg2, models)
        sec = evaluate(run2, None, secondary_path)
        carried = recalibrate(run2, run.cal)  # first dataset's calibration, unchanged
        sec = sec.model_copy(
            update={
                "transfer": [
                    transfer_row(f"{cfg.dataset.name} calibration, not refitted", carried),
                    transfer_row(f"refitted on {cfg2.dataset.name} val", run2),
                ]
            }
        )
        results = results.model_copy(update={"secondary": sec})

    reports = out_dir
    charts = make_all(results, curves_for(run), reports)
    results = results.model_copy(update={"charts": charts})
    write_reports(results, reports)
    run.cal.save(reports / "calibration.json")
    if readme is not None:
        update_readme(results, readme)
    print((reports / "results.md").read_text(encoding="utf-8"))
    return 0


def check_provider(cfg: AppConfig, transport: httpx.BaseTransport | None = None) -> str | None:
    """Preflight for `record`: is the endpoint up and does it serve the model? None = ready."""
    url = cfg.llm.base_url.rstrip("/") + "/models"
    key = os.environ.get(cfg.llm.api_key_env, "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with httpx.Client(headers=headers, timeout=5.0, transport=transport) as http:
            resp = http.get(url)
    except httpx.HTTPError as exc:
        return f"cannot reach {url}: {type(exc).__name__}. Is the server running?"
    if resp.status_code != 200:
        return f"{url} answered HTTP {resp.status_code}"
    try:
        served = [m["id"] for m in resp.json()["data"]]
    except (KeyError, TypeError, ValueError):
        return None  # a provider without a model list: let the first request decide
    if cfg.llm.model not in served:
        return (
            f"model {cfg.llm.model!r} is not served at {cfg.llm.base_url}; available: {served}. "
            f"For Ollama: `ollama pull {cfg.llm.model}`"
        )
    return None


def _with_prompt(cfg: AppConfig, version: str) -> AppConfig:
    return cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"prompt_version": version})})


def tune_on_val(
    run: CascadeRun, make_client: Callable[[], LLMClient]
) -> tuple[TuningResult, dict[str, list[ArbiterResult]]]:
    """Run the arbiter on the val gray zone once per prompt version, then pick on val only."""
    items = llm_items(run, "val")
    by_version: dict[str, list[ArbiterResult]] = {}
    for version in PROMPT_VERSIONS:
        cfg_v = _with_prompt(run.cfg, version)
        by_version[version] = asyncio.run(_run_arbiter(make_client(), items, cfg_v))
    return tune(by_version, run.splits.val, run.decisions["val"], run.cfg.llm), by_version


def tune_from_cassettes(run: CascadeRun) -> TuningResult | None:
    """Replay the val tuning from cassettes; None if they were never recorded."""
    try:
        result, _ = tune_on_val(run, lambda: make_llm(run.cfg, "recorded"))
    except CassetteMiss:
        return None
    return result


def cmd_tune(cfg: AppConfig, kind: str) -> int:
    """Record val answers for every prompt version and choose version + min_confidence on val."""
    if kind == "openai":
        problem = check_provider(cfg)
        if problem:
            print("preflight failed: " + problem)
            return 1
    run = run_cascade(cfg, build_models(cfg))
    directory = resolve_path(cfg.llm.cassette_dir)

    def make_client() -> LLMClient:
        inner = make_llm(cfg, kind)
        return inner if kind == "recorded" else RecordingLLM(inner, directory)

    result, by_version = tune_on_val(run, make_client)
    for version, results in by_version.items():
        failed = sum(1 for r in results if r.answer is None)
        mean = sum(r.latency_s for r in results) / len(results) if results else float("nan")
        print(f"{version}: {len(results)} calls, {failed} unusable, mean latency {mean:.2f} s")
    out = PROJECT_ROOT / "reports" / "llm_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"chosen on val: version {result.chosen_version}, "
        f"min_confidence {result.chosen_min_confidence} (target reached: {result.feasible})"
    )
    print(f"wrote {out}; run `make eval` for the full table")
    return 0


def cmd_record(cfg: AppConfig, kind: str, dry_run: bool) -> int:
    """Run the arbiter on the test split against a real provider and save every answer."""
    if kind == "openai" and cfg.llm.model == "unset":
        print("set llm.model (and llm.base_url / the key in $" + cfg.llm.api_key_env + ") first")
        return 1
    if kind == "openai" and not dry_run:
        problem = check_provider(cfg)
        if problem:
            print("preflight failed: " + problem)
            return 1
    run = run_cascade(cfg, build_models(cfg))
    if dry_run:
        items = llm_items(run, "test")
        directory = resolve_path(cfg.llm.cassette_dir)
        have = RecordedLLM(cfg.llm.model, directory)
        version = cfg.llm.prompt_version
        todo = sum(1 for it in items if not have.has(build_prompt(it.row, it.candidates, version)))
        print(f"provider: {cfg.llm.base_url}  model: {cfg.llm.model}")
        print(f"concurrency: {cfg.llm.max_concurrency}  timeout: {cfg.llm.timeout_s} s")
        print(f"gray-zone rows on the test split: {len(items)} -> {len(items)} LLM calls")
        print(f"cassettes present for this model and prompt: {len(items) - todo}; missing: {todo}")
        print(f"cassettes go to {directory}")
        if items:
            print("--- first prompt ---")
            print(build_prompt(items[0].row, items[0].candidates, cfg.llm.prompt_version))
        return 0
    inner = make_llm(cfg, kind)
    directory = resolve_path(cfg.llm.cassette_dir)
    recorder = RecordingLLM(inner, directory)
    items = llm_items(run, "test")
    results = asyncio.run(_run_arbiter(recorder, items, cfg))
    failed = sum(1 for r in results if r.answer is None)
    print(
        f"recorded {len(items) - failed}/{len(items)} answers from {inner.model!r} to {directory}"
    )
    if results:
        mean = sum(r.latency_s for r in results) / len(results)
        tin = sum(r.input_tokens for r in results)
        tout = sum(r.output_tokens for r in results)
        print(f"live run: mean latency {mean:.2f} s per call, tokens {tin} in / {tout} out")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="entity_cascade.eval.run")
    parser.add_argument("command", choices=["index", "eval", "record", "tune"])
    parser.add_argument("--config", default=str(Path("configs") / "default.yaml"))
    parser.add_argument(
        "--llm",
        choices=["fake", "recorded", "openai"],
        default=None,
        help="override llm.provider (record defaults to openai)",
    )
    parser.add_argument(
        "--secondary",
        default=None,
        help="eval: also run this config (another dataset) and report domain transfer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="record: count the LLM calls and show a prompt, call nothing",
    )
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "reports"), help="eval: where reports are written"
    )
    parser.add_argument(
        "--readme",
        action="store_true",
        help="eval: also regenerate the numbers block of README.md",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "index":
        return cmd_index(cfg)
    if args.command == "eval":
        readme = PROJECT_ROOT / "README.md" if args.readme else None
        return cmd_eval(cfg, args.config, args.llm, args.secondary, Path(args.out), readme)
    if args.command == "tune":
        return cmd_tune(cfg, args.llm or "openai")
    return cmd_record(cfg, args.llm or "openai", args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
