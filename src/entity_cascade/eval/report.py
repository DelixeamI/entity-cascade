"""Results model and Markdown/JSON report writer. Every number in the docs comes from here."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from entity_cascade.calibration import Calibration
from entity_cascade.data.stats import DatasetStats
from entity_cascade.eval.llm_metrics import ArbiterMetrics
from entity_cascade.eval.llm_tuning import TuningResult
from entity_cascade.eval.modes import ModeRow
from entity_cascade.eval.tradeoff import TradeoffRow
from entity_cascade.eval.types import NanFloat
from entity_cascade.eval.zone_metrics import SystemMetrics, ZoneRow
from entity_cascade.zones.assign import NEVER


class RecallRow(BaseModel):
    mode: str
    split: str
    n_rows: int  # rows with a true pair
    recall: dict[int, NanFloat]


class SplitInfo(BaseModel):
    split: str
    rows: int
    new_products: int  # synthetic rows without a pair
    rows_with_pair: int


class Top1Row(BaseModel):
    mode: str
    split: str
    n_rows: int  # rows with a true pair
    accuracy: NanFloat


class Timings(BaseModel):
    """Wall-clock on the machine that ran the eval (CPU, single process)."""

    index_build_s: NanFloat
    retrieval_ms_per_row: NanFloat
    rerank_ms_per_row: NanFloat
    bench_rows: int  # uncached sample the per-row numbers were measured on


class ZoneBlock(BaseModel):
    split: str
    zones: list[ZoneRow]


class SystemRow(BaseModel):
    split: str
    name: str
    metrics: SystemMetrics


class LLMReport(BaseModel):
    model: str
    stand_in: bool  # True: a test double, not a real LLM; numbers show plumbing only
    replayed: bool = False  # answers came from cassettes: latency was not measured in this run
    metrics: ArbiterMetrics


class TransferRow(BaseModel):
    """Test-split behaviour of the second dataset under a given calibration."""

    name: str
    high: NanFloat
    margin_min: NanFloat
    auto_share: NanFloat
    auto_precision: NanFloat
    human_share: NanFloat  # cascade without the LLM
    system_precision: NanFloat
    system_recall: NanFloat


class EvalResults(BaseModel):
    config_path: str
    dataset_name: str
    seed: int
    embedding_model: str
    reranker_model: str
    dataset_stats: DatasetStats
    catalog_size: int
    new_product_fraction: NanFloat
    splits: list[SplitInfo]
    candidate_recall: list[RecallRow]
    modes: list[ModeRow]
    top1: list[Top1Row]
    calibration: Calibration
    target_precision: NanFloat
    zone_blocks: list[ZoneBlock]
    systems: list[SystemRow]
    tradeoff: list[TradeoffRow]
    llm: LLMReport | None
    llm_note: str | None
    llm_tuning: TuningResult | None = None
    timings: Timings
    transfer: list[TransferRow] = []
    secondary: EvalResults | None = None
    charts: list[str] = []


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def _thr(x: float, feasible: bool = True) -> str:
    return "unreachable" if not feasible or abs(x) >= NEVER else f"{x:.2f}"


def _dataset_section(r: EvalResults, h: str) -> list[str]:
    s = r.dataset_stats
    lines = [
        f"{h} Dataset: {r.dataset_name}",
        "",
        f"- price-list rows: {s.left_rows}; catalog items in the raw data: {s.right_rows}; "
        f"true pairs: {s.pairs}",
        f"- rows with a pair / without: {s.left_with_pair} / {s.left_without_pair}; "
        f"rows with several true items: {s.left_with_multiple_pairs}",
        f"- true pairs sharing a model code: {s.pairs_sharing_code} of {s.pairs}",
    ]
    if r.new_product_fraction > 0:
        lines.append(
            f"- **synthetic** new products: {_pct(r.new_product_fraction)} of each split "
            f"(their true items are removed from the catalog, which then has "
            f"{r.catalog_size} items)"
        )
    else:
        lines.append(f"- catalog size used: {r.catalog_size} items; no synthetic rows")
    lines += ["", "| split | rows | with pair | synthetic new products |", "|---|---:|---:|---:|"]
    lines += [
        f"| {sp.split} | {sp.rows} | {sp.rows_with_pair} | {sp.new_products} |" for sp in r.splits
    ]
    return lines


def _recall_section(r: EvalResults, h: str) -> list[str]:
    ks = sorted({k for row in r.candidate_recall for k in row.recall})
    lines = [
        f"{h} Candidate recall",
        "",
        "Share of rows with a true pair whose true item is in the top-k of the mode.",
        "",
        "| mode | split | rows | " + " | ".join(f"recall@{k}" for k in ks) + " |",
        "|---|---|---:|" + "---:|" * len(ks),
    ]
    for row in r.candidate_recall:
        cells = " | ".join(_pct(row.recall[k]) for k in ks)
        lines.append(f"| {row.mode} | {row.split} | {row.n_rows} | {cells} |")
    return lines


def _top1_section(r: EvalResults, h: str) -> list[str]:
    w = r.calibration.weights
    lines = [
        f"{h} Top-1 accuracy (rows with a true pair)",
        "",
        "Feature weights are chosen on val only (added to the cross-encoder logit): "
        f"code_exact=+{w.code_exact}, code_prefix=+{w.code_prefix}, "
        f"code_conflict=-{w.code_conflict}, brand_match=+{w.brand_match}, "
        f"brand_conflict=-{w.brand_conflict}.",
        "",
        "| mode | split | rows | top-1 accuracy |",
        "|---|---|---:|---:|",
    ]
    lines += [f"| {t.mode} | {t.split} | {t.n_rows} | {_pct(t.accuracy)} |" for t in r.top1]
    return lines


def _modes_section(r: EvalResults, h: str) -> list[str]:
    lines = [
        f"{h} Mode comparison at equal footing",
        "",
        "Each mode is trusted by its own score and margin. Thresholds are fitted on val so that "
        f"the auto zone reaches {_pct(r.target_precision)} precision there, then applied to test. "
        "Share = fraction of all rows answered automatically.",
        "",
        "| mode | top-1 (test) | val: auto share | val: precision | test: auto share "
        "| test: precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in r.modes:
        if m.feasible:
            cells = (
                f"{_pct(m.val_auto_share)} | {_pct(m.val_auto_precision)} | "
                f"{_pct(m.test_auto_share)} | {_pct(m.test_auto_precision)}"
            )
        else:
            cells = "target unreachable on val | n/a | 0.0% | n/a"
        lines.append(f"| {m.mode} | {_pct(m.top1_accuracy_test)} | {cells} |")
    return lines


def _thresholds_section(r: EvalResults, h: str) -> list[str]:
    t = r.calibration.thresholds
    flags = f"auto={t.auto_feasible}, llm={t.llm_feasible}, no_match={t.no_match_feasible}"
    lines = [
        f"{h} Thresholds (fitted on val only)",
        "",
        f"- auto_match: score >= {_thr(t.high, t.auto_feasible)} and margin >= {t.margin_min:.2f}; "
        f"target precision {_pct(r.target_precision)} on val",
        f"- llm_arbiter: not auto and score >= {_thr(t.low, t.llm_feasible)}",
        f"- no_match: not auto and score < {_thr(t.floor, t.no_match_feasible)}",
        "- human_review: everything in between",
        f"- zone reachable on val: {flags} (False = target not reachable, zone left empty)",
    ]
    for zb in r.zone_blocks:
        lines += [
            "",
            f"**Zones, {zb.split}**",
            "",
            "| zone | rows | share | top-1 correct | no true pair |",
            "|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {z.zone} | {z.rows} | {_pct(z.share)} | {_pct(z.top1_correct)} "
            f"| {_pct(z.without_pair)} |"
            for z in zb.zones
        ]
    return lines


def _system_section(r: EvalResults, h: str) -> list[str]:
    lines = [
        f"{h} End-to-end (a human is always right)",
        "",
        "Rows sent to a human count as answered correctly. Precision = correct matches / all "
        "matches the system or the human give; recall = correct matches / rows with a pair. "
        "Automated = answered without a human (rules or LLM).",
        "",
        "| split | system | automated | of which LLM | human | auto match precision "
        "| automated accuracy | precision | recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in r.systems:
        m = s.metrics
        lines.append(
            f"| {s.split} | {s.name} | {_pct(m.automation_rate)} | {_pct(m.llm_share)} "
            f"| {_pct(m.human_share)} | {_pct(m.automated_match_precision)} "
            f"| {_pct(m.automated_accuracy)} | {_pct(m.precision)} | {_pct(m.recall)} |"
        )
    return lines


def _tradeoff_section(r: EvalResults, h: str) -> list[str]:
    lines = [
        f"{h} The trade-off: target precision vs automation",
        "",
        "Thresholds refitted on val for each target; the cascade without the LLM, so every "
        "non-auto row goes to a human. Val is where thresholds were fitted; test is held out.",
        "",
        "| target precision | val: auto share | val: precision | test: auto share "
        "| test: precision | test: human share | test: system precision | test: system recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in r.tradeoff:
        if not t.feasible:
            lines.append(f"| {t.label} | unreachable on val | | | | | | |")
            continue
        lines.append(
            f"| {t.label} | {_pct(t.val_auto_share)} | {_pct(t.val_auto_precision)} "
            f"| {_pct(t.test_auto_share)} | {_pct(t.test_auto_precision)} "
            f"| {_pct(t.test_human_share)} | {_pct(t.test_system_precision)} "
            f"| {_pct(t.test_system_recall)} |"
        )
    return lines


def _llm_section(r: EvalResults, h: str) -> list[str]:
    lines = [f"{h} LLM arbiter (test split)", ""]
    if r.llm is None:
        lines.append(
            f"**Not run: no recorded LLM answers** ({r.llm_note or 'no reason given'}). "
            "Numbers appear here only after `make record` with a real provider; "
            "nothing is estimated in the meantime."
        )
        return lines
    a = r.llm.metrics
    if r.llm.stand_in:
        lines += [
            f"**`{r.llm.model}` is a test double, not an LLM.** The numbers below check the "
            "plumbing only and say nothing about LLM quality.",
            "",
        ]
    lines += [
        f"- model: `{r.llm.model}`; min confidence to accept an answer: {a.min_confidence}",
        f"- rows sent to the arbiter: {a.rows_in_zone} of {a.rows_in_run} test rows",
        f"- taken off the human queue: {a.resolved} ({_pct(a.resolved_share)}); "
        f"escalated back to a human: {a.escalated} "
        f"(of which provider or format errors: {a.errors})",
        f"- accuracy of resolved answers: {_pct(a.resolved_accuracy)} "
        f'("this item" answers: {_pct(a.resolved_match_precision)}, '
        f'"none" answers: {_pct(a.resolved_none_precision)})',
        f"- for comparison, top-1 of the cascade on the same rows: "
        f"{_pct(a.top1_accuracy_same_rows)}; a true item is among the shown candidates for "
        f"{_pct(a.true_item_shown_share)} of rows with a pair",
        f"- tokens: {a.input_tokens} in / {a.output_tokens} out over {a.calls} calls; "
        + (
            "latency not measured here (answers are replayed from cassettes; "
            "`make record` prints the latency of the live run)"
            if r.llm.replayed
            else f"mean latency {a.mean_latency_s:.2f} s per call"
        ),
        f"- cost: ${a.cost_usd_run:.4f} for this run, ${a.cost_usd_per_1000_rows:.4f} per "
        "1000 price-list rows (prices from the config; $0 means no prices were configured)",
    ]
    return lines


def _tuning_section(r: EvalResults, h: str) -> list[str]:
    t = r.llm_tuning
    if t is None:
        return []
    lines = [
        f"{h} LLM prompt tuning (val split only)",
        "",
        f"Prompt version and `min_confidence` are chosen on val rows ({t.rows_in_zone} gray-zone "
        f"rows per version), never on test. Rule: resolve as many rows as possible while the "
        f"answers on the resolved rows are at least {_pct(t.target_accuracy)} accurate "
        f"(at least {t.min_resolved_support} resolved rows). Test labels are not used here.",
        "",
        '| version | min confidence | resolved | resolved share | accuracy | "this item" '
        'precision | "none" precision | val system precision | val system recall '
        "| val human share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in t.cells:
        mark = (
            "**"
            if (c.version, c.min_confidence) == (t.chosen_version, t.chosen_min_confidence)
            else ""
        )
        lines.append(
            f"| {mark}{c.version}{mark} | {mark}{c.min_confidence}{mark} | {c.resolved} "
            f"| {_pct(c.resolved_share)} | {_pct(c.resolved_accuracy)} "
            f"| {_pct(c.match_precision)} | {_pct(c.none_precision)} "
            f"| {_pct(c.system_precision)} | {_pct(c.system_recall)} | {_pct(c.human_share)} |"
        )
    verdict = (
        f"Chosen (bold): version {t.chosen_version}, min_confidence {t.chosen_min_confidence}."
        if t.feasible
        else f"**No setting reached {_pct(t.target_accuracy)} accuracy on val.** The most "
        f"accurate one is chosen (bold): version {t.chosen_version}, "
        f"min_confidence {t.chosen_min_confidence}."
    )
    return lines + ["", verdict]


def _latency_section(r: EvalResults, h: str) -> list[str]:
    tm = r.timings
    return [
        f"{h} Latency (CPU, this machine, single process)",
        "",
        f"Measured without caches on {tm.bench_rows} val rows.",
        "",
        f"- index build (BM25 + dense embeddings of the catalog): {tm.index_build_s:.1f} s",
        f"- retrieval: {tm.retrieval_ms_per_row:.1f} ms per row",
        f"- cross-encoder reranking of the top-k candidates: {tm.rerank_ms_per_row:.1f} ms per row",
        f"- cascade without the LLM, per row: "
        f"{tm.retrieval_ms_per_row + tm.rerank_ms_per_row:.1f} ms",
    ]


def _transfer_section(sec: EvalResults, h: str) -> list[str]:
    lines = [
        f"{h} Second dataset: {sec.dataset_name}",
        "",
        "Same code, same models; only the config differs. The first row applies the thresholds "
        "(and feature weights) fitted on the first dataset's val split without any change; the "
        "second row fits them on this dataset's own val split. Metrics are on this dataset's "
        "test split, cascade without the LLM.",
        "",
        "| calibration | high | margin | auto share | auto precision | human share "
        "| system precision | system recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in sec.transfer:
        lines.append(
            f"| {t.name} | {_thr(t.high)} | {t.margin_min:.2f} | {_pct(t.auto_share)} "
            f"| {_pct(t.auto_precision)} | {_pct(t.human_share)} "
            f"| {_pct(t.system_precision)} | {_pct(t.system_recall)} |"
        )
    lines += [""] + _dataset_section(sec, h + "#")
    lines += [""] + _recall_section(sec, h + "#")
    lines += [""] + _modes_section(sec, h + "#")
    lines += [""] + _top1_section(sec, h + "#")
    lines += [""] + _thresholds_section(sec, h + "#")
    lines += [""] + _system_section(sec, h + "#")
    lines += [""] + _tradeoff_section(sec, h + "#")
    return lines


def render_markdown(r: EvalResults) -> str:
    lines: list[str] = [
        "# Results",
        "",
        "Generated by `make eval`. Do not edit by hand.",
        "",
        f"- config: `{r.config_path}`, seed {r.seed}",
        f"- embedding model: `{r.embedding_model}`; reranker: `{r.reranker_model}`",
        "",
    ]
    for chart in r.charts:
        lines += [f"![{chart}]({chart})", ""]
    sections = [
        _dataset_section(r, "##"),
        _recall_section(r, "##"),
        _modes_section(r, "##"),
        _top1_section(r, "##"),
        _thresholds_section(r, "##"),
        _system_section(r, "##"),
        _tradeoff_section(r, "##"),
        _llm_section(r, "##"),
        _tuning_section(r, "##"),
        _latency_section(r, "##"),
    ]
    if r.secondary is not None:
        sections.append(_transfer_section(r.secondary, "##"))
    for sec in sections:
        lines += sec + [""]
    return "\n".join(lines)


def write_reports(r: EvalResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(r.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (out_dir / "results.md").write_text(render_markdown(r), encoding="utf-8")
