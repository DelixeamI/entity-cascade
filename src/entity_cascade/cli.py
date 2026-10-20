"""Command line: `match --input price.csv --output result.csv`."""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from entity_cascade.calibration import Calibration
from entity_cascade.config import PROJECT_ROOT, AppConfig, load_config, resolve_path
from entity_cascade.data.loader import load_dataset, read_csv_rows
from entity_cascade.data.normalize import build_brand_vocab, extract_brand
from entity_cascade.data.split import make_splits
from entity_cascade.eval.llm_metrics import cost_usd
from entity_cascade.eval.run import build_models, make_llm
from entity_cascade.eval.zone_metrics import Decision, system_metrics
from entity_cascade.matcher import MatchOutcome, match_rows
from entity_cascade.pipeline import build_generator
from entity_cascade.retrieval.dense import make_client
from entity_cascade.schemas import Record

OUTPUT_COLUMNS = [
    "id",
    "name",
    "zone",
    "verdict",
    "handled_by",
    "matched_id",
    "matched_name",
    "score",
    "margin",
    "reason",
]


def read_price_list(
    path: Path, cfg: AppConfig, id_col: str, name_col: str, desc_col: str, catalog: Sequence[Record]
) -> list[Record]:
    raw = read_csv_rows(path)
    if not raw:
        raise ValueError(f"{path} has no data rows")
    missing = [c for c in (name_col,) if c not in raw[0]]
    if missing:
        raise ValueError(f"{path} has no column {missing[0]!r}; columns: {list(raw[0])}")
    vocab = build_brand_vocab(r.brand for r in catalog)
    rows: list[Record] = []
    for i, row in enumerate(raw):
        name, desc = row[name_col], row.get(desc_col, "")
        fields = {**row, "name": name, "description": desc, cfg.dataset.left_name: name}
        rows.append(
            Record(
                id=row.get(id_col) or str(i + 1),
                name=name,
                description=desc,
                brand=extract_brand(name, vocab),
                fields=fields,
            )
        )
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        dup = next(i for i, n in Counter(ids).items() if n > 1)
        raise ValueError(f"duplicate row id {dup!r} in {path}")
    return rows


def write_result(path: Path, outcomes: Sequence[MatchOutcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for o in outcomes:
            matched = o.candidate if o.verdict != "no_match" else None
            writer.writerow(
                {
                    "id": o.row.id,
                    "name": o.row.name,
                    "zone": o.zone.value,
                    "verdict": o.verdict,
                    "handled_by": o.handled_by,
                    "matched_id": matched.id if matched else "",
                    "matched_name": matched.name if matched else "",
                    "score": f"{o.score:.3f}",
                    "margin": f"{o.margin:.3f}",
                    "reason": o.reason,
                }
            )


def summarize(
    outcomes: Sequence[MatchOutcome],
    cfg: AppConfig,
    seconds: float,
    truth: dict[str, frozenset[str]] | None,
    note: str | None,
) -> str:
    n = len(outcomes)
    zones = Counter(o.zone.value for o in outcomes)
    verdicts = Counter[str](o.verdict for o in outcomes)
    handled = Counter[str](o.handled_by for o in outcomes)

    def pct(key: str, counter: Counter[str]) -> str:
        return f"{counter.get(key, 0):4d} ({100 * counter.get(key, 0) / n:5.1f}%)"

    lines = [
        f"rows: {n}   time: {seconds:.1f} s ({1000 * seconds / n:.0f} ms per row)",
        "zones:     "
        + "   ".join(
            f"{z}={pct(z, zones)}"
            for z in ("auto_match", "llm_arbiter", "human_review", "no_match")
        ),
        "verdicts:  "
        + "   ".join(f"{v}={pct(v, verdicts)}" for v in ("match", "no_match", "needs_human")),
        "handled:   " + "   ".join(f"{h}={pct(h, handled)}" for h in ("auto", "llm", "human")),
    ]
    tin = sum(o.llm_input_tokens for o in outcomes)
    tout = sum(o.llm_output_tokens for o in outcomes)
    if tin or tout:
        lines.append(
            f"llm tokens: {tin} in / {tout} out, cost ${cost_usd(tin, tout, cfg.llm):.4f} "
            "(prices from the config)"
        )
    if truth is not None:
        decisions = {
            o.row.id: Decision(
                o.row.id,
                o.zone,
                o.candidate.id if o.verdict == "match" and o.candidate else None,
                o.handled_by,
            )
            for o in outcomes
        }
        m = system_metrics(decisions, truth)
        lines.append(
            "against ground truth (a human is assumed right): "
            f"automated {100 * m.automation_rate:.1f}%, auto match precision "
            f"{100 * m.automated_match_precision:.1f}%, system precision "
            f"{100 * m.precision:.1f}%, recall {100 * m.recall:.1f}%"
        )
    if note:
        lines.append("note: " + note)
    return "\n".join(lines)


def cmd_match(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cal_path = resolve_path(Path(args.calibration))
    if not cal_path.exists():
        print(f"{cal_path} not found: run `make eval` first to fit and save the calibration")
        return 1
    cal = Calibration.load(cal_path)

    truth: dict[str, frozenset[str]] | None = None
    if args.demo:
        ds = load_dataset(cfg.dataset)
        sp = make_splits(ds, cfg.val_fraction, cfg.seed, cfg.dataset.new_product_fraction)
        catalog = list(sp.catalog)
        rows = list(sp.test.rows[: args.demo])
        truth = {r.id: sp.test.truth[r.id] for r in rows}
        print(f"demo: first {len(rows)} rows of the {cfg.dataset.name} test split")
    else:
        if not args.input:
            print("--input is required (or use --demo N)")
            return 1
        catalog = list(load_dataset(cfg.dataset).right)
        rows = read_price_list(
            Path(args.input), cfg, args.id_col, args.name_col, args.desc_col, catalog
        )

    t0 = time.perf_counter()
    models = build_models(cfg)
    gen, _ = build_generator(cfg, catalog, models.embedder, make_client(cfg.retrieval.qdrant_url))
    llm = None if args.llm == "none" else make_llm(cfg, args.llm)
    outcomes, note = match_rows(cfg, cal, rows, catalog, gen, models.reranker, llm)
    seconds = time.perf_counter() - t0

    out = Path(args.output)
    write_result(out, outcomes)
    print(summarize(outcomes, cfg, seconds, truth, note))
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="entity-cascade")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("match", help="match a price list against the catalog")
    m.add_argument("--input", help="price list CSV")
    m.add_argument("--output", required=True, help="result CSV")
    m.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    m.add_argument("--calibration", default="reports/calibration.json")
    m.add_argument("--id-col", default="id")
    m.add_argument("--name-col", default="name")
    m.add_argument("--desc-col", default="description")
    m.add_argument(
        "--llm",
        choices=["none", "fake", "recorded", "openai"],
        default="none",
        help="arbiter for the gray zone; none sends those rows to a human",
    )
    m.add_argument("--demo", type=int, default=0, help="use the first N rows of the test split")
    args = parser.parse_args(argv)
    if args.command == "match":
        return cmd_match(args)
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
