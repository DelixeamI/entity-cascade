import numpy as np
import pytest

from entity_cascade.config import ThresholdsConfig
from entity_cascade.data.split import ValSplit
from entity_cascade.eval.zone_metrics import system_metrics, zone_decisions, zone_table
from entity_cascade.rerank.features import MatchFeatures
from entity_cascade.rerank.ranker import RowRanking, ScoredCandidate
from entity_cascade.schemas import Record, Zone
from entity_cascade.zones.assign import NEVER, Thresholds, assign_zone
from entity_cascade.zones.thresholds import fit_thresholds, wilson_lower

NF = MatchFeatures(
    code_exact=False,
    code_prefix=False,
    code_conflict=False,
    brand_match=False,
    brand_conflict=False,
)


def ranking(rid: str, scores: list[float], ids: list[str] | None = None) -> RowRanking:
    ids = ids or [f"{rid}_c{i}" for i in range(len(scores))]
    return RowRanking(
        rid,
        tuple(
            ScoredCandidate(cid, i + 1, s, NF, s)
            for i, (cid, s) in enumerate(zip(ids, scores, strict=True))
        ),
    )


def test_wilson_lower_bound() -> None:
    assert wilson_lower(98, 100, 0.0) == 0.98
    lb = wilson_lower(98, 100, 1.96)
    assert 0.92 < lb < 0.98
    assert wilson_lower(0, 0, 1.96) == 0.0
    assert wilson_lower(100, 100, 1.96) < 1.0


# ---- assign ----

T = Thresholds(high=5.0, margin_min=1.0, low=2.0, floor=-3.0, auto_feasible=True)


@pytest.mark.parametrize(
    ("scores", "zone"),
    [
        ([6.0, 1.0], Zone.AUTO_MATCH),
        ([6.0, 5.5], Zone.LLM_ARBITER),  # high score but thin margin
        ([5.0, 4.0], Zone.AUTO_MATCH),  # boundaries are inclusive
        ([3.0, 0.0], Zone.LLM_ARBITER),
        ([2.0, 0.0], Zone.LLM_ARBITER),
        ([1.9, 0.0], Zone.HUMAN_REVIEW),
        ([-3.0, -9.0], Zone.HUMAN_REVIEW),
        ([-3.1, -9.0], Zone.NO_MATCH),
    ],
)
def test_assign_zone(scores: list[float], zone: Zone) -> None:
    assert assign_zone(ranking("r", scores), T) == zone


def test_assign_zone_without_candidates_is_no_match() -> None:
    assert assign_zone(RowRanking("r", ()), T) == Zone.NO_MATCH


def test_default_thresholds_never_auto_match() -> None:
    assert assign_zone(ranking("r", [1e6, 0.0]), Thresholds()) != Zone.AUTO_MATCH


# ---- fitting on a synthetic val set ----


def make_val(n_good: int = 60, n_noise: int = 60, n_new: int = 30, seed: int = 0):
    """High scores are right, mid scores are mixed, low scores belong to rows without a pair."""
    rng = np.random.default_rng(seed)
    rows, truth, rankings = [], {}, {}
    k = 0

    def add(score: float, margin: float, kind: str) -> None:
        nonlocal k
        rid = f"r{k}"
        k += 1
        rows.append(Record(id=rid, name=rid))
        right = f"{rid}_c0"
        if kind == "match":
            truth[rid] = frozenset({right})
        elif kind == "second":  # the true item is the runner-up
            truth[rid] = frozenset({f"{rid}_c1"})
        else:
            truth[rid] = frozenset()
        rankings[rid] = ranking(rid, [score, score - margin, score - margin - 1])

    for _ in range(n_good):
        add(float(rng.uniform(8, 12)), float(rng.uniform(2, 4)), "match")
    for _ in range(n_noise):
        add(
            float(rng.uniform(0, 6)),
            float(rng.uniform(0, 1)),
            "second" if rng.random() < 0.7 else "match",
        )
    for _ in range(n_new):
        add(float(rng.uniform(-8, -3)), float(rng.uniform(0, 1)), "none")
    return ValSplit(rows=tuple(rows), truth=truth), rankings


CFG = ThresholdsConfig(
    target_precision=0.98,
    min_auto_support=10,
    llm_min_answerable=0.5,
    min_llm_support=5,
    no_match_target_precision=0.95,
    min_no_match_support=5,
)


def test_fit_thresholds_meets_target_on_val_and_orders_the_cutoffs() -> None:
    val, rankings = make_val()
    t = fit_thresholds(val, rankings, CFG)
    assert t.auto_feasible and t.llm_feasible and t.no_match_feasible
    assert t.floor <= t.low <= t.high

    zones = {r.id: assign_zone(rankings[r.id], t) for r in val.rows}
    auto = [rid for rid, z in zones.items() if z is Zone.AUTO_MATCH]
    assert len(auto) >= CFG.min_auto_support
    correct = sum(rankings[rid].top1_id in val.truth[rid] for rid in auto)
    assert correct / len(auto) >= CFG.target_precision
    nm = [rid for rid, z in zones.items() if z is Zone.NO_MATCH]
    assert sum(not val.truth[rid] for rid in nm) / len(nm) >= CFG.no_match_target_precision


def test_fit_thresholds_auto_zone_is_the_largest_feasible() -> None:
    val, rankings = make_val()
    t = fit_thresholds(val, rankings, CFG)
    n_auto = sum(assign_zone(rankings[r.id], t) is Zone.AUTO_MATCH for r in val.rows)
    # All "good" rows (score >= 8, margin >= 2) are right; mixed rows below are not reliable.
    assert n_auto >= 55


def test_infeasible_target_disables_auto_match() -> None:
    # Every row is wrong, so no zone can reach 98% precision.
    rows = tuple(Record(id=f"r{i}", name="x") for i in range(40))
    truth = {r.id: frozenset({"never"}) for r in rows}
    rankings = {r.id: ranking(r.id, [float(i), 0.0]) for i, r in enumerate(rows)}
    t = fit_thresholds(ValSplit(rows=rows, truth=truth), rankings, CFG)
    assert not t.auto_feasible
    assert t.high == NEVER
    assert all(assign_zone(rankings[r.id], t) is not Zone.AUTO_MATCH for r in rows)


def test_wilson_bound_is_more_conservative() -> None:
    val, rankings = make_val()
    plain = fit_thresholds(val, rankings, CFG)
    strict = fit_thresholds(val, rankings, CFG.model_copy(update={"wilson_z": 1.96}))

    def n_auto(t: Thresholds) -> int:
        return sum(assign_zone(rankings[r.id], t) is Zone.AUTO_MATCH for r in val.rows)

    assert n_auto(strict) <= n_auto(plain)


def test_fit_is_deterministic() -> None:
    val, rankings = make_val()
    assert fit_thresholds(val, rankings, CFG) == fit_thresholds(val, rankings, CFG)


# ---- metrics ----


def test_zone_table_and_system_metrics() -> None:
    rows = ["a", "b", "c", "d", "e"]
    truth = {
        "a": frozenset({"a_c0"}),  # auto, right
        "b": frozenset({"b_c1"}),  # auto, wrong item
        "c": frozenset(),  # no_match, right
        "d": frozenset({"d_c0"}),  # human
        "e": frozenset({"e_c0"}),  # no_match, but has a pair -> missed
    }
    rankings = {
        "a": ranking("a", [9.0, 1.0]),
        "b": ranking("b", [9.0, 1.0]),
        "c": ranking("c", [-9.0, -10.0]),
        "d": ranking("d", [0.0, -1.0]),
        "e": ranking("e", [-9.0, -10.0]),
    }
    t = Thresholds(high=5.0, margin_min=1.0, low=2.0, floor=-3.0, auto_feasible=True)
    decisions = zone_decisions(rows, rankings, t)
    zones = {rid: d.zone for rid, d in decisions.items()}
    table = {z.zone: z for z in zone_table(zones, rankings, truth)}
    assert table["auto_match"].rows == 2 and table["auto_match"].top1_correct == 0.5
    assert table["no_match"].rows == 2 and table["no_match"].without_pair == 0.5
    assert table["human_review"].rows == 1

    m = system_metrics(decisions, truth)
    assert m.automation_rate == pytest.approx(4 / 5)
    assert m.human_share == pytest.approx(1 / 5)
    assert m.automated_match_precision == pytest.approx(0.5)
    assert m.automated_accuracy == pytest.approx(2 / 4)  # a ok, b wrong, c ok, e missed
    # TP = a + human d = 2; predicted matches = a, b, d = 3; rows with a pair = a, b, d, e = 4
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 4)
    assert m.automated_wrong == 2
