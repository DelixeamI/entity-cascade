import asyncio
import json
import random
from pathlib import Path

import httpx
import pytest

from entity_cascade.config import LLMConfig
from entity_cascade.eval.llm_metrics import apply_arbiter, arbiter_metrics, cost_usd
from entity_cascade.eval.zone_metrics import Decision
from entity_cascade.llm.arbiter import (
    ArbiterItem,
    ArbiterResult,
    arbitrate,
    build_prompt,
    is_resolved,
    parse_answer,
)
from entity_cascade.llm.base import CassetteMiss, LLMError, LLMResponse, cassette_key
from entity_cascade.llm.fake import FakeLLM, HeuristicFakeLLM
from entity_cascade.llm.openai_compat import OpenAICompatibleLLM
from entity_cascade.llm.recorded import RecordedLLM, RecordingLLM
from entity_cascade.rerank.features import MatchFeatures
from entity_cascade.rerank.ranker import RowRanking, ScoredCandidate
from entity_cascade.schemas import ArbiterAnswer, Record, Zone


def cfg(**kw: object) -> LLMConfig:
    base: dict[str, object] = {
        "model": "m",
        "max_retries": 3,
        "backoff_base_s": 1.0,
        "max_concurrency": 2,
        "price_input_per_1m": 2.0,
        "price_output_per_1m": 10.0,
    }
    return LLMConfig.model_validate({**base, **kw})


def rec(rid: str, name: str, **extra: str) -> Record:
    return Record(id=rid, name=name, description=extra.pop("description", ""), fields=extra)


# ---- cassettes ----


def test_cassette_key_separates_model_and_prompt() -> None:
    assert cassette_key("ab", "c") != cassette_key("a", "bc")
    assert cassette_key("m", "p") == cassette_key("m", "p")
    assert cassette_key("m1", "p") != cassette_key("m2", "p")
    assert len(cassette_key("m", "p")) == 64


async def test_recording_then_replay_roundtrip(tmp_path: Path) -> None:
    inner = FakeLLM(model="model-x")
    rec_client = RecordingLLM(inner, tmp_path)
    original = await rec_client.complete("hello prompt")
    replay = RecordedLLM("model-x", tmp_path)
    assert replay.has("hello prompt")
    assert await replay.complete("hello prompt") == original
    saved = json.loads((tmp_path / f"{cassette_key('model-x', 'hello prompt')}.json").read_text())
    assert saved["prompt"] == "hello prompt" and saved["model"] == "model-x"


async def test_recorded_miss_is_an_error_not_an_answer(tmp_path: Path) -> None:
    client = RecordedLLM("m", tmp_path)
    with pytest.raises(CassetteMiss, match="make record"):
        await client.complete("never recorded")


async def test_recorded_is_keyed_by_model(tmp_path: Path) -> None:
    await RecordingLLM(FakeLLM(model="a"), tmp_path).complete("p")
    with pytest.raises(CassetteMiss):
        await RecordedLLM("b", tmp_path).complete("p")


# ---- OpenAI-compatible client ----


def ok_body(text: str = '{"choice": 1, "confidence": 0.9}') -> dict[str, object]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 15},
    }


def make_client(handler: object, **kw: object) -> tuple[OpenAICompatibleLLM, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    client = OpenAICompatibleLLM(
        cfg(**kw),
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        sleep=fake_sleep,
        rng=random.Random(0),
    )
    return client, sleeps


async def test_openai_success_parses_text_and_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=ok_body())

    client, sleeps = make_client(handler)
    resp = await client.complete("prompt")
    assert resp == LLMResponse(
        text='{"choice": 1, "confidence": 0.9}', input_tokens=120, output_tokens=15
    )
    body = json.loads(seen[0].content)
    assert body["model"] == "m" and body["temperature"] == 0
    assert body["messages"] == [{"role": "user", "content": "prompt"}]
    assert sleeps == []


async def test_openai_retries_429_then_succeeds_with_backoff() -> None:
    codes = iter([429, 429, 200])

    def handler(_req: httpx.Request) -> httpx.Response:
        code = next(codes)
        return httpx.Response(code, json=ok_body() if code == 200 else {"error": "slow down"})

    client, sleeps = make_client(handler)
    assert (await client.complete("p")).output_tokens == 15
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] < 1.5  # base 1 s with jitter
    assert 1.0 <= sleeps[1] < 3.0  # doubled


async def test_openai_honours_retry_after() -> None:
    codes = iter([429, 200])

    def handler(_req: httpx.Request) -> httpx.Response:
        if next(codes) == 429:
            return httpx.Response(429, headers={"retry-after": "7"}, json={})
        return httpx.Response(200, json=ok_body())

    client, sleeps = make_client(handler)
    await client.complete("p")
    assert sleeps == [7.0]


async def test_openai_gives_up_after_max_retries_on_5xx() -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="overloaded")

    client, sleeps = make_client(handler, max_retries=2)
    with pytest.raises(LLMError, match="gave up after 3 attempts"):
        await client.complete("p")
    assert calls == 3
    assert len(sleeps) == 2  # no sleep after the last attempt


async def test_openai_does_not_retry_client_errors() -> None:
    calls = 0

    def handler(_req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="bad key")

    client, sleeps = make_client(handler)
    with pytest.raises(LLMError, match="HTTP 401"):
        await client.complete("p")
    assert calls == 1 and sleeps == []


async def test_openai_retries_timeouts() -> None:
    attempts = iter([0, 1])

    def handler(req: httpx.Request) -> httpx.Response:
        if next(attempts) == 0:
            raise httpx.ReadTimeout("slow", request=req)
        return httpx.Response(200, json=ok_body())

    client, sleeps = make_client(handler)
    await client.complete("p")
    assert len(sleeps) == 1


async def test_openai_malformed_body_is_an_llm_error() -> None:
    client, _ = make_client(lambda _r: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(LLMError, match="malformed"):
        await client.complete("p")


# ---- prompt and parsing ----


def test_build_prompt_lists_candidates_in_order() -> None:
    row = rec("r", "Sony Turntable - PSLX350H", description="Belt drive", price="$99")
    cands = [rec("a", "Sony PSLX350H", manufacturer="Sony"), rec("b", "Other thing")]
    prompt = build_prompt(row, cands)
    assert "Price-list item:\nname: Sony Turntable - PSLX350H" in prompt
    assert "price: $99" in prompt
    assert "1. name: Sony PSLX350H; brand: Sony" in prompt
    assert "2. name: Other thing" in prompt
    assert prompt.index("1. name") < prompt.index("2. name")


def test_build_prompt_truncates_long_descriptions() -> None:
    row = rec("r", "x", description="y" * 1000)
    assert len(build_prompt(row, [rec("a", "z")])) < 2000


@pytest.mark.parametrize(
    ("text", "n", "expected"),
    [
        ('{"choice": 2, "confidence": 0.8, "reason": "same model"}', 3, (2, 0.8)),
        ('{"choice": "none", "confidence": 0.6}', 3, ("none", 0.6)),
        ('```json\n{"choice": 1, "confidence": 1}\n```', 3, (1, 1.0)),
        ('Sure! {"choice": 1, "confidence": 0.5} hope it helps', 3, (1, 0.5)),
        ('{"choice": 4, "confidence": 0.9}', 3, None),  # out of range
        ('{"choice": 0, "confidence": 0.9}', 3, None),
        ('{"choice": 1, "confidence": 1.5}', 3, None),
        ('{"choice": "maybe", "confidence": 0.5}', 3, None),
        ("not json at all", 3, None),
        ("", 3, None),
        ('{"choice": 1', 3, None),
    ],
)
def test_parse_answer(text: str, n: int, expected: tuple[int | str, float] | None) -> None:
    got = parse_answer(text, n)
    if expected is None:
        assert got is None
    else:
        assert got is not None and (got.choice, got.confidence) == expected


# ---- arbitrate ----


def items(n: int) -> list[ArbiterItem]:
    return [
        ArbiterItem(rec(f"r{i}", f"row {i}"), (rec(f"r{i}a", "a"), rec(f"r{i}b", "b")))
        for i in range(n)
    ]


class SlowClient:
    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self._fail_for = fail_for or set()

    @property
    def model(self) -> str:
        return "slow"

    async def complete(self, prompt: str) -> LLMResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        if any(f"name: {f}" in prompt for f in self._fail_for):
            raise LLMError("boom")
        return LLMResponse(
            text='{"choice": 2, "confidence": 0.9}', input_tokens=100, output_tokens=10
        )


async def test_arbitrate_respects_the_concurrency_limit() -> None:
    client = SlowClient()
    results = await arbitrate(client, items(10), cfg(max_concurrency=3))
    assert len(results) == 10
    assert client.max_in_flight == 3
    assert [r.row_id for r in results] == [f"r{i}" for i in range(10)]  # order preserved
    assert all(r.chosen_id == f"{r.row_id}b" for r in results)


async def test_provider_failure_escalates_only_that_row() -> None:
    results = await arbitrate(SlowClient(fail_for={"row 1"}), items(3), cfg())
    by_id = {r.row_id: r for r in results}
    assert by_id["r1"].answer is None and "boom" in (by_id["r1"].error or "")
    assert by_id["r0"].answer is not None and by_id["r2"].answer is not None
    assert not is_resolved(by_id["r1"], 0.5)


async def test_unusable_output_is_recorded_as_error() -> None:
    results = await arbitrate(FakeLLM(lambda _p: "I think maybe the first"), items(1), cfg())
    assert results[0].answer is None and results[0].error == "unusable model output"


async def test_cassette_miss_propagates(tmp_path: Path) -> None:
    with pytest.raises(CassetteMiss):
        await arbitrate(RecordedLLM("m", tmp_path), items(2), cfg())


async def test_heuristic_fake_llm_picks_the_matching_candidate() -> None:
    row = rec("r", "Sony Turntable - PSLX350H")
    cands = [rec("a", "Panasonic Microwave NNSD670S"), rec("b", "Sony PSLX350H Turntable")]
    res = await arbitrate(HeuristicFakeLLM(), [ArbiterItem(row, tuple(cands))], cfg())
    assert res[0].chosen_id == "b"
    nothing = await arbitrate(
        HeuristicFakeLLM(),
        [ArbiterItem(rec("r2", "Canon camera SD1100"), (rec("c", "Blender X900"),))],
        cfg(),
    )
    assert nothing[0].answer is not None and nothing[0].answer.choice == "none"


# ---- metrics ----

NF = MatchFeatures(
    code_exact=False,
    code_prefix=False,
    code_conflict=False,
    brand_match=False,
    brand_conflict=False,
)


def result(
    rid: str, choice: int | str | None, conf: float, tin: int = 100, tout: int = 10
) -> ArbiterResult:
    answer = None if choice is None else ArbiterAnswer(choice=choice, confidence=conf)  # type: ignore[arg-type]
    return ArbiterResult(rid, (f"{rid}a", f"{rid}b"), answer, None, tin, tout, 0.5)


def test_arbiter_metrics_and_cost() -> None:
    truth = {
        "r1": frozenset({"r1b"}),  # right match
        "r2": frozenset({"r2a"}),  # wrong match (picked b)
        "r3": frozenset(),  # right none
        "r4": frozenset({"r4a"}),  # low confidence -> escalated
        "r5": frozenset({"r5a"}),  # provider failure -> escalated
    }
    results = [
        result("r1", 2, 0.9),
        result("r2", 2, 0.8),
        result("r3", "none", 0.9),
        result("r4", 1, 0.3),
        result("r5", None, 0.0, 0, 0),
    ]
    rankings = {
        rid: RowRanking(rid, (ScoredCandidate(f"{rid}a", 1, 0.0, NF, 0.0),)) for rid in truth
    }
    m = arbiter_metrics(results, truth, rankings, cfg(min_confidence=0.7), rows_in_run=100)
    assert (m.rows_in_zone, m.resolved, m.escalated, m.errors) == (5, 3, 2, 1)
    assert m.resolved_share == pytest.approx(0.6)
    assert m.resolved_accuracy == pytest.approx(2 / 3)
    assert m.resolved_match_precision == pytest.approx(0.5)
    assert m.resolved_none_precision == 1.0
    assert m.top1_accuracy_same_rows == pytest.approx(3 / 5)  # top-1 "<id>a": r2, r4, r5 right
    assert m.true_item_shown_share == 1.0
    assert (m.input_tokens, m.output_tokens) == (400, 40)
    assert m.cost_usd_run == pytest.approx(400 * 2 / 1e6 + 40 * 10 / 1e6)
    assert m.cost_usd_per_1000_rows == pytest.approx(m.cost_usd_run / 100 * 1000)


def test_cost_formula() -> None:
    assert cost_usd(1_000_000, 500_000, cfg()) == pytest.approx(2.0 + 5.0)


def test_apply_arbiter_only_takes_resolved_rows() -> None:
    base = {rid: Decision(rid, Zone.LLM_ARBITER, None, "human") for rid in ("r1", "r2", "r3")}
    out = apply_arbiter(
        base, [result("r1", 2, 0.9), result("r2", 1, 0.2), result("r3", "none", 0.95)], 0.7
    )
    assert (out["r1"].handled_by, out["r1"].pred_id) == ("llm", "r1b")
    assert out["r2"].handled_by == "human"
    assert (out["r3"].handled_by, out["r3"].pred_id) == ("llm", None)
    assert base["r1"].handled_by == "human"  # input not mutated


# ---- record preflight ----


def _preflight_cfg() -> object:
    from entity_cascade.config import PROJECT_ROOT, load_config

    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    return cfg.model_copy(update={"llm": cfg.llm.model_copy(update={"model": "qwen2.5:14b"})})


def test_preflight_ok_when_model_is_served() -> None:
    from entity_cascade.eval.run import check_provider

    handler = lambda _r: httpx.Response(200, json={"data": [{"id": "qwen2.5:14b"}]})  # noqa: E731
    assert check_provider(_preflight_cfg(), httpx.MockTransport(handler)) is None  # type: ignore[arg-type]


def test_preflight_names_the_missing_model_and_the_fix() -> None:
    from entity_cascade.eval.run import check_provider

    handler = lambda _r: httpx.Response(200, json={"data": [{"id": "llama3"}]})  # noqa: E731
    msg = check_provider(_preflight_cfg(), httpx.MockTransport(handler))  # type: ignore[arg-type]
    assert msg and "ollama pull qwen2.5:14b" in msg and "llama3" in msg


def test_preflight_reports_unreachable_server() -> None:
    from entity_cascade.eval.run import check_provider

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    msg = check_provider(_preflight_cfg(), httpx.MockTransport(handler))  # type: ignore[arg-type]
    assert msg and "cannot reach" in msg and "running" in msg
