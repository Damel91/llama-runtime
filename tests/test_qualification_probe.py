from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from threading import Barrier
import time
from urllib.parse import parse_qs, urlsplit

import pytest

from runtime_llama import LLMCallResult, LLMUsageStats, build_embedding_adapter
from runtime_llama import LLMStreamEvent, ThinkingDiagnostic, ThinkingDiagnosticsConfig


PROBE_PATH = Path(__file__).resolve().parents[1] / "probes/qualify_replicas_thinking.py"
spec = importlib.util.spec_from_file_location("qualification_probe", PROBE_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def arguments(tmp_path):
    return argparse.Namespace(
        base_url="http://fixture.invalid:8080",
        llm_model="fixture-llm",
        replica=["fixture-embed_0", "fixture-embed_1"],
        expected_build="fixture-build",
        output=tmp_path / "report.json",
        count=32,
        window=32,
    )


def cli_args(tmp_path):
    return [
        "--base-url",
        "http://fixture.invalid:8080",
        "--llm-model",
        "fixture-llm",
        "--replica",
        "fixture-embed_0",
        "--replica",
        "fixture-embed_1",
        "--expected-build",
        "fixture-build",
        "--output",
        str(tmp_path / "report.json"),
    ]


def vector(index):
    result = [0.0] * probe.DIMENSION
    result[index % probe.DIMENSION] = 1.0
    return result


def test_import_and_help_require_only_standard_library(tmp_path):
    # Isolated Python cannot see the editable runtime package or site additions.
    result = subprocess.run(
        [sys.executable, "-I", str(PROBE_PATH), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--live" in result.stdout
    assert not list(tmp_path.iterdir())


def test_missing_live_optin_never_imports_runtime_or_writes_files(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(PROBE_PATH), *cli_args(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "explicit --live" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@host:8080",
        "http://host?",
        "http://host#",
        "file:///tmp/server",
        "http://host:bad",
        "http://host\\bad",
        "http://host/ bad",
    ],
)
def test_live_invalid_url_rejected_before_runtime_import_or_files(tmp_path, url):
    args = cli_args(tmp_path)
    args[args.index("--base-url") + 1] = url
    result = subprocess.run(
        [sys.executable, "-I", str(PROBE_PATH), "--live", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "extra",
    [
        ["--replica", "third"],
        ["--count", "0"],
        ["--window", "8"],
    ],
)
def test_invalid_shape_does_not_reach_run(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(probe, "run", lambda _: pytest.fail("live run must not start"))
    with pytest.raises(SystemExit) as caught:
        probe.main(["--live", *cli_args(tmp_path), *extra])
    assert caught.value.code == 2
    assert not list(tmp_path.iterdir())


def test_corpus_is_reproducible_unique_and_position_anchored():
    rows = [probe.corpus_text(index) for index in range(10_000)]
    assert len(set(rows)) == 10_000
    assert [probe.corpus_index(row) for row in rows] == list(range(10_000))
    assert rows[1] == probe.corpus_text(1)
    with pytest.raises(ValueError, match="identity"):
        probe.corpus_index(rows[1] + "changed")


def test_vector_oracle_rejects_permutation_nonfinite_and_bad_normalization():
    assert probe.compare_vectors(vector(1), vector(1)) == (0.0, 1.0)
    with pytest.raises(ValueError, match="absolute"):
        probe.compare_vectors(vector(1), vector(2))
    bad = vector(1)
    bad[4] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        probe.validate_vector(bad)
    with pytest.raises(ValueError, match="normalization"):
        probe.validate_vector([0.0] * probe.DIMENSION)


def test_numeric_measurement_survives_strict_failure_without_threshold_change():
    measured = probe.measure_vectors(vector(1), vector(2))
    assert measured["passed"] is False
    assert measured["maximum_absolute_error"] == 1
    assert measured["cosine"] == 0
    assert measured["absolute_error_limit"] == 1e-5
    assert measured["cosine_minimum"] == 0.999999
    with pytest.raises(probe.VectorComparisonError) as caught:
        probe.compare_vectors(vector(1), vector(2))
    assert caught.value.measurements == measured
    assert "max_abs=1" in str(caught.value)


def test_same_request_oracle_independently_normalizes_and_rejects_permutation():
    raw = [value * 3 for value in vector(1)]
    probe.same_request_projection(raw, vector(1))
    with pytest.raises(ValueError, match="order or projection"):
        probe.same_request_projection(raw, vector(2))


class MetadataClient:
    def __init__(self, args):
        self.args = args
        self.busy = False
        self.build = args.expected_build
        self.loaded = True
        self.context = 512
        self.metadata_dimension = 768
        self.calls = []

    def health(self, **kwargs):
        self.calls.append("/health")
        return {"status": "ok"}

    def models(self, **kwargs):
        self.calls.append("/v1/models")
        return {
            "data": [
                {
                    "id": alias,
                    "status": {"value": "loaded" if self.loaded else "unloaded"},
                    "meta": {"n_embd": self.metadata_dimension},
                    "tags": [
                        "noxi-content-sha256-v1:" + "a" * 64,
                        "noxi-embedding-sha256-v1:" + "b" * 64,
                    ]
                    if alias in self.args.replica
                    else [],
                }
                for alias in [self.args.llm_model, *self.args.replica]
            ]
        }

    def props(self, model=None, **kwargs):
        self.calls.append("/props")
        return {
            "build_info": self.build,
            "total_slots": 4,
            "default_generation_settings": {"n_ctx": self.context},
            "model_path": "/fixture/private/model.gguf",
        }

    def request(self, method, path, **kwargs):
        self.calls.append(path)
        assert method == "GET"
        assert parse_qs(urlsplit(path).query)["model"][0] in [
            self.args.llm_model,
            *self.args.replica,
        ]
        return [{"id": index, "is_processing": self.busy} for index in range(4)]


def test_snapshot_guards_loaded_alias_build_context_slots_and_dimension(tmp_path):
    args = arguments(tmp_path)
    client = MetadataClient(args)
    before = probe.snapshot(
        client, [args.llm_model, *args.replica], args.replica, args.expected_build
    )
    assert before["models"][args.replica[0]]["context"] == 512
    assert "/fixture/private/model.gguf" not in str(before)
    probe.invariant_snapshots(before, deepcopy(before))
    after = deepcopy(before)
    after["models"][args.replica[0]]["props_sha256"] = "changed"
    with pytest.raises(ValueError, match="properties"):
        probe.invariant_snapshots(before, after)
    for field, value, match in [
        ("busy", True, "busy"),
        ("build", "other", "build"),
        ("loaded", False, "loaded"),
        ("context", 256, "context"),
        ("metadata_dimension", 42, "dimension"),
    ]:
        broken = MetadataClient(args)
        setattr(broken, field, value)
        with pytest.raises(ValueError, match=match):
            probe.snapshot(
                broken,
                [args.llm_model, *args.replica],
                args.replica,
                args.expected_build,
            )
    assert all(
        not path.startswith(("/models/load", "/models/unload")) for path in client.calls
    )


def thinking_record(source="separated"):
    return {
        "text": "The remaining count is 332.",
        "reasoning_text": "Arithmetic checked" if source == "separated" else "",
        "terminal_reason": "completed",
        "sink_records": 1,
        "diagnostic": {
            "call_id": "fixture-call",
            "model": "model",
            "role_id": "role",
            "text": "Arithmetic checked",
            "sources": [source],
            "captured_chars": 18,
            "observed_chars": 18,
            "max_chars": 256,
            "truncated": False,
            "status": "completed",
            "terminal_reason": "completed",
        },
    }


@pytest.mark.parametrize("source", ["separated", "inline"])
def test_thinking_oracle_enforces_correlation_capture_source_and_contract(source):
    record = thinking_record(source)
    probe.validate_thinking(record, source=source, model="model", role="role")
    for mutate, match in [
        (lambda r: r.update(sink_records=2), "sink"),
        (lambda r: r["diagnostic"].update(model="other"), "correlation"),
        (lambda r: r["diagnostic"].update(sources=[]), "unsupported"),
        (lambda r: r["diagnostic"].update(truncated=True), "truncation"),
        (lambda r: r.update(text="<think>private</think>answer"), "visible"),
    ]:
        broken = deepcopy(record)
        mutate(broken)
        with pytest.raises(ValueError, match=match):
            probe.validate_thinking(broken, source=source, model="model", role="role")
    if source == "inline":
        record["reasoning_text"] = "changed"
        with pytest.raises(ValueError, match="contract"):
            probe.validate_thinking(record, source=source, model="model", role="role")


def test_llm_visible_cap_keeps_unknown_usage_without_inventing_counters():
    result = LLMCallResult(
        "Synthetic partial answer.", terminal_reason="visible_output_cap"
    )
    record = probe.llm_output(result)
    assert record["usage"] is None
    assert record["usage_available"] is False
    assert (
        record["usage_unavailable_reason"]
        == "client_visible_output_cap_before_provider_terminal"
    )
    diagnostic = thinking_record("inline")
    diagnostic.update(text=result.text, terminal_reason=result.terminal_reason)
    diagnostic["diagnostic"].update(
        status="interrupted", terminal_reason=result.terminal_reason
    )
    probe.validate_thinking(diagnostic, source="inline", model="model", role="role")


@pytest.mark.parametrize("terminal", ["completed", "length_cap"])
def test_provider_terminal_without_usage_fails_and_preserves_record(terminal):
    records = []
    with pytest.raises(ValueError, match="usage absent"):
        probe.llm_output(
            LLMCallResult("Synthetic answer.", terminal_reason=terminal),
            records=records,
        )
    assert len(records) == 1
    assert records[0]["usage"] is None
    assert records[0]["terminal_reason"] == terminal
    assert records[0]["usage_unavailable_reason"] == "provider_usage_not_observed"


def test_all_thinking_modes_accept_client_visible_cap_without_usage(tmp_path):
    args = arguments(tmp_path)

    class CappedThinkingAdapter:
        def __init__(self, config):
            self.config = config

        def result(self, options):
            source = (
                "inline"
                if options["request_options"]["reasoning_format"] == "none"
                else "separated"
            )
            diagnostic = ThinkingDiagnostic(
                call_id=self.config["role_id"],
                role_id=self.config["role_id"],
                model=self.config["model"],
                text="Arithmetic checked",
                sources=(source,),
                observed_chars=18,
                captured_chars=18,
                max_chars=256,
                truncated=False,
                status="interrupted",
                terminal_reason="visible_output_cap",
            )
            options["thinking_sink"](diagnostic)
            return LLMCallResult(
                "Synthetic partial visible answer.",
                reasoning_text="Arithmetic checked" if source == "separated" else "",
                terminal_reason="visible_output_cap",
                thinking_diagnostic=diagnostic,
            )

        def generate_with_metadata(self, prompt, **kwargs):
            return self.result(kwargs)

        def stream_with_metadata(self, prompt, **kwargs):
            result = self.result(kwargs)
            if result.reasoning_text:
                yield LLMStreamEvent(kind="reasoning", text=result.reasoning_text)
            yield LLMStreamEvent(kind="content", text=result.text)
            yield LLMStreamEvent(
                kind="terminal",
                terminal_reason=result.terminal_reason,
                thinking_diagnostic=result.thinking_diagnostic,
            )

        def close(self):
            pass

    report = {"thinking": []}
    probe.thinking_cases(
        args,
        {"llm": CappedThinkingAdapter, "thinking": ThinkingDiagnosticsConfig},
        report,
    )
    assert len(report["thinking"]) == 4
    assert all(record["usage"] is None for record in report["thinking"])
    assert all(record["usage_available"] is False for record in report["thinking"])
    assert all(
        record["diagnostic"]["status"] == "interrupted" for record in report["thinking"]
    )


class InferenceClient:
    def __init__(self, *args, **kwargs):
        self.barrier = Barrier(4)
        self.closed = False

    def health(self, **kwargs):
        return {"status": "ok"}

    def models(self, **kwargs):
        if hasattr(self, "args"):
            return MetadataClient.models(self, **kwargs)
        return {
            "data": [
                {
                    "id": "fixture-embed_" + str(index),
                    "status": {"value": "loaded"},
                    "tags": [
                        "noxi-content-sha256-v1:" + "a" * 64,
                        "noxi-embedding-sha256-v1:" + "b" * 64,
                    ],
                }
                for index in range(2)
            ]
        }

    def props(self, **kwargs):
        return {"total_slots": 4, "default_generation_settings": {"n_ctx": 512}}

    def tokenize(self, text, **kwargs):
        return {"tokens": list(range(50))}

    def embeddings(self, texts, **kwargs):
        self.barrier.wait(timeout=3)
        return {
            "data": [
                {"index": index, "embedding": vector(probe.corpus_index(text))}
                for index, text in reversed(list(enumerate(texts)))
            ]
        }

    def chat_completion(self, *args, **kwargs):
        for _ in range(10):
            time.sleep(0.01)
            yield {"content": "fixture"}

    def close(self):
        self.closed = True


def inference_adapter(config, client=None):
    client = client or InferenceClient()
    # Native slots no longer multiply HTTP admission on one replica. The
    # barrier still proves real overlap across independently verified aliases.
    underlying = getattr(client, "client", client)
    underlying.barrier = Barrier(len(config["embedding_replicas"]))
    return build_embedding_adapter(config, client=client)


class LLMAdapter:
    def __init__(self, config, client):
        self.model = config["model"]
        self.client = client

    def generate_with_metadata(self, prompt, **kwargs):
        list(self.client.chat_completion([], model=self.model, stream=True))
        return LLMCallResult(
            "Synthetic complete answer.",
            usage=LLMUsageStats(completion_tokens=3),
            terminal_reason="completed",
        )

    def close(self):
        pass


def test_offline_four_phases_preserve_every_vector_and_measure_real_call_overlap(
    tmp_path,
):
    args = arguments(tmp_path)
    report = {"phases": []}
    runtime = {
        "client": InferenceClient,
        "embedding": inference_adapter,
        "llm": lambda config, client: LLMAdapter(config, client),
    }
    for label, caps, baseline, mixed in [
        ("alias0", {args.replica[0]: 4}, True, False),
        ("alias1", {args.replica[1]: 4}, False, False),
        ("pooled", dict.fromkeys(args.replica, 2), False, False),
        ("mixed", dict.fromkeys(args.replica, 2), False, True),
    ]:
        phase = probe.run_phase(
            args,
            tmp_path,
            report,
            label,
            caps,
            baseline=baseline,
            mixed=mixed,
            runtime=runtime,
        )
        assert phase["result"] == "passed"
        assert phase["processed"] == 32
        assert phase["activity"]["maximum_embedding_requests"] == len(caps)
        assert phase["activity"]["active_at_finish"] == 0
        assert phase["activity"]["response_associations_checked"] == 32
        assert phase["activity"]["same_request_order_projection_checks"] == 32
        assert phase["activity"]["unconsumed_raw_vectors"] == 0
        assert phase["maximum_absolute_error"] == 0
        if mixed:
            assert phase["activity"]["mixed_overlap_observations"] > 0
            assert len(phase["llm_calls"]) == 3
    assert (tmp_path / "alias0-reference.f64le").stat().st_size == 32 * 768 * 8


def test_campaign_continues_numeric_failures_and_never_reports_false_pass(
    monkeypatch, tmp_path
):
    import runtime_llama
    import runtime_llama.client

    args = arguments(tmp_path)

    class CampaignClient(InferenceClient, MetadataClient):
        def __init__(self, *positional, **kwargs):
            InferenceClient.__init__(self)
            MetadataClient.__init__(self, args)

        def props(self, **kwargs):
            return MetadataClient.props(self, **kwargs)

        def embeddings(self, texts, **kwargs):
            if len(texts) > 1:
                self.barrier.wait(timeout=3)
            data = []
            for index, text in enumerate(texts):
                position = probe.corpus_index(text)
                # Independent anchors differ at index zero; the second alias
                # differs at index seven. Both numeric failures must be retained.
                changed = (len(texts) == 1 and position == 0) or (
                    kwargs["model"] == args.replica[1] and position == 7
                )
                data.append(
                    {"index": index, "embedding": vector(position + int(changed))}
                )
            return {"data": data}

    monkeypatch.setattr(runtime_llama.client, "LlamaCppClient", CampaignClient)
    monkeypatch.setattr(
        runtime_llama,
        "build_embedding_adapter",
        lambda config, client=None: inference_adapter(
            config, client=client or CampaignClient()
        ),
    )
    monkeypatch.setattr(
        runtime_llama,
        "build_adapter",
        lambda config, client=None: LLMAdapter(config, client),
    )
    monkeypatch.setattr(
        probe,
        "thinking_cases",
        lambda args, runtime, report: report["thinking"].append(
            {"offline_executed": True}
        ),
    )
    assert probe.run(args) == 1
    report = json.loads(args.output.read_text())
    assert report["result"] == "failed"
    assert report["failure"]["class"] == "NumericOracleFailure"
    assert report["numeric_failures"] >= 2
    assert report["compared_vectors"] == 32 * 3
    assert len(report["phases"]) == 4
    assert all(phase["processed"] == 32 for phase in report["phases"])
    second = report["phases"][1]
    assert second["result"] == "failed"
    assert second["numeric_failures"] == 1
    assert second["numeric_failure_samples"][0]["index"] == 7
    assert second["numeric_failure_samples"][0]["maximum_absolute_error"] == 1
    assert len(report["independent_single_input_anchors"]) == 3
    assert report["independent_single_input_anchors"][0]["result"] == "failed"
    assert report["thinking"] == [{"offline_executed": True}]
    assert report["server_configuration_unchanged"] is True


def test_phase_fails_immediately_for_output_permuted_against_own_request(tmp_path):
    args = arguments(tmp_path)

    class PermutedAdapter:
        def __init__(self, config, client):
            self.adapter = inference_adapter(config, client=client)

        def embed_texts(self, texts, **kwargs):
            values = self.adapter.embed_texts(texts, **kwargs)
            values[0], values[1] = values[1], values[0]
            return values

        def close(self):
            self.adapter.close()

    runtime = {
        "client": InferenceClient,
        "embedding": lambda config, client: PermutedAdapter(config, client),
    }
    report = {"phases": []}
    with pytest.raises(ValueError, match="same-request output order"):
        probe.run_phase(
            args,
            tmp_path,
            report,
            "permuted",
            {args.replica[0]: 4},
            baseline=True,
            mixed=False,
            runtime=runtime,
        )
    assert report["phases"][0]["result"] == "failed"
    assert report["phases"][0]["processed"] == 0
