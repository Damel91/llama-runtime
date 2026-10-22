"""Offline oracle, admission and opt-in checks for the reusable live probe."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
from threading import Barrier
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

PATH = Path(__file__).resolve().parents[1] / "probes/qualify_queue_engine.py"
spec = importlib.util.spec_from_file_location("queue_engine_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def vector(index=0, scale=1.0):
    result = [0.0] * probe.DIMENSION
    result[index] = scale
    return result


def test_catalog_profile_pin_excludes_only_native_observation_timestamp():
    first = {
        "data": [
            {
                "id": "replica",
                "created": 1,
                "tags": ["weights"],
                "status": {"value": "loaded", "args": ["--parallel", "1"]},
            }
        ]
    }
    later = deepcopy(first)
    later["data"][0]["created"] = 2
    assert probe.catalog_profile_digest(first) == probe.catalog_profile_digest(later)
    later["data"][0]["status"]["args"][-1] = "4"
    assert probe.catalog_profile_digest(first) != probe.catalog_profile_digest(later)


def test_help_and_missing_optin_need_no_runtime_or_output(tmp_path):
    help_result = subprocess.run(
        [sys.executable, "-I", str(PATH), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    assert "--allow-live" in help_result.stdout
    assert "--connection-policy" in help_result.stdout
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(PATH),
            "--base-url",
            "http://fixture.invalid",
            "--model",
            "embedding",
            "--output",
            str(tmp_path / "report.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "explicit --allow-live" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@host",
        "http://host?",
        "http://host#",
        "http://host:bad",
        "file:///tmp/endpoint",
    ],
)
def test_invalid_endpoint_is_rejected_before_runtime_import_or_files(tmp_path, url):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(PATH),
            "--allow-live",
            "--base-url",
            url,
            "--model",
            "embedding",
            "--output",
            str(tmp_path / "report.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr
    assert not list(tmp_path.iterdir())


def test_fixed_corpus_and_numeric_oracles_preserve_thresholds_and_order():
    assert len(set(probe.corpus())) == 511
    assert probe.corpus() == probe.corpus()
    assert [len(window) for window in probe.source_windows(probe.corpus())] == [
        64
    ] * 7 + [63]
    assert probe.compare(vector(), vector())["passed"]
    measured = probe.compare(vector(), vector(1))
    assert not measured["passed"]
    assert measured["maximum_absolute_error"] == 1
    rows = {
        "first": {"vector": vector(0, 3), "input_format": "tokens"},
        "second": {"vector": vector(1, 7), "input_format": "tokens"},
    }
    assert (
        probe.validate_pipeline(list(rows), [vector(), vector(1)], rows)["correlations"]
        == 2
    )
    with pytest.raises(ValueError, match="projection"):
        probe.validate_pipeline(list(rows), [vector(1), vector()], rows)
    invalid = vector()
    invalid[-1] = float("nan")
    with pytest.raises(ValueError, match="invalid vector"):
        probe.validate_pipeline(list(rows), [invalid, vector(1)], rows)
    assert (probe.MAX_ABS, probe.MIN_COSINE, probe.MAX_NORM, probe.MAX_PROJECTION) == (
        1e-5,
        0.999999,
        1e-6,
        1e-12,
    )


def test_window_audit_requires_complete_boundaries_and_mixed_http():
    windows = [["a", "b"], ["c"]]
    request = {"phase": "pipeline", "kind": "embedding", "source_windows": [0, 1]}
    result = probe.audit_windows(windows, [2, 1], [request])
    assert result["mixed_source_window_posts"] == 1
    assert result["maximum_source_windows_per_post"] == 2
    with pytest.raises(ValueError, match="cardinality"):
        probe.audit_windows(windows, [1, 2], [request])
    with pytest.raises(ValueError, match="mixed source windows"):
        probe.audit_windows(windows, [2, 1], [{**request, "source_windows": [0]}])


def test_response_indices_are_required_unique_and_reordered():
    payload = {
        "data": [
            {"index": 1, "embedding": vector(1)},
            {"index": 0, "embedding": vector()},
        ]
    }
    assert probe.raw_vectors(payload, 2) == [vector(), vector(1)]
    payload["data"][1]["index"] = 1
    with pytest.raises(ValueError, match="response index"):
        probe.raw_vectors(payload, 2)


class FixtureClient:
    def __init__(self):
        self.calls = []

    def request(self, method, path, body=None, **kwargs):
        self.calls.append((method, path, body, kwargs))
        if path == "/tokenize":
            return {"tokens": [1, 2]}
        if path == "/v1/embeddings":
            return {"data": [{"index": 0, "embedding": vector(0, 3)}]}
        return {"okay": True}


def test_observer_correlates_native_ids_and_records_bounded_http():
    budget = probe.Budget()
    client = probe.observed_client_type(FixtureClient)(budget=budget)
    client.request("POST", "/tokenize", {"model": "replica", "content": "first"})
    budget.source_positions = {"first": (3, 7)}
    budget.phase = "pipeline"
    client.request("POST", "/v1/embeddings", {"model": "replica", "input": [[1, 2]]})
    assert budget.raw["first"]["model"] == "replica"
    assert budget.raw["first"]["source_position"] == (3, 7)
    assert (
        probe.validate_pipeline(["first"], [vector()], budget.raw)["correlations"] == 1
    )
    assert budget.counts["tokenize"] == budget.counts["embedding"] == 1
    assert not any(budget.active.values())
    assert budget.records[-1]["tokens"] == 2
    assert budget.records[-1]["source_windows"] == [3]
    assert all(
        call[3]["max_response_bytes"] == 8 * 1024 * 1024 for call in client.calls
    )
    with pytest.raises(ValueError, match="endpoint"):
        client.request("POST", "/models/load", {"model": "replica"})
    assert len(client.calls) == 2


def test_request_cap_is_atomic_across_threads_and_never_replays():
    barrier = Barrier(2)

    class ConcurrentClient(FixtureClient):
        def request(self, *args, **kwargs):
            barrier.wait(timeout=2)
            return super().request(*args, **kwargs)

    budget = probe.Budget(limits={"metadata": 2, "embedding": 1, "tokenize": 1})
    client = probe.observed_client_type(ConcurrentClient)(budget=budget)
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(client.request, "GET", "/props") for _ in range(4)]
        successes = failures = 0
        for future in futures:
            try:
                future.result(timeout=3)
                successes += 1
            except ValueError as exc:
                assert "request budget" in str(exc)
                failures += 1
    assert successes == failures == 2
    assert len(client.calls) == budget.counts["metadata"] == 2
    assert budget.peaks["metadata"] == 2
    assert not any(budget.active.values())


def test_phase_maps_do_not_treat_later_reuse_as_duplicate_inference():
    budget = probe.Budget()
    budget.ids["replica", (1, 2)] = "first"
    client = probe.observed_client_type(FixtureClient)(budget=budget)
    for phase in ("pipeline", "reuse_raw_1"):
        budget.phase = phase
        budget.phase_raw[phase] = {}
        budget.phase_positions[phase] = {"first": (0, 0)}
        client.request(
            "POST", "/v1/embeddings", {"model": "replica", "input": [[1, 2]]}
        )
    assert all(len(rows) == 1 for rows in budget.phase_raw.values())
    assert probe.phase_counts(budget, "pipeline") == {"embedding": 1}
    assert probe.phase_counts(budget, "reuse_raw_1") == {"embedding": 1}


def test_id_audit_cross_alias_lookup_requires_pinned_and_unambiguous_text():
    budget = probe.Budget()
    budget.ids["a", (1, 2)] = "first"
    budget.compatible_aliases = frozenset({"a", "b", "c"})
    assert budget.text_for_ids("b", [1, 2]) == "first"
    with pytest.raises(ValueError, match="uncorrelated"):
        budget.text_for_ids("unknown", [1, 2])
    budget.ids["a", (3, 4)] = "first"
    budget.ids["b", (3, 4)] = "second"
    with pytest.raises(ValueError, match="ambiguous"):
        budget.text_for_ids("c", [3, 4])


def test_observer_audits_mixed_source_positions_and_response_reordering():
    class MixedClient(FixtureClient):
        def request(self, method, path, body=None, **kwargs):
            self.calls.append((method, path, body, kwargs))
            return {
                "data": [
                    {"index": 1, "embedding": vector(1)},
                    {"index": 0, "embedding": vector()},
                ]
            }

    budget = probe.Budget()
    budget.phase = "pipeline"
    budget.source_positions = {"first": (0, 7), "second": (1, 2)}
    budget.ids = {("replica", (1, 2)): "first", ("replica", (3, 4)): "second"}
    client = probe.observed_client_type(MixedClient)(budget=budget)
    client.request(
        "POST", "/v1/embeddings", {"model": "replica", "input": [[1, 2], [3, 4]]}
    )
    assert budget.records[0]["source_windows"] == [0, 1]
    assert budget.raw["first"]["source_position"] == (0, 7)
    assert budget.raw["second"]["source_position"] == (1, 2)
    assert (
        probe.validate_pipeline(["first", "second"], [vector(), vector(1)], budget.raw)[
            "correlations"
        ]
        == 2
    )


def test_oversize_oracle_requires_remote_exact_count_and_zero_embedding_posts():
    class BackendError(Exception):
        pass

    class Adapter:
        def prepare_texts(self, texts, **kwargs):
            assert texts == ["oversize"]
            return [SimpleNamespace(token_count=5, max_input_tokens=4)]

        def embed_prepared(self, records, **kwargs):
            raise BackendError("llamacpp_embedding_input_exceeds_context")

    budget = probe.Budget()
    budget.ids["replica", (1, 2, 3, 4, 5)] = "oversize"
    result = probe.oversize_canary(Adapter(), budget, "oversize", BackendError)
    assert result["exact_token_count"] == 5
    assert result["rejected_without_embedding_http"]
    budget.ids.clear()
    with pytest.raises(ValueError, match="remote tokenization"):
        probe.oversize_canary(Adapter(), budget, "oversize", BackendError)


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("connection_policy", ["close", "reuse"])
def test_complete_probe_runs_against_offline_native_fixture(
    monkeypatch, tmp_path, reuse, connection_policy
):
    import runtime_llama
    import runtime_llama.client
    from runtime_llama.client import LlamaCppClient

    observed_policies = []

    class NativeFixture(LlamaCppClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed_policies.append(self.connection_policy)
            self.text_by_ids = {}

        def request(self, method, path, body=None, **kwargs):
            endpoint = urlsplit(path).path
            model = (body or {}).get("model")
            if endpoint == "/health":
                return {"status": "ok"}
            if endpoint == "/v1/models":
                return {
                    "data": [
                        {
                            "id": alias,
                            "tags": [
                                "noxi-content-sha256-v1:" + "a" * 64,
                                "noxi-embedding-sha256-v1:" + "b" * 64,
                                "noxi-model-name-v1:fixture.gguf",
                            ],
                            "status": {
                                "value": "loaded",
                                "args": ["llama-server", "--batch-size", "2048"],
                            },
                        }
                        for alias in ["embedding_0", "embedding_1"]
                    ]
                }
            if endpoint == "/props":
                alias = parse_qs(urlsplit(path).query)["model"][0]
                return {
                    "model_alias": alias,
                    "total_slots": 1,
                    "build_info": "fixture-build",
                    "model_path": "/private/fixture/model.gguf",
                    "default_generation_settings": {"n_ctx": 2048},
                }
            if endpoint == "/slots":
                return [{"is_processing": False}]
            if endpoint == "/tokenize":
                ids = [1, *[ord(char) + 3 for char in body["content"]], 2]
                self.text_by_ids[model, tuple(ids)] = body["content"]
                return {"tokens": ids}
            assert endpoint == "/v1/embeddings"
            rows = body["input"]
            data = []
            for index, row in enumerate(rows):
                text = (
                    row if isinstance(row, str) else self.text_by_ids[model, tuple(row)]
                )
                position = int(text.split("document ", 1)[1].split(":", 1)[0])
                values = vector(0, 3)
                values[1] = position + 1
                data.append({"index": index, "embedding": values})
            return {"data": list(reversed(data))}

    monkeypatch.setattr(runtime_llama.client, "LlamaCppClient", NativeFixture)
    monkeypatch.setattr(
        runtime_llama, "__file__", "/fixture/site-packages/runtime_llama/__init__.py"
    )
    distribution = SimpleNamespace(
        version=runtime_llama.__version__, read_text=lambda _: "{}"
    )
    monkeypatch.setattr(
        probe.importlib.metadata, "distribution", lambda _: distribution
    )
    args = SimpleNamespace(
        base_url="http://fixture.invalid",
        model="fixture.gguf",
        prefix="",
        wall_timeout=30,
        qualify_reuse=reuse,
        connection_policy=connection_policy,
        output=tmp_path / "report.json",
    )
    report, artifacts = probe.run(args)
    assert report["result"] == "passed", report.get("failure")
    assert report["http_connection_policy"] == connection_policy
    assert observed_policies == [connection_policy]
    assert report["profiles_unchanged"]
    assert report["pipeline"]["correlations"] == 511
    assert report["pipeline"]["source_window_sizes"] == [64] * 7 + [63]
    assert report["pipeline"]["mixed_source_window_posts"] > 0
    assert report["effective_batch_wait_sec"] == 0
    assert report["oversize"]["exact_token_count"] > 2048
    assert report["request_counts"]["tokenize"] == 528 + (8 if reuse else 0)
    assert report["pipeline"]["request_counts"]["tokenize"] == 511
    assert report["pipeline"]["request_counts"]["metadata"] < 50
    assert all(item["passed"] for item in report["pipeline"]["canary_comparisons"])
    if reuse:
        assert report["reuse"]["prepare_request_counts"]["tokenize"] == 8
        assert len(report["reuse"]["phases"]) == 4
        assert report["reuse"]["cache_after_close"]["entries"] == 0
        for result in report["reuse"]["phases"].values():
            assert result["correlations"] == 8
            assert result["request_counts"].get("tokenize", 0) == 0
    assert not any(report["active_at_finish"].values())
    assert len(artifacts["outputs"]) == len(artifacts["raw_by_text"]) == 511
    assert "/private/fixture/model.gguf" not in str(report)
    assert not list(tmp_path.iterdir())
