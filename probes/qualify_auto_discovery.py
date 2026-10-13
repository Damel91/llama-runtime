"""Small opt-in smoke of automatic discovery on the configured remote server.

This exercises a shared installed adapter with a handful of short texts. It
does not load models, benchmark ingest, or measure retrieval quality or RAM.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import runtime_llama
from runtime_llama import build_embedding_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("live requests require --allow-live")
    if args.max_input_tokens < 1:
        parser.error("--max-input-tokens must be positive")
    result = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model_selector": args.model,
        "module": runtime_llama.__file__,
        "version": runtime_llama.__version__,
        "runtime_admin_actions": False,
        "consumer_changes": False,
        "retrieval_quality_measured": False,
        "memory_measured": False,
        "scope": "Bounded automatic discovery and short embedding smoke",
    }
    profile = {
        "base_url": args.base_url,
        "model": args.model,
        "embedding_replica_discovery": True,
        "batch_size": 1,
        "max_input_tokens": args.max_input_tokens,
        "batch_token_limit": args.max_input_tokens,
        "timeout": 30,
        "normalize": True,
        "embedding_prefixes": {
            "query": "task: search result | query: ",
            "document": "title: none | text: ",
        },
    }
    if args.api_key_env:
        profile["api_key_env"] = args.api_key_env
    adapter = build_embedding_adapter(profile)
    try:
        before = adapter.health_status()
        result["before"] = before
        if not before.get("model_ready"):
            result.update(result="blocked", reason="model_not_ready")
            return 2
        verified = before["embedding_replica_discovery"]["identity_verified"]
        if args.require_verified and not verified:
            result.update(result="blocked", reason="content_identity_tags_missing")
            return 2
        documents = [
            "Noxi gestisce il processo di inferenza.",
            "La libreria gestisce le repliche embedding.",
            "Il consumer sceglie il modello logico.",
            "Il catalogo usa lo stesso endpoint configurato.",
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            doc_job = executor.submit(adapter.embed_texts, documents)
            query_job = executor.submit(
                adapter.embed_query, "Chi gestisce le repliche?"
            )
            document_vectors, query_vector = doc_job.result(), query_job.result()
        vectors = document_vectors + [query_vector]
        assert len(document_vectors) == len(documents)
        dimensions = {len(vector) for vector in vectors}
        assert len(dimensions) == 1 and next(iter(dimensions)) > 0
        norms = []
        for vector in vectors:
            assert all(math.isfinite(value) for value in vector)
            norm = math.sqrt(sum(value * value for value in vector))
            assert abs(norm - 1.0) <= 1e-6
            norms.append(norm)
        after = adapter.health_status()
        result.update(
            result="passed",
            verified_pool=verified,
            vectors=vectors,
            dimension=next(iter(dimensions)),
            vector_count=len(vectors),
            norms=norms,
            after=after,
        )
        return 0
    except Exception as exc:
        # Never copy remote bodies, prompts or credentials into diagnostics.
        result.update(result="failed", error_type=type(exc).__name__)
        return 1
    finally:
        adapter.close()
        result["closed"] = True
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "result": result.get("result"),
                    "verified_pool": result.get("verified_pool"),
                    "output": str(args.output),
                    "closed": result["closed"],
                }
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
