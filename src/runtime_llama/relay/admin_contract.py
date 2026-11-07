"""Version-one browser/admin contract; no HTTP or framework imports."""

from __future__ import annotations

from copy import deepcopy


ADMIN_API_VERSION = 1
ADMIN_PREFIX = "/api/admin"
ADMIN_TOKEN_HEADER = "X-Relay-Admin-Token"
ADMIN_ROUTES = (
    ("GET", "/api/admin/session"),
    ("GET", "/api/admin/config"),
    ("PUT", "/api/admin/config"),
    ("GET", "/api/admin/status"),
    ("GET", "/api/admin/catalog"),
    ("POST", "/api/admin/catalog/refresh"),
    ("GET", "/api/admin/requests"),
    ("POST", "/api/admin/requests/{request_id}/cancel"),
    ("DELETE", "/api/admin/requests/last-completed"),
    ("POST", "/api/admin/tailnet/start"),
    ("POST", "/api/admin/tailnet/stop"),
    ("GET", "/api/admin/opencode"),
    ("POST", "/api/admin/opencode"),
    ("POST", "/api/admin/opencode/apply"),
)

_CONTRACT = {
    "api_version": ADMIN_API_VERSION,
    "schema_version": 1,
    "token_header": ADMIN_TOKEN_HEADER,
    "session_response": {"admin_token": "string", "api_version": 1},
    "config_response": {
        "revision": "integer",
        "config": "redacted configuration; secret keys are omitted",
        "secret_fields": "JSON pointer strings for configured nonempty secrets",
    },
    "config_write": {"expected_revision": "integer", "config": "configuration"},
    "config_fields": [
        "schema_version",
        "listener",
        "access",
        "runtime_endpoints",
        "llm_defaults",
        "embedding_defaults",
        "model_overrides",
        "tailnet",
    ],
    "secret_write_semantics": "omitted preserves; null clears; value replaces",
    "requests_response": {
        "active": "array",
        "last_completed": "summary or null; not history",
    },
    "catalog_response": {
        "revision": "configuration revision",
        "observed_at": "UTC timestamp or null before explicit refresh",
        "runtimes": "all endpoint observations, including unavailable endpoints",
    },
    "status_response": {
        "revision": "configuration revision",
        "listener": "active binding metadata",
        "restart_required": "boolean",
        "tailnet": "content-free helper state",
        "requests": "same shape as requests_response",
    },
    "error_response": {"error": {"code": "string", "message": "safe string"}},
    "opencode_versions": ["v1", "v2"],
    "opencode_export_request": {
        "version": "v1 or v2",
        "catalog_revision": "configuration revision of the observed catalog",
        "default_model": "selected exact model ID",
        "models": "map of model IDs to observed_context, context and output",
    },
}


def admin_contract() -> dict:
    """Return independent machine-readable contract metadata."""
    value = deepcopy(_CONTRACT)
    value["routes"] = [
        {"method": method, "path": path} for method, path in ADMIN_ROUTES
    ]
    return value
