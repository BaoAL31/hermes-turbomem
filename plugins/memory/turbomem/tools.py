"""OpenAI-style tool schemas for the turbomem memory provider (full PRD v1 surface)."""

from __future__ import annotations

from typing import Any

MEMORY_STORE_SCHEMA: dict[str, Any] = {
    "name": "memory_store",
    "description": (
        "Local experiences only (facts, preferences, fixes, conversation summaries). "
        "Actions: retain, recall, list. For source code use code_recall / code_peek."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["retain", "recall", "list"]},
            "text": {"type": "string", "description": "Required for retain."},
            "query": {"type": "string", "description": "Required for recall."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "retain: optional labels (e.g. topic:auth). recall/list: filter — any tag overlap (v2).",
            },
            "project_path": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["action"],
    },
}

_PROJECT_FILTER = {
    "project_path": {"type": "string", "description": "Optional repo root."},
    "project_id": {"type": "string", "description": "Optional stable project id."},
}

ALL_PROVIDER_SCHEMAS: list[dict[str, Any]] = [
    MEMORY_STORE_SCHEMA,
    {
        "name": "index_codebase",
        "description": "Index a project root into the global store (semantic code chunks).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        },
    },
    {
        "name": "code_recall",
        "description": "Hybrid search over Code Entries across indexed projects.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                **_PROJECT_FILTER,
            },
            "required": ["query"],
        },
    },
    {
        "name": "code_peek",
        "description": "Code search returning paths and symbols only (no source bodies).",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                **_PROJECT_FILTER,
            },
            "required": ["query"],
        },
    },
    {
        "name": "code_call_graph",
        "description": "List callers or callees of a symbol (TS/JS/Python/Go/Rust when supported).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "direction": {
                    "type": "string",
                    "enum": ["callers", "callees"],
                    "default": "callers",
                },
                "symbol_id": {"type": "integer"},
                **_PROJECT_FILTER,
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_code_projects",
        "description": "List all indexed projects in the catalog.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "index_status",
        "description": "Report index readiness, chunk counts, branch, embedding cache.",
        "parameters": {"type": "object", "properties": dict(_PROJECT_FILTER)},
    },
    {
        "name": "index_health_check",
        "description": "Remove stale/orphan index rows for a project or globally.",
        "parameters": {"type": "object", "properties": dict(_PROJECT_FILTER)},
    },
    {
        "name": "index_logs",
        "description": "Tail index/embedding logs with optional category and level filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "level": {"type": "string"},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "index_metrics",
        "description": "Indexing and search timing counters.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "preload_models",
        "description": "Download/cache local embedding weights before offline use.",
        "parameters": {"type": "object", "properties": {}},
    },
]
