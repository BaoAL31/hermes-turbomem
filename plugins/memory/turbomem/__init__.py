"""turbomem — Hermes MemoryProvider (local hybrid recall)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider

from hermes_turbomem.config import TurbomemConfig, load_config
from hermes_turbomem.embedder import Embedder
from hermes_turbomem.project_id import resolve_project
from hermes_turbomem.store import MemoryStore

from .tools import ALL_PROVIDER_SCHEMAS

logger = logging.getLogger(__name__)


def _load_provider_config(hermes_home: str) -> TurbomemConfig:
    cfg = load_config()
    data_dir = Path(hermes_home) / "turbomem"
    return TurbomemConfig(
        data_dir=data_dir,
        auto_index_on_first_use=cfg.auto_index_on_first_use,
        embedding_model=cfg.embedding_model,
        bit_width=cfg.bit_width,
        default_recall_limit=cfg.default_recall_limit,
    )


def _resolve_project_id(project_path: str | None, project_id: str | None) -> str | None:
    if project_id:
        return project_id
    if project_path:
        return resolve_project(project_path).project_id
    return None


class TurbomemMemoryProvider(MemoryProvider):
    """Local memory provider — full PRD tool surface (grill Q4-A)."""

    def __init__(self) -> None:
        self._store: MemoryStore | None = None
        self._embedder: Embedder | None = None
        self._prefetch_cache = ""
        self._prefetch_lock = threading.Lock()
        self._sync_thread: threading.Thread | None = None
        self._compress_thread: threading.Thread | None = None
        self._hermes_home = ""
        self._auto_retain_turns = True
        self._recall_max_chars = 4000
        self._retain_max_chars_per_side = 1200
        self._retain_every_n_turns = 1
        self._turn_counter = 0

    @property
    def name(self) -> str:
        return "turbomem"

    def is_available(self) -> bool:
        try:
            import hermes_turbomem  # noqa: F401
            import turbovec  # noqa: F401
            return True
        except ImportError:
            return False

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "auto_retain_turns",
                "description": "Retain raw turn summary after each turn (background)",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "recall_max_chars",
                "description": "Max characters injected via prefetch",
                "default": "4000",
            },
            {
                "key": "retain_max_chars_per_side",
                "description": "Max characters per user/assistant side in auto-retain (sync_turn)",
                "default": "1200",
            },
            {
                "key": "retain_every_n_turns",
                "description": "Auto-retain every N completed turns (1 = each turn; N>1 skips intermediate turns)",
                "default": "1",
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        import yaml

        config_dir = Path(hermes_home) / "turbomem"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "provider.yaml").write_text(
            yaml.safe_dump(values, sort_keys=False),
            encoding="utf-8",
        )

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._hermes_home = str(kwargs.get("hermes_home", ""))
        if kwargs.get("agent_context", "primary") != "primary":
            return

        turbomem_config = _load_provider_config(self._hermes_home)
        self._embedder = Embedder(turbomem_config.embedding_model)
        self._store = MemoryStore(turbomem_config, self._embedder)

        provider_yaml = Path(self._hermes_home) / "turbomem" / "provider.yaml"
        if provider_yaml.is_file():
            import yaml

            raw = yaml.safe_load(provider_yaml.read_text(encoding="utf-8")) or {}
            self._auto_retain_turns = str(raw.get("auto_retain_turns", "true")).lower() == "true"
            self._recall_max_chars = int(raw.get("recall_max_chars", 4000))
            self._retain_max_chars_per_side = int(raw.get("retain_max_chars_per_side", 1200))
            self._retain_every_n_turns = max(1, int(raw.get("retain_every_n_turns", 1)))

        self._session_id = session_id

    def system_prompt_block(self) -> str:
        return (
            "# turbomem (local)\n"
            "Full tool surface: memory_store (experiences), code_* (source), index_*/list_code_projects, "
            "preload_models. Prefetch injects experience recall before each turn."
        )

    def _prefetch_experiences(self, query: str, limit: int = 5) -> str:
        if not self._store or not query.strip():
            return ""
        text = self._store.recall(
            query=query,
            limit=limit,
            types=["experience"],
            exclude_categories=["conversation", "compression"],
        )
        if not text or "empty" in text.lower() or "no matching" in text.lower():
            return ""
        if len(text) > self._recall_max_chars:
            text = text[: self._recall_max_chars] + "\n…(truncated)"
        return "## turbomem memory\n" + text

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._prefetch_lock:
            if self._prefetch_cache:
                return self._prefetch_cache
        try:
            return self._prefetch_experiences(query)
        except Exception as exc:
            logger.debug("turbomem prefetch failed: %s", exc)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._store or not query.strip():
            return

        def _run() -> None:
            try:
                result = self._prefetch_experiences(query)
                with self._prefetch_lock:
                    self._prefetch_cache = result
            except Exception as exc:
                logger.debug("turbomem queue_prefetch failed: %s", exc)

        threading.Thread(target=_run, daemon=True).start()

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._auto_retain_turns or not self._store:
            return
        self._turn_counter += 1
        if self._turn_counter % self._retain_every_n_turns != 0:
            return
        cap = self._retain_max_chars_per_side
        summary = f"User: {user_content[:cap]}\nAssistant: {assistant_content[:cap]}"
        if not summary.strip():
            return

        sid = (session_id or getattr(self, "_session_id", "")).strip()
        auto_tags = ["conversation"] + ([f"session:{sid}"] if sid else [])

        def _retain() -> None:
            try:
                self._store.remember(text=summary, category="conversation", tags=auto_tags)
            except Exception as exc:
                logger.warning("turbomem sync_turn failed: %s", exc)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
        self._sync_thread = threading.Thread(target=_retain, daemon=True)
        self._sync_thread.start()
        with self._prefetch_lock:
            self._prefetch_cache = ""

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(ALL_PROVIDER_SCHEMAS)

    def _dispatch_store(self, method: str, **kwargs: Any) -> str:
        if not self._store:
            return json.dumps({"error": "turbomem not initialized"})
        fn = getattr(self._store, method, None)
        if fn is None:
            return (
                f"{method} is not implemented yet. "
                "Complete nightshift integration (backlog B-01–B-04) or use a merged store."
            )
        return fn(**kwargs)

    def _handle_memory_store(self, args: dict[str, Any]) -> str:
        action = args["action"]
        if action == "retain":
            if not args.get("text"):
                return json.dumps({"error": "retain requires text"})
            return self._store.remember(
                text=args["text"],
                category=args.get("category", "general"),
                project_path=args.get("project_path"),
                tags=args.get("tags"),
            )
        if action == "recall":
            if not args.get("query"):
                return json.dumps({"error": "recall requires query"})
            return self._store.recall(
                query=args["query"],
                limit=args.get("limit"),
                project_path=args.get("project_path"),
                types=["experience"],
                tags=args.get("tags"),
            )
        if action == "list":
            return self._store.list_experiences(
                limit=int(args.get("limit", 20)),
                category=args.get("category"),
                tags=args.get("tags"),
            )
        return json.dumps({"error": f"unknown action: {action}"})

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if tool_name == "memory_store":
            return self._handle_memory_store(args)

        pid = _resolve_project_id(args.get("project_path"), args.get("project_id"))

        if tool_name == "index_codebase":
            if hasattr(self._store, "index_codebase"):
                return self._store.index_codebase(path=args["path"], force=args.get("force", False))
            return self._store.index_project(path=args["path"], force=args.get("force", False))

        if tool_name == "code_recall":
            if hasattr(self._store, "code_recall"):
                return self._store.code_recall(
                    query=args["query"],
                    limit=args.get("limit"),
                    project_id=pid,
                    project_path=args.get("project_path"),
                )
            return self._store.recall(
                query=args["query"],
                limit=args.get("limit"),
                project_id=pid,
                project_path=args.get("project_path"),
                types=["code"],
            )

        if tool_name == "code_peek":
            return self._dispatch_store(
                "code_peek",
                query=args["query"],
                limit=args.get("limit"),
                project_id=pid,
                project_path=args.get("project_path"),
            )

        if tool_name == "code_call_graph":
            return self._dispatch_store(
                "code_call_graph",
                name=args["name"],
                direction=args.get("direction", "callers"),
                project_id=pid,
                project_path=args.get("project_path"),
                symbol_id=args.get("symbol_id"),
            )

        if tool_name == "list_code_projects":
            return self._dispatch_store("list_code_projects") or self._store.list_projects()

        if tool_name == "index_status":
            return self._dispatch_store("index_status", project_id=pid, project_path=args.get("project_path"))

        if tool_name == "index_health_check":
            return self._dispatch_store("index_health_check", project_id=pid, project_path=args.get("project_path"))

        if tool_name == "index_logs":
            return self._dispatch_store(
                "index_logs",
                category=args.get("category"),
                level=args.get("level"),
                limit=args.get("limit", 50),
            )

        if tool_name == "index_metrics":
            return self._dispatch_store("index_metrics")

        if tool_name == "preload_models":
            if self._embedder is None:
                return "Embedder not initialized."
            self._embedder.preload()
            cached = self._embedder.is_cached()
            return (
                f"Embedding model ready (dim={self._embedder.dimension}, "
                f"model='{self._embedder.model_name}', cached={cached})."
            )

        return json.dumps({"error": f"unknown tool: {tool_name}"})

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Retain a truncated summary of messages about to leave the context window."""
        if not self._store or not messages:
            return ""

        parts: list[str] = []
        for msg in messages[-10:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            if role not in ("user", "assistant"):
                continue
            label = "User" if role == "user" else "Assistant"
            parts.append(f"{label}: {content[:500]}")

        if not parts:
            return ""

        summary = "\n".join(parts)
        sid = getattr(self, "_session_id", "").strip()
        tags = ["compression"] + ([f"session:{sid}"] if sid else [])

        def _retain() -> None:
            try:
                self._store.remember(text=summary, category="compression", tags=tags)
            except Exception as exc:
                logger.warning("turbomem on_pre_compress failed: %s", exc)

        self._compress_thread = threading.Thread(
            target=_retain, daemon=True, name="turbomem-pre-compress"
        )
        self._compress_thread.start()
        return ""

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Wait for background retains, then store a final session summary."""
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        if self._compress_thread and self._compress_thread.is_alive():
            self._compress_thread.join(timeout=5.0)

        if not self._auto_retain_turns or not self._store or not messages:
            return

        parts: list[str] = []
        per_msg_cap = 500
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            if role not in ("user", "assistant"):
                continue
            label = "User" if role == "user" else "Assistant"
            parts.append(f"{label}: {content[:per_msg_cap]}")

        if not parts:
            return

        summary = "\n".join(parts)
        max_summary = max(self._retain_max_chars_per_side * 4, 4000)
        if len(summary) > max_summary:
            summary = summary[:max_summary] + "\n…(truncated)"

        sid = getattr(self, "_session_id", "").strip()
        tags = ["session-end", "conversation"] + ([f"session:{sid}"] if sid else [])

        try:
            self._store.remember(text=summary, category="conversation", tags=tags)
        except Exception as exc:
            logger.warning("turbomem on_session_end failed: %s", exc)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.remember(text=content, category=category)
            except Exception as exc:
                logger.debug("turbomem on_memory_write failed: %s", exc)

    def shutdown(self) -> None:
        self._store = None
        self._embedder = None
        with self._prefetch_lock:
            self._prefetch_cache = ""


def register(ctx: Any) -> None:
    ctx.register_memory_provider(TurbomemMemoryProvider())
