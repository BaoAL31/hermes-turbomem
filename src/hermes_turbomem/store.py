from __future__ import annotations

import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from turbovec import IdMapIndex

from hermes_turbomem.code_index import extract_chunks, file_content_hash, iter_source_files
from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.embedder import Embedder
from hermes_turbomem.project_id import ProjectInfo, resolve_project

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, config: TurbomemConfig, embedder: Embedder) -> None:
        self.config = config
        self.embedder = embedder
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.config.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._index = self._load_index()
        self._next_id = self._allocate_next_id()
        self._metrics: dict[str, float] = defaultdict(float)

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY,
                entry_type TEXT NOT NULL,
                project_id TEXT,
                text TEXT NOT NULL,
                category TEXT,
                path TEXT,
                symbol TEXT,
                start_line INTEGER,
                end_line INTEGER,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                root_path TEXT NOT NULL,
                git_remote TEXT,
                indexed_at REAL
            );
            CREATE TABLE IF NOT EXISTS file_hashes (
                project_id TEXT NOT NULL,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                PRIMARY KEY (project_id, path)
            );
            CREATE TABLE IF NOT EXISTS call_edges (
                id INTEGER PRIMARY KEY,
                project_id TEXT NOT NULL,
                caller_id INTEGER NOT NULL,
                callee_id INTEGER NOT NULL,
                kind TEXT,
                FOREIGN KEY (caller_id) REFERENCES entries(id),
                FOREIGN KEY (callee_id) REFERENCES entries(id)
            );
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    def _log(self, category: str, level: str, message: str) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT INTO logs (category, level, message, created_at) VALUES (?, ?, ?, ?)",
            (category, level, message, now),
        )
        self._conn.commit()

    def _load_index(self) -> IdMapIndex:
        if self.config.index_path.is_file():
            return IdMapIndex.load(str(self.config.index_path))
        return IdMapIndex(bit_width=self.config.bit_width)

    def _allocate_next_id(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM entries").fetchone()
        max_sqlite = int(row[0])
        return max(max_sqlite + 1, 1)

    def _persist_index(self) -> None:
        self._index.write(str(self.config.index_path))

    def _ensure_index(self) -> None:
        if self._index.dim is None:
            self._index = IdMapIndex(dim=self.embedder.dimension, bit_width=self.config.bit_width)

    def _insert_vectors(self, entry_ids: list[int], texts: list[str]) -> None:
        if not texts:
            return
        self._ensure_index()
        vectors = self.embedder.encode(texts)
        self._index.add_with_ids(
            vectors,
            np.array(entry_ids, dtype=np.uint64),
        )

    def is_project_indexed(self, project_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM projects WHERE project_id = ? AND indexed_at IS NOT NULL",
            (project_id,),
        ).fetchone()
        return row is not None

    def maybe_auto_index(self, project_path: str | None) -> str | None:
        if not project_path or not self.config.auto_index_on_first_use:
            return None
        info = resolve_project(project_path)
        if self.is_project_indexed(info.project_id):
            return None
        return self.index_codebase(project_path)

    def index_codebase(self, path: str, force: bool = False) -> str:
        info = resolve_project(path)
        root = info.root
        if not root.is_dir():
            return f"Project root not found: {root}"

        t0 = time.time()
        files = iter_source_files(root)
        added = 0
        skipped = 0
        removed = 0

        for file_path in files:
            rel = file_path.relative_to(root).as_posix()
            try:
                raw = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                skipped += 1
                continue

            whole_hash = file_content_hash(raw)
            if not force:
                row = self._conn.execute(
                    "SELECT content_hash FROM file_hashes WHERE project_id = ? AND path = ?",
                    (info.project_id, rel),
                ).fetchone()
                if row and row["content_hash"] == whole_hash:
                    skipped += 1
                    continue

            if not force:
                old = self._conn.execute(
                    """
                    SELECT id FROM entries
                    WHERE project_id = ? AND entry_type = 'code' AND path = ?
                    """,
                    (info.project_id, rel),
                ).fetchall()
                for old_row in old:
                    oid = int(old_row["id"])
                    if self._index.contains(oid):
                        self._index.remove(oid)
                    self._conn.execute("DELETE FROM entries WHERE id = ?", (oid,))
                    removed += 1

            chunks = extract_chunks(file_path, root)
            batch_ids: list[int] = []
            batch_texts: list[str] = []
            now = time.time()
            for chunk in chunks:
                entry_id = self._next_id
                self._next_id += 1
                embed_text = f"{chunk.symbol}\n{chunk.text}" if chunk.symbol else chunk.text
                self._conn.execute(
                    """
                    INSERT INTO entries (
                        id, entry_type, project_id, text, path, symbol,
                        start_line, end_line, created_at
                    ) VALUES (?, 'code', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry_id,
                        info.project_id,
                        chunk.text,
                        chunk.path,
                        chunk.symbol,
                        chunk.start_line,
                        chunk.end_line,
                        now,
                    ),
                )
                batch_ids.append(entry_id)
                batch_texts.append(embed_text)
                added += 1
            if batch_ids:
                self._insert_vectors(batch_ids, batch_texts)

            self._conn.execute(
                """
                INSERT INTO file_hashes (project_id, path, content_hash)
                VALUES (?, ?, ?)
                ON CONFLICT(project_id, path) DO UPDATE SET content_hash = excluded.content_hash
                """,
                (info.project_id, rel, whole_hash),
            )

        self._conn.execute(
            """
            INSERT INTO projects (project_id, root_path, git_remote, indexed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                root_path = excluded.root_path,
                git_remote = excluded.git_remote,
                indexed_at = excluded.indexed_at
            """,
            (info.project_id, str(root), info.git_remote, time.time()),
        )
        self._conn.commit()
        self._persist_index()

        elapsed = time.time() - t0
        self._metrics["index_time_sec"] += elapsed
        self._metrics["index_runs"] += 1
        self._log("index", "info", f"Indexed {info.project_id}: {added} added, {skipped} skipped, {removed} removed in {elapsed:.2f}s")

        return (
            f"Indexed project {info.project_id} at {root}: "
            f"{added} code entries added, {skipped} files/chunks skipped, {removed} stale entries removed "
            f"({elapsed:.2f}s)."
        )

    def _entry_row(self, entry_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()

    def code_recall(
        self,
        query: str,
        limit: int | None = None,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        t0 = time.time()
        self.maybe_auto_index(project_path)
        if not query.strip():
            return "Query is empty."

        k = limit or self.config.default_recall_limit
        if len(self._index) == 0:
            return (
                "Index is empty. Use index_codebase(path) to index a project first. "
                "Run index_status() for diagnostics."
            )

        query_vec = self.embedder.encode([query.strip()])
        self._metrics["search_count"] += 1

        allowlist: np.ndarray | None = None
        if project_id:
            rows = self._conn.execute(
                "SELECT id FROM entries WHERE project_id = ? AND entry_type = 'code'",
                (project_id,),
            ).fetchall()
            if not rows:
                return "No code entries match the requested project filter."
            allowlist = np.array([int(r["id"]) for r in rows], dtype=np.uint64)

        scores, ids = self._index.search(query_vec, k=k, allowlist=allowlist)
        if ids.size == 0:
            return (
                "No matching code entries found. "
                "Try index_status() to check index readiness, or broaden your query."
            )

        lines: list[str] = []
        for score, entry_id in zip(scores[0].tolist(), ids[0].tolist(), strict=False):
            row = self._entry_row(int(entry_id))
            if row is None:
                continue
            lines.append(self._format_hit(row, float(score)))

        if not lines:
            return "No matching code entries found."

        elapsed = time.time() - t0
        self._metrics["search_time_sec"] += elapsed
        return "\n\n".join(lines)

    def code_peek(
        self,
        query: str,
        limit: int | None = None,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        t0 = time.time()
        self.maybe_auto_index(project_path)
        if not query.strip():
            return "Query is empty."

        k = limit or self.config.default_recall_limit
        if len(self._index) == 0:
            return "Index is empty. Use index_codebase(path) to index a project first."

        query_vec = self.embedder.encode([query.strip()])

        allowlist: np.ndarray | None = None
        if project_id:
            rows = self._conn.execute(
                "SELECT id FROM entries WHERE project_id = ? AND entry_type = 'code'",
                (project_id,),
            ).fetchall()
            if not rows:
                return "No code entries match the requested project filter."
            allowlist = np.array([int(r["id"]) for r in rows], dtype=np.uint64)

        scores, ids = self._index.search(query_vec, k=k, allowlist=allowlist)
        if ids.size == 0:
            return "No matching code entries found."

        lines: list[str] = []
        for score, entry_id in zip(scores[0].tolist(), ids[0].tolist(), strict=False):
            row = self._entry_row(int(entry_id))
            if row is None:
                continue
            proj = row["project_id"] or "unknown"
            sym = row["symbol"] or "(file)"
            loc = f"{row['path']}:{row['start_line']}-{row['end_line']}"
            lines.append(f"[code | {proj} | score {float(score):.3f}]\n{sym} @ {loc}")

        elapsed = time.time() - t0
        self._metrics["peek_time_sec"] += elapsed
        if not lines:
            return "No matching code entries found."
        return "\n\n".join(lines)

    def _format_hit(self, row: sqlite3.Row, score: float) -> str:
        proj = row["project_id"] or "unknown"
        sym = row["symbol"] or "(file)"
        loc = f"{row['path']}:{row['start_line']}-{row['end_line']}"
        preview = (row["text"] or "")[:400]
        confidence = ""
        if score < 0.4:
            confidence = " [low confidence]"
        return (
            f"[code | {proj} | score {score:.3f}{confidence}]\n"
            f"{sym} @ {loc}\n"
            f"{preview}"
        )

    def list_code_projects(self) -> str:
        rows = self._conn.execute(
            "SELECT project_id, root_path, indexed_at FROM projects ORDER BY indexed_at DESC"
        ).fetchall()
        if not rows:
            return "No projects indexed yet."
        lines = []
        for row in rows:
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(row["indexed_at"]))
                if row["indexed_at"]
                else "never"
            )
            lines.append(f"- {row['project_id']}\n  root: {row['root_path']}\n  indexed: {when}")
        return "\n".join(lines)

    def code_call_graph(
        self,
        name: str,
        direction: str = "callers",
        project_id: str | None = None,
        project_path: str | None = None,
        symbol_id: int | None = None,
    ) -> str:
        self.maybe_auto_index(project_path)
        if project_path and not project_id:
            project_id = resolve_project(project_path).project_id

        if not project_id:
            return "A project_id or project_path is required for call graph queries."

        if symbol_id is not None:
            entry = self._entry_row(symbol_id)
            if entry is None:
                return f"Symbol id {symbol_id} not found."
            symbol_name = entry["symbol"] or f"id:{symbol_id}"
            target_id = symbol_id
        else:
            rows = self._conn.execute(
                "SELECT id FROM entries WHERE project_id = ? AND entry_type = 'code' AND symbol = ?",
                (project_id, name),
            ).fetchall()
            if not rows:
                return f"Symbol '{name}' not found in project {project_id}. Try code_recall first."
            target_id = int(rows[0]["id"])
            symbol_name = name

        if direction == "callers":
            edges = self._conn.execute(
                """
                SELECT e.id, e.symbol, e.path, e.start_line, e.end_line
                FROM call_edges ce
                JOIN entries e ON e.id = ce.caller_id
                WHERE ce.callee_id = ? AND ce.project_id = ?
                """,
                (target_id, project_id),
            ).fetchall()
        elif direction == "callees":
            edges = self._conn.execute(
                """
                SELECT e.id, e.symbol, e.path, e.start_line, e.end_line
                FROM call_edges ce
                JOIN entries e ON e.id = ce.callee_id
                WHERE ce.caller_id = ? AND ce.project_id = ?
                """,
                (target_id, project_id),
            ).fetchall()
        else:
            return f"Invalid direction '{direction}'. Use 'callers' or 'callees'."

        if not edges:
            return (
                f"No {direction} found for '{symbol_name}' in {project_id}. "
                "Call graph extraction may not be available for this language."
            )

        lines = [f"{direction.title()} of '{symbol_name}' in {project_id}:"]
        for edge in edges:
            sym = edge["symbol"] or f"id:{edge['id']}"
            loc = f"{edge['path']}:{edge['start_line']}-{edge['end_line']}"
            lines.append(f"  {sym} @ {loc}")
        return "\n".join(lines)

    def index_status(self, project_id: str | None = None, project_path: str | None = None) -> str:
        if project_path and not project_id:
            project_id = resolve_project(project_path).project_id

        if project_id:
            row = self._conn.execute(
                "SELECT * FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
            if not row:
                return f"Project '{project_id}' not found in catalog."
            chunk_count = self._conn.execute(
                "SELECT COUNT(*) FROM entries WHERE project_id = ? AND entry_type = 'code'",
                (project_id,),
            ).fetchone()[0]
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(row["indexed_at"]))
                if row["indexed_at"]
                else "never"
            )
            idx_entries = len(self._index)
            model_cached = "yes" if hasattr(self.embedder._model, "_model") else "unknown"
            return (
                f"Project: {project_id}\n"
                f"  root: {row['root_path']}\n"
                f"  indexed: {when}\n"
                f"  code entries: {chunk_count}\n"
                f"  vector index entries: {idx_entries}\n"
                f"  embed model: {self.config.embedding_model}\n"
                f"  model cached: {model_cached}"
            )

        projects = self._conn.execute(
            "SELECT project_id, root_path, indexed_at FROM projects ORDER BY indexed_at DESC"
        ).fetchall()
        if not projects:
            total_entries = self._conn.execute(
                "SELECT COUNT(*) FROM entries WHERE entry_type = 'code'"
            ).fetchone()[0]
            idx_entries = len(self._index)
            return (
                "No projects registered.\n"
                f"  total code entries: {total_entries}\n"
                f"  vector index entries: {idx_entries}\n"
                f"  embed model: {self.config.embedding_model}\n"
                f"  Use index_codebase(path) to index a project."
            )

        lines = ["Projects in catalog:"]
        for p in projects:
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(p["indexed_at"]))
                if p["indexed_at"]
                else "never"
            )
            count = self._conn.execute(
                "SELECT COUNT(*) FROM entries WHERE project_id = ? AND entry_type = 'code'",
                (p["project_id"],),
            ).fetchone()[0]
            lines.append(f"  {p['project_id']}: {count} entries (indexed {when})")
        lines.append(f"Embed model: {self.config.embedding_model}")
        return "\n".join(lines)

    def index_health_check(self, project_id: str | None = None, project_path: str | None = None) -> str:
        if project_path and not project_id:
            project_id = resolve_project(project_path).project_id

        removed_orphans = 0
        removed_stale = 0

        if project_id:
            rows = self._conn.execute(
                "SELECT id, path, project_id FROM entries WHERE entry_type = 'code' AND project_id = ?",
                (project_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, path, project_id FROM entries WHERE entry_type = 'code'"
            ).fetchall()

        for row in rows:
            eid = int(row["id"])
            if not self._index.contains(eid):
                self._conn.execute("DELETE FROM entries WHERE id = ?", (eid,))
                removed_orphans += 1

        stale_project_ids = self._conn.execute(
            "SELECT project_id FROM projects WHERE indexed_at IS NULL"
        ).fetchall()
        for p in stale_project_ids:
            pid = p["project_id"]
            self._conn.execute("DELETE FROM entries WHERE project_id = ?", (pid,))
            self._conn.execute("DELETE FROM file_hashes WHERE project_id = ?", (pid,))
            self._conn.execute("DELETE FROM projects WHERE project_id = ?", (pid,))
            removed_stale += 1

        self._conn.commit()

        result = f"Health check complete: removed {removed_orphans} orphan entries, {removed_stale} stale projects."
        self._log("health", "info", result)
        return result

    def index_logs(
        self,
        category: str | None = None,
        level: str | None = None,
        limit: int | None = 50,
    ) -> str:
        clauses = ["1=1"]
        params: list[object] = []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if level:
            clauses.append("level = ?")
            params.append(level)

        rows = self._conn.execute(
            f"SELECT * FROM logs WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?",
            [*params, limit or 50],
        ).fetchall()
        if not rows:
            return "No log entries match the requested filters."

        lines = []
        for row in reversed(rows):
            when = time.strftime("%H:%M:%S", time.localtime(row["created_at"]))
            lines.append(f"[{when}] [{row['level']}] ({row['category']}) {row['message']}")
        return "\n".join(lines)

    def index_metrics(self) -> str:
        total_entries = self._conn.execute(
            "SELECT COUNT(*) FROM entries WHERE entry_type = 'code'"
        ).fetchone()[0]
        total_projects = self._conn.execute(
            "SELECT COUNT(*) FROM projects"
        ).fetchone()[0]
        idx_entries = len(self._index)
        return (
            f"Index metrics:\n"
            f"  total code entries: {total_entries}\n"
            f"  vector index entries: {idx_entries}\n"
            f"  total projects: {total_projects}\n"
            f"Session counters:\n"
            f"  index runs: {self._metrics.get('index_runs', 0)}\n"
            f"  index time (sec): {self._metrics.get('index_time_sec', 0):.2f}\n"
            f"  searches: {self._metrics.get('search_count', 0)}\n"
            f"  search time (sec): {self._metrics.get('search_time_sec', 0):.2f}\n"
            f"  peeks: {self._metrics.get('peek_time_sec', 0):.2f}"
        )

    def preload_models(self) -> str:
        self._log("preload", "info", "Starting model preload...")
        dim = self.embedder.dimension
        self._ensure_index()
        msg = f"Embedding model '{self.config.embedding_model}' loaded (dim={dim}). Vector index ready."
        self._log("preload", "info", msg)
        return msg
