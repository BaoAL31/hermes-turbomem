from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from turbovec import IdMapIndex

from hermes_turbomem.code_index import extract_chunks, file_content_hash, find_callees, iter_source_files
from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.project_id import ProjectInfo, resolve_project

if TYPE_CHECKING:
    from hermes_turbomem.embedder import Embedder

EntryType = Literal["experience", "code"]

NO_HIT_HINTS = (
    " Try `index_status` to check index readiness, "
    "`index_health_check` to clean stale entries, "
    "or re-run `index_codebase` to rebuild the index."
)


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
            CREATE TABLE IF NOT EXISTS call_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                caller_entry_id INTEGER,
                callee_entry_id INTEGER,
                caller_name TEXT NOT NULL,
                callee_name TEXT NOT NULL,
                caller_path TEXT,
                callee_path TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_call_graph_project
                ON call_graph(project_id);
            CREATE INDEX IF NOT EXISTS idx_call_graph_caller
                ON call_graph(caller_entry_id);
            CREATE INDEX IF NOT EXISTS idx_call_graph_callee
                ON call_graph(callee_entry_id);
            """
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

    def _insert_vector(self, entry_id: int, text: str) -> None:
        self._insert_vectors([entry_id], [text])
        self._persist_index()

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
        return self.index_project(project_path)

    def remember(
        self,
        text: str,
        category: str = "general",
        project_path: str | None = None,
    ) -> str:
        self.maybe_auto_index(project_path)
        project_id = None
        if project_path:
            project_id = resolve_project(project_path).project_id

        entry_id = self._next_id
        self._next_id += 1
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO entries (id, entry_type, project_id, text, category, created_at)
            VALUES (?, 'experience', ?, ?, ?, ?)
            """,
            (entry_id, project_id, text.strip(), category, now),
        )
        self._conn.commit()
        self._insert_vector(entry_id, text.strip())
        return f"Stored experience #{entry_id}" + (f" for project {project_id}" if project_id else "")

    def index_project(self, path: str, force: bool = False) -> str:
        info = resolve_project(path)
        root = info.root
        if not root.is_dir():
            return f"Project root not found: {root}"

        files = iter_source_files(root)
        added = 0
        skipped = 0
        removed = 0
        edges_stored = 0

        chunk_symbols: list[tuple[int, str | None, str, str]] = []

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
                chunk_symbols.append(
                    (entry_id, chunk.symbol, chunk.path, chunk.text)
                )
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

        if chunk_symbols:
            symbol_to_entry: dict[str, tuple[int, str]] = {}
            for eid, sym, cpath, _ctext in chunk_symbols:
                if sym:
                    symbol_to_entry[sym] = (eid, cpath)
            known_symbols = set(symbol_to_entry.keys())
            for eid, sym, cpath, ctext in chunk_symbols:
                if not sym:
                    continue
                caller_eid, _ = symbol_to_entry[sym]
                callee_names = find_callees(ctext, sym, known_symbols)
                for callee_name in callee_names:
                    if callee_name not in symbol_to_entry:
                        continue
                    callee_eid, callee_path = symbol_to_entry[callee_name]
                    self._conn.execute(
                        """
                        INSERT INTO call_graph
                            (project_id, caller_entry_id, callee_entry_id,
                             caller_name, callee_name, caller_path, callee_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (info.project_id, caller_eid, callee_eid,
                         sym, callee_name, cpath, callee_path),
                    )
                    edges_stored += 1

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

        msg = (
            f"Indexed project {info.project_id} at {root}: "
            f"{added} code entries added, {skipped} files/chunks skipped, {removed} stale entries removed."
        )
        if edges_stored:
            msg += f" {edges_stored} call graph edges stored."
        return msg

    def _entry_row(self, entry_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()

    def recall(
        self,
        query: str,
        limit: int | None = None,
        project_id: str | None = None,
        types: list[str] | None = None,
        project_path: str | None = None,
    ) -> str:
        self.maybe_auto_index(project_path)
        if not query.strip():
            return "Query is empty."

        k = limit or self.config.default_recall_limit
        if len(self._index) == 0:
            return f"No matching memories found.{NO_HIT_HINTS}"

        query_vec = self.embedder.encode([query.strip()])
        allowlist: np.ndarray | None = None
        if project_id or types:
            clauses = ["1=1"]
            params: list[object] = []
            if project_id:
                clauses.append("project_id = ?")
                params.append(project_id)
            if types:
                placeholders = ",".join("?" for _ in types)
                clauses.append(f"entry_type IN ({placeholders})")
                params.extend(types)
            rows = self._conn.execute(
                f"SELECT id FROM entries WHERE {' AND '.join(clauses)}",
                params,
            ).fetchall()
            if not rows:
                return f"No entries match the requested filters.{NO_HIT_HINTS}"
            allowlist = np.array([int(r["id"]) for r in rows], dtype=np.uint64)

        scores, ids = self._index.search(query_vec, k=k, allowlist=allowlist)
        if ids.size == 0:
            return f"No matching memories found.{NO_HIT_HINTS}"

        lines: list[str] = []
        for score, entry_id in zip(scores[0].tolist(), ids[0].tolist(), strict=False):
            row = self._entry_row(int(entry_id))
            if row is None:
                continue
            lines.append(self._format_hit(row, float(score)))

        if not lines:
            return f"No matching memories found.{NO_HIT_HINTS}"
        return "\n\n".join(lines)

    def _format_hit(self, row: sqlite3.Row, score: float) -> str:
        entry_type = row["entry_type"]
        if entry_type == "code":
            proj = row["project_id"] or "unknown"
            sym = row["symbol"] or "(file)"
            loc = f"{row['path']}:{row['start_line']}-{row['end_line']}"
            preview = (row["text"] or "")[:400]
            return (
                f"[code | {proj} | score {score:.3f}]\n"
                f"{sym} @ {loc}\n"
                f"{preview}"
            )
        cat = row["category"] or "general"
        proj = f" | {row['project_id']}" if row["project_id"] else ""
        return (
            f"[experience | {cat}{proj} | score {score:.3f}]\n"
            f"{row['text']}"
        )

    def list_projects(self) -> str:
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

    def code_peek(
        self,
        query: str,
        limit: int | None = None,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        """Metadata-only recall: returns path, symbol, line range—no source body."""
        self.maybe_auto_index(project_path)
        if not query.strip():
            return "Query is empty."

        k = limit or self.config.default_recall_limit
        if len(self._index) == 0:
            return f"No matching Code Entries.{NO_HIT_HINTS}"

        resolved_id = project_id
        if project_path and not resolved_id:
            resolved_id = resolve_project(project_path).project_id

        query_vec = self.embedder.encode([query.strip()])
        allowlist: np.ndarray | None = None

        clauses = ["entry_type = 'code'"]
        params: list[object] = []
        if resolved_id:
            clauses.append("project_id = ?")
            params.append(resolved_id)

        rows = self._conn.execute(
            f"SELECT id FROM entries WHERE {' AND '.join(clauses)}",
            params,
        ).fetchall()
        if not rows:
            return f"No matching Code Entries.{NO_HIT_HINTS}"
        allowlist = np.array([int(r["id"]) for r in rows], dtype=np.uint64)

        scores, ids = self._index.search(query_vec, k=k, allowlist=allowlist)
        if ids.size == 0:
            return f"No matching Code Entries.{NO_HIT_HINTS}"

        lines: list[str] = []
        for score, entry_id in zip(scores[0].tolist(), ids[0].tolist(), strict=False):
            row = self._entry_row(int(entry_id))
            if row is None:
                continue
            lines.append(self._format_peek(row, float(score)))

        if not lines:
            return f"No matching Code Entries.{NO_HIT_HINTS}"
        return "\n\n".join(lines)

    def _format_peek(self, row: sqlite3.Row, score: float) -> str:
        proj = row["project_id"] or "unknown"
        sym = row["symbol"] or "(file)"
        loc = f"{row['path']}:{row['start_line']}-{row['end_line']}"
        return (
            f"[code | {proj} | score {score:.3f}]\n"
            f"{sym} @ {loc}"
        )

    def health_check(
        self,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        """Remove stale Code Entries for files that no longer exist on disk.
        Cleans up orphaned vectors, file_hashes, and returns a report.
        """
        if project_path and not project_id:
            project_id = resolve_project(project_path).project_id

        if project_id:
            rows = self._conn.execute(
                "SELECT project_id, root_path FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT project_id, root_path FROM projects"
            ).fetchall()

        if not rows:
            return "No projects indexed; nothing to check."

        total_entries = 0
        total_hashes = 0
        total_projects = 0
        total_edges = 0

        for row in rows:
            pid = row["project_id"]
            root = Path(row["root_path"])

            if not root.is_dir():
                self._conn.execute("DELETE FROM entries WHERE project_id = ?", (pid,))
                self._conn.execute("DELETE FROM file_hashes WHERE project_id = ?", (pid,))
                self._conn.execute("DELETE FROM call_graph WHERE project_id = ?", (pid,))
                self._conn.execute("DELETE FROM projects WHERE project_id = ?", (pid,))
                total_projects += 1
                continue

            stale_entry_ids = [
                int(s["id"])
                for s in self._conn.execute(
                    """
                    SELECT id, path FROM entries
                    WHERE entry_type = 'code' AND path IS NOT NULL AND project_id = ?
                    """,
                    (pid,),
                ).fetchall()
                if not (root / s["path"]).is_file()
            ]

            if stale_entry_ids:
                for sid in stale_entry_ids:
                    if self._index.contains(sid):
                        self._index.remove(sid)
                    self._conn.execute(
                        "DELETE FROM call_graph WHERE caller_entry_id = ? OR callee_entry_id = ?",
                        (sid, sid),
                    )
                placeholders = ",".join("?" for _ in stale_entry_ids)
                self._conn.execute(
                    f"DELETE FROM entries WHERE id IN ({placeholders})",
                    stale_entry_ids,
                )
                total_entries += len(stale_entry_ids)

            stale_hashes = [
                h["path"]
                for h in self._conn.execute(
                    "SELECT path FROM file_hashes WHERE project_id = ?", (pid,)
                ).fetchall()
                if not (root / h["path"]).is_file()
            ]
            if stale_hashes:
                placeholders = ",".join("?" for _ in stale_hashes)
                self._conn.execute(
                    f"DELETE FROM file_hashes WHERE project_id = ? AND path IN ({placeholders})",
                    (pid, *stale_hashes),
                )
                total_hashes += len(stale_hashes)

            orphan_edges = self._conn.execute(
                """
                SELECT cg.id FROM call_graph cg
                LEFT JOIN entries e1 ON cg.caller_entry_id = e1.id
                LEFT JOIN entries e2 ON cg.callee_entry_id = e2.id
                WHERE cg.project_id = ? AND (e1.id IS NULL OR e2.id IS NULL)
                """,
                (pid,),
            ).fetchall()
            if orphan_edges:
                orphan_ids = [r["id"] for r in orphan_edges]
                placeholders = ",".join("?" for _ in orphan_ids)
                self._conn.execute(
                    f"DELETE FROM call_graph WHERE id IN ({placeholders})",
                    orphan_ids,
                )
                total_edges += len(orphan_ids)

        self._conn.commit()
        self._persist_index()

        parts = []
        if total_entries:
            parts.append(f"{total_entries} stale Code Entries")
        if total_hashes:
            parts.append(f"{total_hashes} orphaned file hashes")
        if total_edges:
            parts.append(f"{total_edges} orphaned call graph edges")
        if total_projects:
            parts.append(f"{total_projects} absent project(s) removed")

        if not parts:
            return "Index health check complete: no stale entries found."
        return f"Index health check complete. Removed " + ", ".join(parts) + "."

    def index_status(
        self,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        """Report index readiness, entry counts, and embedding model state."""
        if project_path and not project_id:
            project_id = resolve_project(project_path).project_id

        if project_id:
            rows = self._conn.execute(
                """
                SELECT p.project_id, p.root_path, p.indexed_at,
                       (SELECT COUNT(*) FROM entries e WHERE e.project_id = p.project_id) AS entry_count
                FROM projects p WHERE p.project_id = ?
                """,
                (project_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT p.project_id, p.root_path, p.indexed_at,
                       (SELECT COUNT(*) FROM entries e WHERE e.project_id = p.project_id) AS entry_count
                FROM projects p ORDER BY p.indexed_at DESC
                """
            ).fetchall()

        if not rows:
            return "No projects indexed. Use `index_codebase` to index a project."

        embed_dim = self.embedder.dimension if self._index.dim is not None else 0
        lines = [f"Embedding model: {self.config.embedding_model} (dim={embed_dim})"]
        lines.append(f"Turbovec index: {len(self._index)} vectors total")
        lines.append("")

        for r in rows:
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(r["indexed_at"]))
                if r["indexed_at"]
                else "never"
            )
            lines.append(
                f"Project: {r['project_id']}\n"
                f"  root: {r['root_path']}\n"
                f"  indexed: {when}\n"
                f"  entries: {r['entry_count']}"
            )

        return "\n".join(lines)
