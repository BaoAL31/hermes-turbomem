from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Literal

import numpy as np
from turbovec import IdMapIndex

from hermes_turbomem.code_index import extract_chunks, file_content_hash, iter_source_files
from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.embedder import Embedder
from hermes_turbomem.project_id import ProjectInfo, resolve_project

EntryType = Literal["experience", "code"]


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
            """
        )
        self._conn.commit()
        try:
            self._conn.execute("ALTER TABLE entries ADD COLUMN tags TEXT")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    @staticmethod
    def _encode_tags(tags: list[str] | None) -> str | None:
        if not tags:
            return None
        normalized = []
        seen: set[str] = set()
        for tag in tags:
            t = str(tag).strip()
            if t and t not in seen:
                seen.add(t)
                normalized.append(t)
        return json.dumps(normalized) if normalized else None

    @staticmethod
    def _decode_tags(raw: str | None) -> list[str]:
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t) for t in parsed if str(t).strip()]
        except json.JSONDecodeError:
            pass
        return []

    @staticmethod
    def _tags_overlap(entry_tags: list[str], filter_tags: list[str] | None) -> bool:
        if not filter_tags:
            return True
        wanted = {str(t).strip() for t in filter_tags if str(t).strip()}
        if not wanted:
            return True
        stored = {str(t).strip() for t in entry_tags if str(t).strip()}
        return bool(wanted & stored)

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
        tags: list[str] | None = None,
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
            INSERT INTO entries (id, entry_type, project_id, text, category, tags, created_at)
            VALUES (?, 'experience', ?, ?, ?, ?, ?)
            """,
            (entry_id, project_id, text.strip(), category, self._encode_tags(tags), now),
        )
        self._conn.commit()
        self._insert_vector(entry_id, text.strip())
        return f"Stored experience #{entry_id}" + (f" for project {project_id}" if project_id else "")

    def list_experiences(
        self,
        limit: int = 20,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        clauses = ["entry_type = 'experience'"]
        params: list[object] = []
        if category:
            clauses.append("category = ?")
            params.append(category)
        rows = self._conn.execute(
            f"""
            SELECT id, category, text, tags, created_at FROM entries
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()
        if tags:
            rows = [
                r
                for r in rows
                if self._tags_overlap(self._decode_tags(r["tags"]), tags)
            ]
        rows = rows[:limit]
        if not rows:
            if tags:
                return "No experiences match the requested tags."
            return "No experiences stored."
        lines = [f"- #{r['id']} [{r['category']}] {r['text'][:200]}" for r in rows]
        return f"{len(rows)} experience(s):\n" + "\n".join(lines)

    def index_project(self, path: str, force: bool = False) -> str:
        info = resolve_project(path)
        root = info.root
        if not root.is_dir():
            return f"Project root not found: {root}"

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

        return (
            f"Indexed project {info.project_id} at {root}: "
            f"{added} code entries added, {skipped} files/chunks skipped, {removed} stale entries removed."
        )

    def _entry_row(self, entry_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()

    def recall(
        self,
        query: str,
        limit: int | None = None,
        project_id: str | None = None,
        types: list[str] | None = None,
        project_path: str | None = None,
        exclude_categories: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        self.maybe_auto_index(project_path)
        if not query.strip():
            return "Query is empty."

        k = limit or self.config.default_recall_limit
        if len(self._index) == 0:
            return "Persistent memory is empty. Use remember() or index_project() first."

        query_vec = self.embedder.encode([query.strip()])
        allowlist: np.ndarray | None = None
        if project_id or types or exclude_categories or tags:
            clauses = ["1=1"]
            params: list[object] = []
            if project_id:
                clauses.append("project_id = ?")
                params.append(project_id)
            if types:
                placeholders = ",".join("?" for _ in types)
                clauses.append(f"entry_type IN ({placeholders})")
                params.extend(types)
            if exclude_categories:
                placeholders = ",".join("?" for _ in exclude_categories)
                clauses.append(f"(category IS NULL OR category NOT IN ({placeholders}))")
                params.extend(exclude_categories)
            rows = self._conn.execute(
                f"SELECT id, tags FROM entries WHERE {' AND '.join(clauses)}",
                params,
            ).fetchall()
            if tags:
                rows = [
                    r
                    for r in rows
                    if self._tags_overlap(self._decode_tags(r["tags"]), tags)
                ]
            if not rows:
                return "No entries match the requested filters."
            allowlist = np.array([int(r["id"]) for r in rows], dtype=np.uint64)

        scores, ids = self._index.search(query_vec, k=k, allowlist=allowlist)
        if ids.size == 0:
            return "No matching memories found."

        lines: list[str] = []
        for score, entry_id in zip(scores[0].tolist(), ids[0].tolist(), strict=False):
            row = self._entry_row(int(entry_id))
            if row is None:
                continue
            lines.append(self._format_hit(row, float(score)))

        if not lines:
            return "No matching memories found."
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
            return (
                "No projects indexed yet.\n\n"
                "Use index_codebase(<path>) to add a project to the catalog."
            )
        lines = []
        for row in rows:
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(row["indexed_at"]))
                if row["indexed_at"]
                else "never"
            )
            lines.append(f"- {row['project_id']}\n  root: {row['root_path']}\n  indexed: {when}")
        return "\n".join(lines)

    def list_code_projects(self) -> str:
        return self.list_projects()

    def index_status(
        self,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        from hermes_turbomem.embedder import is_model_cached

        if project_path and not project_id:
            project_id = resolve_project(project_path).project_id

        if project_id:
            project_row = self._conn.execute(
                "SELECT project_id, root_path, indexed_at FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            if project_row is None:
                return (
                    f"Project '{project_id}' is not in the catalog.\n\n"
                    "Use index_codebase(<path>) to index it."
                )
            code_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM entries WHERE project_id = ? AND entry_type = 'code'",
                    (project_id,),
                ).fetchone()[0]
            )
            exp_count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM entries WHERE project_id = ? AND entry_type = 'experience'",
                    (project_id,),
                ).fetchone()[0]
            )
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(project_row["indexed_at"]))
                if project_row["indexed_at"]
                else "never"
            )
            cached = is_model_cached(self.config.embedding_model)
            lines = [
                f"Index status for {project_id}:",
                f"  Root: {project_row['root_path']}",
                f"  Last indexed: {when}",
                f"  Code entries: {code_count}",
                f"  Experiences: {exp_count}",
                f"  Embed model '{self.config.embedding_model}': {'cached' if cached else 'not cached'}",
            ]
            if code_count == 0:
                lines.append("")
                lines.append("No code entries yet. Re-run index_codebase(<path>) if indexing failed.")
            if not cached:
                lines.append("Embedding model not cached. Use preload_models() before offline use.")
            return "\n".join(lines)

        project_count = int(
            self._conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        )
        code_count = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM entries WHERE entry_type = 'code'"
            ).fetchone()[0]
        )
        exp_count = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM entries WHERE entry_type = 'experience'"
            ).fetchone()[0]
        )
        row = self._conn.execute("SELECT MAX(indexed_at) FROM projects").fetchone()
        last_indexed = row[0] if row[0] is not None else None
        cached = is_model_cached(self.config.embedding_model)

        lines = [
            "Index status:",
            f"  Data dir: {self.config.data_dir}",
            f"  Database: {project_count} project(s), {code_count} code entry/chunk(s), {exp_count} experience(s)",
            f"  Embed model '{self.config.embedding_model}': {'cached' if cached else 'not cached'}",
        ]
        if last_indexed is not None:
            lines.append(
                f"  Last indexed: {time.strftime('%Y-%m-%d %H:%M', time.localtime(last_indexed))}"
            )
        else:
            lines.append("  Last indexed: never")

        if project_count == 0:
            lines.append("")
            lines.append("No projects indexed yet. Use index_codebase(<path>) to index a project.")
        if not cached:
            lines.append("Embedding model not cached. Use preload_models() to download before offline use.")
        return "\n".join(lines)
