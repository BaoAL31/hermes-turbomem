from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
from turbovec import IdMapIndex

from hermes_turbomem.code_index import (
    CodeChunk,
    extract_chunks,
    file_content_hash,
    iter_indexable_files,
)
from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.embedder import Embedder
from hermes_turbomem.project_id import ProjectInfo, resolve_project


class ProjectIndex:
    def __init__(self, project_info: ProjectInfo, embedder: Embedder, config: TurbomemConfig) -> None:
        self._project_info = project_info
        self._embedder = embedder
        self._config = config
        self._index_dir = project_info.root / ".turbomem"
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._index_dir / "project_index.db"
        self._tvim_path = self._index_dir / "index.tvim"
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._index = self._load_index()
        self._next_id = self._allocate_next_id()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS code_entries (
                id INTEGER PRIMARY KEY,
                project_id TEXT NOT NULL,
                path TEXT NOT NULL,
                symbol TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS file_hashes (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entries_path ON code_entries(path);
            CREATE INDEX IF NOT EXISTS idx_entries_project ON code_entries(project_id);
            """
        )
        self._conn.commit()

    def _load_index(self) -> IdMapIndex:
        if self._tvim_path.is_file():
            return IdMapIndex.load(str(self._tvim_path))
        return IdMapIndex(bit_width=self._config.bit_width)

    def _allocate_next_id(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM code_entries").fetchone()
        return max(int(row[0]) + 1, 1)

    def _persist_index(self) -> None:
        self._index.write(str(self._tvim_path))

    def _ensure_index(self) -> None:
        if self._index.dim is None:
            self._index = IdMapIndex(dim=self._embedder.dimension, bit_width=self._config.bit_width)

    def index_file(self, file_path: Path, root: Path) -> tuple[int, int]:
        rel = file_path.relative_to(root).as_posix()
        try:
            raw = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return 0, 1

        whole_hash = file_content_hash(raw)
        row = self._conn.execute(
            "SELECT content_hash FROM file_hashes WHERE path = ?",
            (rel,),
        ).fetchone()
        if row and row["content_hash"] == whole_hash:
            return 0, 1

        old = self._conn.execute(
            "SELECT id FROM code_entries WHERE path = ?",
            (rel,),
        ).fetchall()
        removed = 0
        for old_row in old:
            oid = int(old_row["id"])
            if self._index.contains(oid):
                self._index.remove(oid)
            self._conn.execute("DELETE FROM code_entries WHERE id = ?", (oid,))
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
                INSERT INTO code_entries (
                    id, project_id, path, symbol,
                    start_line, end_line, content, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    self._project_info.project_id,
                    chunk.path,
                    chunk.symbol,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.text,
                    chunk.content_hash,
                    now,
                ),
            )
            batch_ids.append(entry_id)
            batch_texts.append(embed_text)

        if batch_ids:
            self._ensure_index()
            vectors = self._embedder.encode(batch_texts)
            self._index.add_with_ids(
                vectors,
                np.array(batch_ids, dtype=np.uint64),
            )

        self._conn.execute(
            """
            INSERT INTO file_hashes (path, content_hash, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                content_hash = excluded.content_hash,
                updated_at = excluded.updated_at
            """,
            (rel, whole_hash, now),
        )
        self._conn.commit()
        self._persist_index()

        added = len(batch_ids)
        return added, removed

    def search(
        self,
        query_vec: np.ndarray,
        k: int,
    ) -> list[tuple[float, dict]]:
        if len(self._index) == 0:
            return []
        scores, ids = self._index.search(query_vec, k=k)
        results: list[tuple[float, dict]] = []
        for score, entry_id in zip(scores[0].tolist(), ids[0].tolist(), strict=False):
            row = self._conn.execute(
                "SELECT * FROM code_entries WHERE id = ?",
                (int(entry_id),),
            ).fetchone()
            if row is None:
                continue
            results.append((float(score), dict(row)))
        return results

    @property
    def size(self) -> int:
        return len(self._index)

    def entry_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM code_entries").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._conn.close()


class Catalog:
    def __init__(self, config: TurbomemConfig) -> None:
        self._config = config
        config.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = config.catalog_path
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                root_path TEXT NOT NULL,
                git_remote TEXT,
                indexed_at REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    def register(self, project_info: ProjectInfo) -> None:
        self._conn.execute(
            """
            INSERT INTO projects (project_id, root_path, git_remote, indexed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                root_path = excluded.root_path,
                git_remote = excluded.git_remote,
                indexed_at = excluded.indexed_at
            """,
            (project_info.project_id, str(project_info.root), project_info.git_remote, time.time()),
        )
        self._conn.commit()

    def list_projects(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT project_id, root_path, indexed_at FROM projects ORDER BY indexed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_project(self, project_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_all(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT project_id, root_path, indexed_at FROM projects ORDER BY indexed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


class MemoryStore:
    def __init__(self, config: TurbomemConfig, embedder: Embedder) -> None:
        self.config = config
        self.embedder = embedder
        self.catalog = Catalog(config)

    def index_codebase(self, path: str, force: bool = False) -> str:
        info = resolve_project(path)
        root = info.root
        if not root.is_dir():
            return f"Project root not found: {root}"

        proj_index = ProjectIndex(info, self.embedder, self.config)
        files = iter_indexable_files(root)
        added = 0
        skipped = 0
        removed = 0

        for file_path in files:
            if not force:
                a, r = proj_index.index_file(file_path, root)
                if a == 0:
                    skipped += 1
                else:
                    added += a
                    removed += r
            else:
                a, r = proj_index.index_file(file_path, root)
                added += a
                removed += r

        proj_index.close()
        self.catalog.register(info)

        return (
            f"Indexed project {info.project_id} at {root}: "
            f"{added} code entries added, {skipped} files skipped, {removed} stale entries removed."
        )

    def code_recall(
        self,
        query: str,
        limit: int | None = None,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        if not query.strip():
            return "Query is empty."

        k = limit or self.config.default_recall_limit
        query_vec = self.embedder.encode([query.strip()])

        projects = self.catalog.get_all()
        if not projects:
            return "No projects indexed yet. Use index_codebase(path) first."

        candidates: list[tuple[float, dict, str]] = []

        for proj in projects:
            pid = proj["project_id"]
            if project_id and pid != project_id:
                continue
            if project_path:
                resolved = resolve_project(project_path)
                if pid != resolved.project_id:
                    continue

            root_path = Path(proj["root_path"])
            info = ProjectInfo(project_id=pid, root=root_path, git_remote=proj.get("git_remote"))
            proj_index = ProjectIndex(info, self.embedder, self.config)
            try:
                results = proj_index.search(query_vec, k)
                proj_index.close()
            except KeyError:
                proj_index.close()
                continue
            for score, entry in results:
                candidates.append((score, entry, pid))

        candidates.sort(key=lambda x: -x[0])
        top = candidates[:k]

        if not top:
            return (
                "No matching code entries found. "
                "Use index_status to check index health, or re-run index_codebase."
            )

        lines: list[str] = []
        for score, entry, pid in top:
            lines.append(self._format_hit(entry, pid, score))
        return "\n\n".join(lines)

    def _format_hit(self, entry: dict, project_id: str, score: float) -> str:
        sym = entry.get("symbol") or "(file)"
        path = entry.get("path", "")
        start = entry.get("start_line", 0)
        end = entry.get("end_line", 0)
        preview = (entry.get("content") or "")[:400]
        return (
            f"[code | {project_id} | score {score:.3f}]\n"
            f"{sym} @ {path}:{start}-{end}\n"
            f"{preview}"
        )

    def list_code_projects(self) -> str:
        projects = self.catalog.list_projects()
        if not projects:
            return "No projects indexed yet."
        lines = []
        for proj in projects:
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(proj["indexed_at"]))
                if proj.get("indexed_at")
                else "never"
            )
            lines.append(f"- {proj['project_id']}\n  root: {proj['root_path']}\n  indexed: {when}")
        return "\n".join(lines)
