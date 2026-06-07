from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Literal

import numpy as np
from turbovec import IdMapIndex

from hermes_turbomem.bm25 import BM25Index, rrf_fuse
from hermes_turbomem.call_graph import extract_edges
from hermes_turbomem.code_index import extract_chunks, file_content_hash, iter_indexable_files
from hermes_turbomem.config import TurbomemConfig
from hermes_turbomem.embedder import Embedder
from hermes_turbomem.project_id import ProjectInfo, resolve_project

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
        self._bm25_index: BM25Index | None = None
        self._bm25_dirty = True

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
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS call_edges (
                project_id TEXT NOT NULL,
                path TEXT NOT NULL,
                caller_symbol TEXT NOT NULL,
                callee TEXT NOT NULL,
                caller_start_line INTEGER NOT NULL,
                caller_end_line INTEGER NOT NULL,
                callee_line INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_call_edges_project ON call_edges(project_id);
            CREATE INDEX IF NOT EXISTS idx_call_edges_caller ON call_edges(project_id, caller_symbol);
            CREATE INDEX IF NOT EXISTS idx_call_edges_callee ON call_edges(project_id, callee);
            """
        )
        self._conn.commit()

    def _no_hit_suffix(self) -> str:
        return NO_HIT_HINTS

    def _persist_index_metadata(self) -> None:
        for key, value in (
            ("embedding_model", self.config.embedding_model),
            ("bit_width", str(self.config.bit_width)),
        ):
            self._conn.execute(
                """
                INSERT INTO index_metadata (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
        self._conn.commit()

    def _check_embed_config(self) -> str | None:
        for key, current in (
            ("embedding_model", self.config.embedding_model),
            ("bit_width", str(self.config.bit_width)),
        ):
            row = self._conn.execute(
                "SELECT value FROM index_metadata WHERE key = ?", (key,)
            ).fetchone()
            if row and row["value"] != current:
                return (
                    f"Index metadata mismatch for {key}: stored '{row['value']}', "
                    f"config '{current}'. Re-run index_codebase with force on all projects."
                )
        return None

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

    def _build_bm25_index(self, project_id: str | None = None) -> BM25Index:
        if self._bm25_index is not None and not self._bm25_dirty:
            return self._bm25_index

        if project_id:
            rows = self._conn.execute(
                "SELECT id, text, symbol FROM entries WHERE entry_type = 'code' AND project_id = ?",
                (project_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, text, symbol FROM entries WHERE entry_type = 'code'",
            ).fetchall()

        bm25 = BM25Index()
        for row in rows:
            bm25.add_document(int(row["id"]), row["text"] or "", row["symbol"])
        self._bm25_index = bm25
        self._bm25_dirty = False
        return bm25

    def _hybrid_search(
        self,
        query_vec: np.ndarray,
        query_text: str,
        k: int,
        allowlist: np.ndarray | None = None,
        project_id: str | None = None,
    ) -> list[tuple[int, float, float, bool]]:
        semantic_k = k * 3
        scores, ids = self._index.search(query_vec, k=semantic_k, allowlist=allowlist)
        semantic_results: list[tuple[int, float]] = []
        if ids.size > 0:
            semantic_results = [
                (int(eid), float(sc)) for eid, sc in zip(ids[0].tolist(), scores[0].tolist(), strict=False)
            ]

        bm25 = self._build_bm25_index(project_id=project_id)
        bm25_results = bm25.search(query_text, top_k=semantic_k)
        fused = rrf_fuse(semantic_results, bm25_results)

        semantic_lookup = {eid: sc for eid, sc in semantic_results}
        threshold = self.config.confidence_threshold

        results: list[tuple[int, float, float, bool]] = []
        for eid, rrf_score in fused[:k]:
            sem_score = semantic_lookup.get(eid, 0.0)
            is_low = sem_score < threshold
            results.append((int(eid), float(sem_score), float(rrf_score), is_low))
        return results

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

        files = iter_indexable_files(root)
        self._bm25_dirty = True
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
                self._conn.execute(
                    "DELETE FROM call_edges WHERE project_id = ? AND path = ?",
                    (info.project_id, rel),
                )

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

            edges = extract_edges(file_path, raw)
            if edges:
                for edge in edges:
                    self._conn.execute(
                        """
                        INSERT INTO call_edges (
                            project_id, path, caller_symbol, callee,
                            caller_start_line, caller_end_line, callee_line
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            info.project_id,
                            rel,
                            edge.caller_symbol,
                            edge.callee,
                            edge.caller_start_line,
                            edge.caller_end_line,
                            edge.callee_line,
                        ),
                    )

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

        self._persist_index_metadata()
        return (
            f"Indexed project {info.project_id} at {root}: "
            f"{added} code entries added, {skipped} files/chunks skipped, {removed} stale entries removed."
        )

    def index_codebase(self, path: str, force: bool = False) -> str:
        return self.index_project(path, force)

    def _code_search(
        self,
        query: str,
        limit: int | None,
        project_id: str | None,
        project_path: str | None,
        *,
        peek: bool = False,
    ) -> str:
        if not query.strip():
            return "Query is empty."
        self.maybe_auto_index(project_path)
        if not self._conn.execute("SELECT 1 FROM projects LIMIT 1").fetchone():
            return "No projects indexed yet. Use index_codebase(path) first." + self._no_hit_suffix()
        err = self._check_embed_config()
        if err:
            return err
        if project_path and not project_id:
            project_id = resolve_project(project_path).project_id

        k = limit or self.config.default_recall_limit
        if len(self._index) == 0:
            empty = "No matching Code Entries." if peek else "No matching memories found."
            return empty + self._no_hit_suffix()

        query_vec = self.embedder.encode([query.strip()])
        clauses = ["entry_type = 'code'"]
        params: list[object] = []
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        rows = self._conn.execute(
            f"SELECT id FROM entries WHERE {' AND '.join(clauses)}",
            params,
        ).fetchall()
        if not rows:
            msg = "No matching Code Entries." if peek else "No entries match the requested filters."
            return msg + self._no_hit_suffix()
        allowlist = np.array([int(r["id"]) for r in rows], dtype=np.uint64)

        results = self._hybrid_search(
            query_vec, query.strip(), k, allowlist=allowlist, project_id=project_id
        )
        if not results:
            empty = "No matching Code Entries." if peek else "No matching memories found."
            return empty + self._no_hit_suffix()

        lines: list[str] = []
        for eid, sem_score, _rrf_score, is_low in results:
            row = self._entry_row(int(eid))
            if row is None:
                continue
            if peek:
                lines.append(self._format_peek_hit(row, sem_score, is_low=is_low))
            else:
                lines.append(self._format_hit(row, sem_score, is_low=is_low))

        if not lines:
            empty = "No matching Code Entries." if peek else "No matching memories found."
            return empty + self._no_hit_suffix()
        return "\n\n".join(lines)

    def code_recall(
        self,
        query: str,
        limit: int | None = None,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        return self._code_search(query, limit, project_id, project_path, peek=False)

    def code_peek(
        self,
        query: str,
        limit: int | None = None,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        return self._code_search(query, limit, project_id, project_path, peek=True)

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
            return "No matching memories found." + self._no_hit_suffix()

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
                return "No entries match the requested filters." + self._no_hit_suffix()
            allowlist = np.array([int(r["id"]) for r in rows], dtype=np.uint64)

        scores, ids = self._index.search(query_vec, k=k, allowlist=allowlist)
        if ids.size == 0:
            return "No matching memories found." + self._no_hit_suffix()

        lines: list[str] = []
        for score, entry_id in zip(scores[0].tolist(), ids[0].tolist(), strict=False):
            row = self._entry_row(int(entry_id))
            if row is None:
                continue
            lines.append(self._format_hit(row, float(score)))

        if not lines:
            return "No matching memories found." + self._no_hit_suffix()
        return "\n\n".join(lines)

    def _format_hit(self, row: sqlite3.Row, score: float, is_low: bool = False) -> str:
        low_flag = " [LOW-CONFIDENCE]" if is_low else ""
        entry_type = row["entry_type"]
        if entry_type == "code":
            proj = row["project_id"] or "unknown"
            sym = row["symbol"] or "(file)"
            loc = f"{row['path']}:{row['start_line']}-{row['end_line']}"
            preview = (row["text"] or "")[:400]
            return (
                f"[code | {proj} | score {score:.3f}{low_flag}]\n"
                f"{sym} @ {loc}\n"
                f"{preview}"
            )
        cat = row["category"] or "general"
        proj = f" | {row['project_id']}" if row["project_id"] else ""
        return (
            f"[experience | {cat}{proj} | score {score:.3f}{low_flag}]\n"
            f"{row['text']}"
        )

    def _format_peek_hit(self, row: sqlite3.Row, score: float, is_low: bool = False) -> str:
        low_flag = " [LOW-CONFIDENCE]" if is_low else ""
        entry_type = row["entry_type"]
        if entry_type == "code":
            proj = row["project_id"] or "unknown"
            sym = row["symbol"] or "(file)"
            loc = f"{row['path']}:{row['start_line']}-{row['end_line']}"
            return (
                f"[code | {proj} | score {score:.3f}{low_flag}]\n"
                f"{sym} @ {loc}"
            )
        cat = row["category"] or "general"
        proj = f" | {row['project_id']}" if row["project_id"] else ""
        return (
            f"[experience | {cat}{proj} | score {score:.3f}{low_flag}]\n"
            f"(peek)"
        )

    def code_call_graph(
        self,
        name: str,
        direction: str = "callers",
        project_id: str | None = None,
        project_path: str | None = None,
        symbol_id: str | None = None,
    ) -> str:
        from hermes_turbomem.call_graph import SUPPORTED_LANGUAGES
        from hermes_turbomem.code_index import TS_LANGUAGE_MAP

        _ = symbol_id  # reserved for future disambiguation
        if direction not in ("callers", "callees"):
            return f"Invalid direction '{direction}'. Use 'callers' or 'callees'."

        if project_path and not project_id:
            project_id = resolve_project(project_path).project_id

        sym_row = self._conn.execute(
            """
            SELECT path FROM entries
            WHERE entry_type = 'code'
                AND (? IS NULL OR project_id = ?)
                AND (symbol = ? OR path LIKE ?)
            LIMIT 1
            """,
            (project_id, project_id, name, f"%{name}%"),
        ).fetchone()
        if sym_row is not None:
            ext = Path(sym_row["path"]).suffix.lower()
            lang = TS_LANGUAGE_MAP.get(ext)
            if lang is None or lang not in SUPPORTED_LANGUAGES:
                supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
                return (
                    f"Call graph not supported for '{sym_row['path']}' "
                    f"(language: {lang or 'unknown'}). "
                    f"Supported languages: {supported}."
                )

        if direction == "callees":
            rows = self._conn.execute(
                """
                SELECT callee, path, callee_line, caller_symbol, caller_start_line, caller_end_line
                FROM call_edges
                WHERE caller_symbol = ? AND (? IS NULL OR project_id = ?)
                ORDER BY callee
                """,
                (name, project_id, project_id),
            ).fetchall()
            if not rows:
                return f"No callees found for '{name}'."
            lines = ["Callees:"]
            for r in rows:
                loc = f"{r['path']}:{r['callee_line']}" if r["callee_line"] else r["path"]
                lines.append(f"  {r['callee']} @ {loc}")
            return "\n".join(lines)

        rows = self._conn.execute(
            """
            SELECT caller_symbol, path, caller_start_line, caller_end_line
            FROM call_edges
            WHERE callee = ? AND (? IS NULL OR project_id = ?)
            ORDER BY caller_symbol
            """,
            (name, project_id, project_id),
        ).fetchall()
        if not rows:
            return f"No callers found for '{name}'."
        lines = ["Callers:"]
        for r in rows:
            loc = f"{r['path']}:{r['caller_start_line']}-{r['caller_end_line']}"
            lines.append(f"  {r['caller_symbol']} @ {loc}")
        return "\n".join(lines)

    def health_check(
        self,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        if project_path and not project_id:
            project_id = resolve_project(project_path).project_id

        if project_id:
            rows = self._conn.execute(
                "SELECT project_id, root_path FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT project_id, root_path FROM projects").fetchall()

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
                entry_ids = [
                    int(r["id"])
                    for r in self._conn.execute(
                        "SELECT id FROM entries WHERE project_id = ?", (pid,)
                    ).fetchall()
                ]
                for eid in entry_ids:
                    if self._index.contains(eid):
                        self._index.remove(eid)
                self._conn.execute("DELETE FROM entries WHERE project_id = ?", (pid,))
                self._conn.execute("DELETE FROM file_hashes WHERE project_id = ?", (pid,))
                self._conn.execute("DELETE FROM call_edges WHERE project_id = ?", (pid,))
                self._conn.execute("DELETE FROM projects WHERE project_id = ?", (pid,))
                total_projects += 1
                continue

            stale_rows = [
                s
                for s in self._conn.execute(
                    """
                    SELECT id, path FROM entries
                    WHERE entry_type = 'code' AND path IS NOT NULL AND project_id = ?
                    """,
                    (pid,),
                ).fetchall()
                if not (root / s["path"]).is_file()
            ]
            stale_entry_ids = [int(s["id"]) for s in stale_rows]
            stale_paths = {s["path"] for s in stale_rows}

            if stale_entry_ids:
                for sid in stale_entry_ids:
                    if self._index.contains(sid):
                        self._index.remove(sid)
                placeholders = ",".join("?" for _ in stale_entry_ids)
                self._conn.execute(
                    f"DELETE FROM entries WHERE id IN ({placeholders})",
                    stale_entry_ids,
                )
                total_entries += len(stale_entry_ids)

            if stale_paths:
                for stale_path in stale_paths:
                    cur = self._conn.execute(
                        "DELETE FROM call_edges WHERE project_id = ? AND path = ?",
                        (pid, stale_path),
                    )
                    total_edges += cur.rowcount

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

        self._conn.commit()
        self._persist_index()
        self._bm25_dirty = True

        parts = []
        if total_entries:
            parts.append(f"{total_entries} stale Code Entries")
        if total_hashes:
            parts.append(f"{total_hashes} orphaned file hashes")
        if total_edges:
            parts.append(f"{total_edges} stale call graph edges")
        if total_projects:
            parts.append(f"{total_projects} absent project(s) removed")

        if not parts:
            return "Index health check complete: no stale entries found."
        return "Index health check complete. Removed " + ", ".join(parts) + "."

    def index_health_check(
        self,
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> str:
        return self.health_check(project_id=project_id, project_path=project_path)

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
