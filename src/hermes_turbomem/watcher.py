from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from hermes_turbomem.code_index import CODE_EXTENSIONS, SKIP_DIRS

logger = logging.getLogger(__name__)


class _CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, project_id: str, root: Path, store) -> None:
        self.project_id = project_id
        self.root = root
        self.store = store

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        self._reindex(event.src_path)

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        self._reindex(event.src_path)

    def _reindex(self, src_path: str) -> None:
        path = Path(src_path)
        try:
            rel = path.relative_to(self.root).as_posix()
        except ValueError:
            return
        if path.suffix.lower() not in CODE_EXTENSIONS:
            return
        if any(part in SKIP_DIRS for part in path.parts):
            return
        try:
            changed = self.store.reindex_file(self.project_id, self.root, rel)
            if changed:
                logger.info("Re-indexed %s/%s (watcher)", self.project_id, rel)
        except Exception:
            logger.exception("Error re-indexing %s/%s (watcher)", self.project_id, rel)


def _get_git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def _git_diff_files(root: Path, old_commit: str, new_commit: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", old_commit, new_commit],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return []


class ProjectWatcher:
    def __init__(
        self,
        project_id: str,
        root: Path,
        store,
        poll_interval: float = 2.0,
    ) -> None:
        self.project_id = project_id
        self.root = root
        self.store = store
        self.poll_interval = poll_interval
        self._observer: Observer | None = None
        self._branch_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_commit: str | None = None

    def start(self) -> None:
        if self._observer is not None:
            return

        handler = _CodeChangeHandler(self.project_id, self.root, self.store)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.root), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        logger.info("Started file watcher for %s at %s", self.project_id, self.root)

        self._last_commit = _get_git_commit(self.root)
        self._stop_event.clear()
        self._branch_thread = threading.Thread(
            target=self._branch_check_loop,
            daemon=True,
        )
        self._branch_thread.start()
        logger.info("Started branch tracker for %s", self.project_id)

    def stop(self) -> None:
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        if self._branch_thread is not None:
            self._branch_thread.join(timeout=5)
            self._branch_thread = None
        logger.info("Stopped watcher for %s", self.project_id)

    def _branch_check_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval)
            if self._stop_event.is_set():
                break
            try:
                current = _get_git_commit(self.root)
                if current is None:
                    continue
                if (
                    self._last_commit is not None
                    and current != self._last_commit
                ):
                    logger.info(
                        "HEAD changed for %s: %s..%s",
                        self.project_id,
                        self._last_commit[:12],
                        current[:12],
                    )
                    changed = _git_diff_files(
                        self.root, self._last_commit, current
                    )
                    reindexed = 0
                    for rel in changed:
                        path = self.root / rel
                        if not path.is_file():
                            continue
                        if path.suffix.lower() not in CODE_EXTENSIONS:
                            continue
                        if any(part in SKIP_DIRS for part in path.parts):
                            continue
                        try:
                            ok = self.store.reindex_file(
                                self.project_id, self.root, rel
                            )
                            if ok:
                                reindexed += 1
                        except Exception:
                            logger.exception(
                                "Error re-indexing %s/%s (branch switch)",
                                self.project_id,
                                rel,
                            )
                    logger.info(
                        "Branch switch update for %s: %d files re-indexed",
                        self.project_id,
                        reindexed,
                    )
                self._last_commit = current
            except Exception:
                logger.exception(
                    "Error in branch check loop for %s", self.project_id
                )
