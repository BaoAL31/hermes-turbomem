from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectInfo:
    project_id: str
    root: Path
    git_remote: str | None


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def find_repo_root(start: Path) -> Path:
    path = start.resolve()
    if path.is_file():
        path = path.parent
    for parent in [path, *path.parents]:
        if (parent / ".git").is_dir():
            return parent
    return path


def normalize_remote(url: str) -> str:
    url = url.strip()
    for prefix in ("https://", "http://", "git@", "ssh://"):
        if url.startswith(prefix):
            url = url[len(prefix) :]
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower()


def resolve_project(path: str | Path) -> ProjectInfo:
    start = Path(path).expanduser()
    root = find_repo_root(start)
    remote = _run_git(["remote", "get-url", "origin"], root)
    canonical = str(root.resolve())

    if remote:
        project_id = f"git:{normalize_remote(remote)}"
    else:
        project_id = f"local:{canonical}"

    return ProjectInfo(project_id=project_id, root=root, git_remote=remote)
