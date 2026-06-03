from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pathspec

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
}

DENY_PATTERNS = [
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
    "bin",
    "obj",
    ".next",
    ".nuxt",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.bin",
    "*.class",
    "*.o",
    "*.obj",
    "*.min.js",
    "*.min.css",
    "*.map",
    ".env",
    ".env.*",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Gemfile.lock",
    "poetry.lock",
    ".turbomem",
]

MAX_FILE_SIZE = 512_000
MAX_CHUNKS_PER_FILE = 200

TS_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".cs": "c_sharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
}


@dataclass(frozen=True)
class CodeChunk:
    path: str
    symbol: str | None
    start_line: int
    end_line: int
    text: str
    content_hash: str


def file_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _find_git_root(root: Path) -> Path | None:
    for parent in [root, *root.parents]:
        if (parent / ".git").is_dir():
            return parent
    git_file = root / ".git"
    if git_file.is_file():
        return root
    return None


def _load_gitignore_spec(root: Path) -> pathspec.PathSpec:
    patterns = []
    gitignore_path = root / ".gitignore"
    if gitignore_path.is_file():
        try:
            text = gitignore_path.read_text(encoding="utf-8")
            patterns.extend(text.splitlines())
        except (OSError, UnicodeDecodeError):
            pass
    turbomemignore_path = root / ".turbomemignore"
    if turbomemignore_path.is_file():
        try:
            text = turbomemignore_path.read_text(encoding="utf-8")
            patterns.extend(text.splitlines())
        except (OSError, UnicodeDecodeError):
            pass
    return pathspec.GitIgnoreSpec.from_lines(patterns)


def _is_denied(rel_path: str) -> bool:
    for pattern in DENY_PATTERNS:
        if pattern.startswith("*"):
            if rel_path.endswith(pattern[1:]):
                return True
        elif pattern.endswith("/"):
            if pattern[:-1] in rel_path.split("/"):
                return True
        elif pattern in rel_path.split("/"):
            return True
        else:
            from fnmatch import fnmatch

            if fnmatch(rel_path, pattern):
                return True
    return False


def iter_indexable_files(root: Path) -> list[Path]:
    root = root.resolve()
    files: list[Path] = []

    git_root = _find_git_root(root)

    if git_root:
        tracked = _run_git(["ls-files"], git_root)
        if tracked is not None:
            raw_paths = tracked.splitlines()
            for raw in raw_paths:
                full = (git_root / raw).resolve()
                if not full.is_file():
                    continue
                if full.suffix.lower() not in CODE_EXTENSIONS:
                    continue
                try:
                    if full.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                if _is_denied(raw):
                    continue
                files.append(full)
            files.sort(key=lambda p: p.relative_to(root).as_posix())
            return files

    spec = _load_gitignore_spec(root) if root == _find_git_root(root) else pathspec.GitIgnoreSpec.from_lines([])
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if _is_denied(rel):
            continue
        if spec.match_file(rel):
            continue
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue
        files.append(path)
    return files


def _chunks_tree_sitter(path: Path, source: str) -> list[CodeChunk] | None:
    try:
        from tree_sitter import Node
        from tree_sitter_languages import get_parser
    except ImportError:
        return None

    lang = TS_LANGUAGE_MAP.get(path.suffix.lower())
    if not lang:
        return None

    try:
        parser = get_parser(lang)
    except Exception:
        return None

    tree = parser.parse(source.encode("utf-8"))
    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    symbol_types = {
        "function_definition",
        "function_declaration",
        "method_definition",
        "class_definition",
        "class_declaration",
        "impl_item",
        "interface_declaration",
        "struct_item",
        "enum_item",
    }

    def walk(node: Node) -> None:
        if node.type in symbol_types:
            start_row = node.start_point[0]
            end_row = node.end_point[0]
            snippet = "\n".join(lines[start_row : end_row + 1])
            name_node = node.child_by_field_name("name")
            symbol = None
            if name_node is not None:
                symbol = source[name_node.start_byte : name_node.end_byte]
            rel = path.as_posix()
            chunks.append(
                CodeChunk(
                    path=rel,
                    symbol=symbol,
                    start_line=start_row + 1,
                    end_line=end_row + 1,
                    text=snippet[:12000],
                    content_hash=file_content_hash(snippet),
                )
            )
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return chunks if chunks else None


def _chunks_regex(path: Path, source: str) -> list[CodeChunk]:
    lines = source.splitlines()
    pattern = re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:def|class|func|fn|interface|struct|enum|type)\s+(\w+)",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(source))
    if not matches:
        rel = path.as_posix()
        return [
            CodeChunk(
                path=rel,
                symbol=None,
                start_line=1,
                end_line=max(1, len(lines)),
                text=source[:12000],
                content_hash=file_content_hash(source),
            )
        ]

    chunks: list[CodeChunk] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        snippet = source[start:end]
        start_line = source[:start].count("\n") + 1
        end_line = start_line + snippet.count("\n")
        rel = path.as_posix()
        chunks.append(
            CodeChunk(
                path=rel,
                symbol=match.group(1),
                start_line=start_line,
                end_line=end_line,
                text=snippet[:12000],
                content_hash=file_content_hash(snippet),
            )
        )
    return chunks


def extract_chunks(path: Path, root: Path) -> list[CodeChunk]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    rel_path = path.relative_to(root)
    ts_chunks = _chunks_tree_sitter(rel_path, source)
    if ts_chunks:
        return ts_chunks[:MAX_CHUNKS_PER_FILE]
    return _chunks_regex(rel_path, source)[:MAX_CHUNKS_PER_FILE]
