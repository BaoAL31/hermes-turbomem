from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from hermes_turbomem.diagnostics import get_logger, get_metrics

_log = get_logger()
_metrics = get_metrics()

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

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
}

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


def iter_source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


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
    except Exception as exc:
        _metrics.increment("parse_error")
        _log.log("parse", "WARN", f"tree-sitter parser for {lang} failed: {exc}")
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
    except (OSError, UnicodeDecodeError) as exc:
        _metrics.increment("parse_error")
        _log.log("parse", "WARN", f"Failed to read {path}: {exc}")
        return []

    rel_path = path.relative_to(root)
    ts_chunks = _chunks_tree_sitter(rel_path, source)
    if ts_chunks:
        return ts_chunks
    return _chunks_regex(rel_path, source)
