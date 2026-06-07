from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node
from tree_sitter_languages import get_parser

from hermes_turbomem.code_index import TS_LANGUAGE_MAP

SUPPORTED_LANGUAGES = {"python", "javascript", "typescript", "tsx", "go", "rust"}

_CALL_TYPES = {
    "python": "call",
    "javascript": "call_expression",
    "typescript": "call_expression",
    "tsx": "call_expression",
    "go": "call_expression",
    "rust": "call_expression",
}

_SYMBOL_TYPES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "method_definition", "arrow_function"},
    "tsx": {"function_declaration", "method_definition", "arrow_function"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item"},
}


@dataclass(frozen=True)
class CallEdge:
    caller_symbol: str
    callee: str
    caller_start_line: int
    caller_end_line: int
    callee_line: int


def _extract_callee_name(node: Node, source: str, lang: str) -> str | None:
    func = node.child_by_field_name("function")
    if func is None:
        return None
    if lang == "python":
        if func.type == "identifier":
            return source[func.start_byte : func.end_byte]
        if func.type == "attribute":
            attr = func.child_by_field_name("attribute")
            if attr is not None:
                return source[attr.start_byte : attr.end_byte]
        return None
    if lang in ("javascript", "typescript", "tsx"):
        if func.type == "identifier":
            return source[func.start_byte : func.end_byte]
        if func.type == "member_expression":
            prop = func.child_by_field_name("property")
            if prop is not None:
                return source[prop.start_byte : prop.end_byte]
        return None
    if lang == "go":
        if func.type == "identifier":
            return source[func.start_byte : func.end_byte]
        if func.type == "selector_expression":
            field = func.child_by_field_name("field")
            if field is not None:
                return source[field.start_byte : field.end_byte]
        return None
    if lang == "rust":
        if func.type == "identifier":
            return source[func.start_byte : func.end_byte]
        if func.type == "field_expression":
            field = func.child_by_field_name("field")
            if field is not None:
                return source[field.start_byte : field.end_byte]
        return None
    return None




def _extract_edges_python_regex(source: str) -> list[CallEdge] | None:
    import re as _re

    lines = source.splitlines()
    func_pat = _re.compile(r"^def\s+(\w+)\s*\(")
    call_pat = _re.compile(r"\b([A-Za-z_]\w*)\s*\(")
    edges: list[CallEdge] = []
    current: tuple[str, int, int] | None = None
    for i, line in enumerate(lines):
        m = func_pat.match(line)
        if m:
            current = (m.group(1), i + 1, i + 1)
            continue
        if current is None:
            continue
        caller, start, _ = current
        current = (caller, start, i + 1)
        for cm in call_pat.finditer(line):
            callee = cm.group(1)
            if callee in ("def", "if", "for", "while", "return", "print"):
                continue
            if callee == caller:
                continue
            edges.append(
                CallEdge(
                    caller_symbol=caller,
                    callee=callee,
                    caller_start_line=start,
                    caller_end_line=i + 1,
                    callee_line=i + 1,
                )
            )
    return edges if edges else None



def _extract_edges_regex(source: str, lang: str) -> list[CallEdge] | None:
    import re as _re

    if lang == "python":
        return _extract_edges_python_regex(source)

    patterns = {
        "javascript": _re.compile(r"^function\s+(\w+)\s*\("),
        "typescript": _re.compile(r"^function\s+(\w+)\s*\("),
        "tsx": _re.compile(r"^function\s+(\w+)\s*\("),
        "go": _re.compile(r"^func\s+(\w+)\s*\("),
        "rust": _re.compile(r"^fn\s+(\w+)\s*\("),
    }
    func_pat = patterns.get(lang)
    if func_pat is None:
        return None
    call_pat = _re.compile(r"\b([A-Za-z_]\w*)\s*\(")
    skip = {"def", "if", "for", "while", "return", "print", "function", "func", "fn"}
    lines = source.splitlines()
    edges: list[CallEdge] = []
    current: tuple[str, int, int] | None = None
    for i, line in enumerate(lines):
        m = func_pat.match(line.strip())
        if m:
            current = (m.group(1), i + 1, i + 1)
            continue
        if current is None:
            continue
        caller, start, _ = current
        current = (caller, start, i + 1)
        for cm in call_pat.finditer(line):
            callee = cm.group(1)
            if callee in skip or callee == caller:
                continue
            edges.append(
                CallEdge(
                    caller_symbol=caller,
                    callee=callee,
                    caller_start_line=start,
                    caller_end_line=i + 1,
                    callee_line=i + 1,
                )
            )
    return edges if edges else None

def extract_edges(path: Path, source: str) -> list[CallEdge] | None:
    lang = TS_LANGUAGE_MAP.get(path.suffix.lower())
    if lang is None or lang not in SUPPORTED_LANGUAGES:
        return None
    call_type = _CALL_TYPES.get(lang)
    if call_type is None:
        return None
    try:
        parser = get_parser(lang)
        tree = parser.parse(source.encode("utf-8"))
    except Exception:
        return _extract_edges_regex(source, lang)

    sym_types = _SYMBOL_TYPES.get(lang, set())
    edges: list[CallEdge] = []
    caller_stack: list[tuple[str, int, int]] = []

    def walk(node: Node) -> None:
        if node.type in sym_types:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = source[name_node.start_byte : name_node.end_byte]
                caller_stack.append(
                    (name, node.start_point[0] + 1, node.end_point[0] + 1)
                )
                for child in node.children:
                    walk(child)
                caller_stack.pop()
                return
        if node.type == call_type and caller_stack:
            callee = _extract_callee_name(node, source, lang)
            if callee is not None:
                caller_name, cs, ce = caller_stack[-1]
                edges.append(
                    CallEdge(
                        caller_symbol=caller_name,
                        callee=callee,
                        caller_start_line=cs,
                        caller_end_line=ce,
                        callee_line=node.start_point[0] + 1,
                    )
                )
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return edges if edges else None
