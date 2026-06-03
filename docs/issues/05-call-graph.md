## Parent

#1

## What to build

During **Indexing**, extract **Call graph** edges (caller → callee) for Python, TypeScript, JavaScript, Go, and Rust. Expose `code_call_graph(name, direction, ...)` returning callers or callees for a symbol. Other languages remain searchable via `code_recall` but return a clear unsupported message from call-graph tool.

## Acceptance criteria

- [ ] Fixture with known call chain indexes edges in SQLite (or equivalent)
- [ ] `code_call_graph` with `direction=callers` returns expected caller symbols
- [ ] `code_call_graph` with `direction=callees` returns expected callee symbols (symbol id if needed per PRD)
- [ ] Unsupported language returns explicit message, not empty error
- [ ] Call graph survives incremental re-index when file unchanged

## Blocked by

- #3
