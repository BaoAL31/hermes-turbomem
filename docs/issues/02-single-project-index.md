## Parent

#1

## What to build

Deliver the first complete indexing path for one **Project**: resolve **Project** identity (git remote or local path), apply **Index scope** (gitignore, deny list, caps, `git ls-files`), tree-sitter semantic **Code Entries**, local embeddings, Turbovec per-project storage under `{root}/.turbomem/`, SQLite metadata, register in **Project catalog**, incremental content-hash skip on re-run. Expose `index_codebase` and vector-only `code_recall` for that project. Realign/remove pre-PRD scaffold patterns (global-only store, `remember`, old tool names).

## Acceptance criteria

- [ ] `index_codebase(path)` indexes a small fixture repo into `{root}/.turbomem/`
- [ ] Project appears in catalog via `list_code_projects`
- [ ] `code_recall(query)` returns ranked **Code Entries** with path, symbol, line range for indexed project
- [ ] Re-index with unchanged files skips re-embed (incremental hash)
- [ ] Index scope excludes gitignored / denied paths in fixture tests
- [ ] No `remember` tool exposed (complement mode)

## Blocked by

- #2
