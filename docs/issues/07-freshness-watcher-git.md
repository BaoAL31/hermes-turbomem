## Parent

#1

## What to build

Keep **Project index** fresh after first manual **Indexing**: file watcher re-indexes changed files (on by default once project registered; configurable off). Git-aware branch tracking and incremental updates on branch switch (opencode-style branch catalog). Optional `auto_index_on_first_use` config (default off) triggers first index when recall targets unknown project path.

## Acceptance criteria

- [ ] Saving a tracked file updates **Code Entries** without full rebuild
- [ ] Watcher can be disabled via config
- [ ] Branch switch updates branch membership / re-indexes changed files without full re-embed of unchanged content
- [ ] `auto_index_on_first_use: true` indexes on first recall against new path; default false preserves manual-first behavior
- [ ] Watcher errors logged, not swallowed

## Blocked by

- #3
