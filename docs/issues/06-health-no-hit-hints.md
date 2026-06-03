## Parent

#1

## What to build

Implement `index_health_check` to remove stale **Code Entries**, orphaned vectors, and dead **Call graph** edges after files are deleted or renamed. Update `code_recall` / `code_peek` no-hit responses to opencode-style guidance: suggest `index_status`, `index_health_check`, and re-run `index_codebase`—without auto-invoking those tools inside recall.

## Acceptance criteria

- [ ] After deleting an indexed file, `index_health_check` removes stale rows and reports counts
- [ ] Turbovec ids stay consistent with metadata after cleanup
- [ ] Zero-hit recall returns explicit message plus remediation hints (status/health/re-index)
- [ ] Recall does not silently trigger health repair

## Blocked by

- #3
