## Parent

#1

## What to build

Add observability tools: `index_logs` (filter by category and level, tail recent entries) and `index_metrics` (counters/timings for index, embed, search). Wire logging in index and search paths so failures are diagnosable without attaching a debugger.

## Acceptance criteria

- [ ] `index_logs` returns recent log lines; category and level filters work
- [ ] `index_metrics` exposes at least index duration, search duration, embed call counts
- [ ] Failed embed or parse writes to logs retrievable via `index_logs`
- [ ] Metrics reset or accumulate per process lifetime (document behavior)

## Blocked by

- #2
