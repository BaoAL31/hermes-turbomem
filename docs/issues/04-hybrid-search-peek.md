## Parent

#1

## What to build

Add hybrid retrieval: Turbovec semantic top-k plus BM25 keyword search per **Project**, fused (RRF or weighted per PRD). Implement `code_peek` returning metadata-only hits (path, symbol, lines, score—no source bodies). Flag low-confidence results when scores fall below configured threshold.

## Acceptance criteria

- [ ] Exact symbol name query ranks via BM25/hybrid even when semantic alone is weak
- [ ] Conceptual query still returns relevant **Code Entries**
- [ ] `code_peek` omits code bodies but includes location metadata
- [ ] Low-confidence hits are explicitly labeled in output
- [ ] Tests cover fusion ordering with controlled fixtures or stubs

## Blocked by

- #4
