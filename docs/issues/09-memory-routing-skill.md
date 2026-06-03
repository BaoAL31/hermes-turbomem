## Parent

#1

## What to build

Ship complement-mode agent guidance: a Hermes **memory routing** skill (markdown) documenting when to use `hindsight_*` vs `code_*` tools. Remove any remaining pre-PRD tools (`remember`, `recall_memory`, `index_project`, etc.) from MCP surface and README. Align package entrypoint and tool names with PRD v1 list.

## Acceptance criteria

- [ ] Skill documents: facts → Hindsight; code location → `code_*`; never duplicate facts in turbomem
- [ ] MCP tool list matches PRD v1 (no `remember`)
- [ ] README tool table matches implemented tools
- [ ] Skill included in repo (e.g. `docs/` or `skills/`) with install instructions for `~/.hermes/skills/`

## Blocked by

- #4
