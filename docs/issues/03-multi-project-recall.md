## Parent

#1

## What to build

Extend **Code recall** to search the full **Project catalog** by default: query all registered projects in one `code_recall` call. Support optional `project_id` and `project_path` filters to narrow scope. Verify with two indexed fixture projects that recall from cwd outside both roots still finds symbols in the correct **Project**.

## Acceptance criteria

- [ ] Two projects indexed; `code_recall` without filter returns hits from both when relevant
- [ ] `project_id` filter limits results to one **Project**
- [ ] `project_path` filter resolves to correct **Project** and limits results
- [ ] Results label which **Project** each **Code Entry** belongs to

## Blocked by

- #3
