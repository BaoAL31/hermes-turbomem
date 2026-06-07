# Close nightshift PRs #11–#19 (HITL)

Run **after** v2 integration slices merge to `main`.

## PowerShell script

```powershell
.\scripts\close-nightshift-prs.ps1 -DryRun
.\scripts\close-nightshift-prs.ps1 -IntegrationPrUrl "https://github.com/BaoAL31/hermes-turbomem/pull/33"
```

## Manual `gh` commands

Replace `INTEGRATION_PR` with the merged provider integration PR URL.

```bash
COMMENT="Superseded by v2 Hermes memory provider integration. See https://github.com/BaoAL31/hermes-turbomem/issues/21 and INTEGRATION_PR and docs/nightshift-pr-triage.md"

for n in 11 12 13 14 15 16 17 18 19; do
  gh pr close "$n" --comment "$COMMENT"
done

gh pr list --state open
```

## Acceptance

- Each PR #11–#19 closed with superseded comment
- No open PRs still target complement-mode MCP-only integration
