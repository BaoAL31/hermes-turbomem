# Close superseded nightshift PRs #11-#19 after v2 provider integration lands on main.
# Requires: gh CLI authenticated for BaoAL31/hermes-turbomem
#
# Usage:
#   .\scripts\close-nightshift-prs.ps1
#   .\scripts\close-nightshift-prs.ps1 -DryRun
#   .\scripts\close-nightshift-prs.ps1 -IntegrationPrUrl "https://github.com/BaoAL31/hermes-turbomem/pull/33"

param(
    [switch]$DryRun,
    [string]$IntegrationPrUrl = "https://github.com/BaoAL31/hermes-turbomem/pull/33",
    [string]$IssueUrl = "https://github.com/BaoAL31/hermes-turbomem/issues/21"
)

$Comment = @"
Superseded by v2 Hermes memory provider integration.

- Parent epic: $IssueUrl
- Landed integration: $IntegrationPrUrl
- Triage notes: docs/nightshift-pr-triage.md

This PR targeted complement-era MCP slices; v2 is provider-only at ``plugins/memory/turbomem``.
"@

$prs = 11..19

foreach ($n in $prs) {
    $cmd = "gh pr close $n --comment `"$Comment`""
    Write-Host $cmd
    if (-not $DryRun) {
        Invoke-Expression $cmd
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Failed to close PR #$n (may already be closed)."
        }
    }
}

Write-Host "Done. Verify with: gh pr list --state open"
