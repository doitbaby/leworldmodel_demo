param(
    [Parameter(Mandatory = $true)]
    [string]$TargetRepo,

    [string]$SourceRepo = ".",

    [switch]$IncludeAdapters
)

$ErrorActionPreference = "Stop"

$source = (Resolve-Path -LiteralPath $SourceRepo).Path
if (-not (Test-Path -LiteralPath $TargetRepo)) {
    New-Item -ItemType Directory -Force -Path $TargetRepo | Out-Null
}
$target = (Resolve-Path -LiteralPath $TargetRepo).Path

function Copy-Path {
    param(
        [string]$RelativePath
    )

    $src = Join-Path $source $RelativePath
    $dst = Join-Path $target $RelativePath

    if (-not (Test-Path -LiteralPath $src)) {
        return
    }

    $parent = Split-Path $dst -Parent
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    if ((Get-Item -LiteralPath $src).PSIsContainer) {
        Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
    } else {
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}

Copy-Path "AGENTS.md"
Copy-Path ".agents/skills"
Copy-Path "docs/agent"

if ($IncludeAdapters) {
    Copy-Path ".github/copilot-instructions.md"
    Copy-Path ".github/instructions"
    Copy-Path ".cursor/rules"
    Copy-Path "CLAUDE.md"
}

Write-Output "Agent kit installed to $target"
