param(
    [string]$Root = "."
)

$ErrorActionPreference = "Stop"

$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
$requiredFiles = @(
    "AGENTS.md",
    ".agents/skills",
    "docs/agent/workflow-playbook.md",
    "docs/agent/quality-gates.md",
    "docs/agent/prompt-library.md"
)

$errors = New-Object System.Collections.Generic.List[string]

foreach ($path in $requiredFiles) {
    $fullPath = Join-Path $resolvedRoot $path
    if (-not (Test-Path -LiteralPath $fullPath)) {
        $errors.Add("Missing required path: $path")
    }
}

$skillsRoot = Join-Path $resolvedRoot ".agents/skills"
if (Test-Path -LiteralPath $skillsRoot) {
    $skillFiles = Get-ChildItem -LiteralPath $skillsRoot -Directory | ForEach-Object {
        Join-Path $_.FullName "SKILL.md"
    }

    foreach ($skillFile in $skillFiles) {
        if (-not (Test-Path -LiteralPath $skillFile)) {
            $errors.Add("Missing SKILL.md: $skillFile")
            continue
        }

        $text = Get-Content -LiteralPath $skillFile -Raw
        if ($text -notmatch "(?ms)^---\s.*?\s---") {
            $errors.Add("Missing YAML frontmatter: $skillFile")
        }
        if ($text -notmatch "(?m)^name:\s*[a-z0-9]+[a-z0-9-]*$") {
            $errors.Add("Invalid or missing name field: $skillFile")
        }
        if ($text -notmatch "(?m)^description:\s*.+") {
            $errors.Add("Missing description field: $skillFile")
        }

        $directoryName = Split-Path (Split-Path $skillFile -Parent) -Leaf
        $nameLine = [regex]::Match($text, "(?m)^name:\s*(.+)$")
        if ($nameLine.Success -and $nameLine.Groups[1].Value.Trim() -ne $directoryName) {
            $errors.Add("Skill name does not match directory: $skillFile")
        }
    }
}

if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Error $_ }
    exit 1
}

Write-Output "Agent kit validation passed."
