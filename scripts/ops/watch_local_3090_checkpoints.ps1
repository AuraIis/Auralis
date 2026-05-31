param(
  [string]$SourceCheckpointDir = "\\BITBASTION\Auralis\NEWGPT\v2data\checkpoints\pretrain_1b_bilingual_de55_en45_ramp",
  [string]$SourceTokenizerDir = "\\BITBASTION\Auralis\NEWGPT\v2data\tokenizer",
  [string]$LocalRoot = "$env:USERPROFILE\Auralis_3090_Test",
  [string]$LocalRepo = "$env:USERPROFILE\Auralis_3090_Test\AuralisV2",
  [string]$WslPython = "/mnt/c/Users/_Michael_/Auralis_3090_Test/.venv27/bin/python",
  [int]$PollSeconds = 300,
  [int]$StableSeconds = 20,
  [int]$MinStep = 0,
  [int]$StepModulo = 500,
  [int]$BestStepModulo = 500,
  [switch]$IncludeBest,
  [switch]$Once,
  [switch]$SyncCode
)

$ErrorActionPreference = "Stop"

function Write-Log {
  param([string]$Message)
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Write-Host "[$stamp] $Message"
}

function To-WslPath {
  param([string]$Path)
  $full = [System.IO.Path]::GetFullPath($Path)
  $drive = $full.Substring(0, 1).ToLowerInvariant()
  $rest = $full.Substring(2).Replace("\", "/")
  return "/mnt/$drive$rest"
}

function Ensure-LocalLayout {
  New-Item -ItemType Directory -Force -Path "$LocalRoot\checkpoints\pretrain_1b_bilingual_de55_en45_ramp" | Out-Null
  New-Item -ItemType Directory -Force -Path "$LocalRepo\tokenizer" | Out-Null
  New-Item -ItemType Directory -Force -Path "$LocalRepo\eval\results\capability" | Out-Null
  New-Item -ItemType Directory -Force -Path "\\BITBASTION\Auralis\AuralisV2\reports\local_3090_checkpoint_tests" | Out-Null
  Copy-Item -Force "$SourceTokenizerDir\helix_v2_tokenizer.model" "$LocalRepo\tokenizer\helix_v2_tokenizer.model"
  Copy-Item -Force "$SourceTokenizerDir\helix_v2_tokenizer.vocab" "$LocalRepo\tokenizer\helix_v2_tokenizer.vocab"
}

function Sync-CodeIfRequested {
  if (-not $SyncCode) { return }
  $sourceRepo = "\\BITBASTION\Auralis\AuralisV2"
  foreach ($d in @("src", "scripts", "configs", "eval")) {
    Write-Log "sync $d -> local repo"
    robocopy "$sourceRepo\$d" "$LocalRepo\$d" /E /NFL /NDL /NJH /NJS /NP | Out-Null
  }
}

function Read-Step {
  param([string]$JsonPath)
  if (-not (Test-Path $JsonPath)) { return $null }
  try {
    $obj = Get-Content $JsonPath -Raw | ConvertFrom-Json
    return [int]$obj.state.step
  } catch {
    return $null
  }
}

function Is-StableFile {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return $false }
  $a = Get-Item $Path
  Start-Sleep -Seconds $StableSeconds
  if (-not (Test-Path $Path)) { return $false }
  $b = Get-Item $Path
  return ($a.Length -eq $b.Length -and $a.LastWriteTimeUtc -eq $b.LastWriteTimeUtc)
}

function Get-Candidates {
  $items = @()
  foreach ($pt in Get-ChildItem -Path $SourceCheckpointDir -Filter "*.pt" -File -ErrorAction SilentlyContinue) {
    if ($pt.Name.EndsWith(".tmp")) { continue }
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($pt.Name)
    if ($stem -ne "best" -and $stem -notmatch "^step_\d+$") { continue }
    $json = Join-Path $SourceCheckpointDir "$stem.json"
    $step = Read-Step $json
    if ($null -eq $step) {
      if ($stem -match "^step_(\d+)$") { $step = [int]$Matches[1] } else { continue }
    }
    if ($step -lt $MinStep) { continue }
    if ($stem -eq "best") {
      if (-not $IncludeBest) { continue }
      if ($BestStepModulo -gt 0 -and ($step % $BestStepModulo) -ne 0) { continue }
    }
    if ($stem -match "^step_\d+$" -and $StepModulo -gt 0 -and ($step % $StepModulo) -ne 0) {
      continue
    }
    $items += [pscustomobject]@{
      Name = $stem
      Step = $step
      Pt = $pt.FullName
      Json = $json
      LastWriteTimeUtc = $pt.LastWriteTimeUtc
      Length = $pt.Length
    }
  }
  return $items | Sort-Object Step, Name
}

function Run-Diagnosis {
  param([object]$Candidate)

  $tag = "local_3090_1b_ramp_$($Candidate.Name)_step$($Candidate.Step)"
  $donePath = Join-Path $LocalRepo "eval\results\capability\$tag.done"
  if (Test-Path $donePath) {
    Write-Log "skip already tested $tag"
    return
  }

  if (-not (Is-StableFile $Candidate.Pt)) {
    Write-Log "skip unstable checkpoint $($Candidate.Name) step $($Candidate.Step)"
    return
  }

  $localCkptDir = "$LocalRoot\checkpoints\pretrain_1b_bilingual_de55_en45_ramp"
  $localPt = Join-Path $localCkptDir "$($Candidate.Name)_step$($Candidate.Step).pt"
  $localJson = Join-Path $localCkptDir "$($Candidate.Name)_step$($Candidate.Step).json"

  Write-Log "copy $($Candidate.Name) step $($Candidate.Step) ($([math]::Round($Candidate.Length / 1GB, 2)) GiB)"
  Copy-Item -Force $Candidate.Pt $localPt
  if (Test-Path $Candidate.Json) {
    Copy-Item -Force $Candidate.Json $localJson
  }

  $repoW = To-WslPath $LocalRepo
  $ckptW = To-WslPath $localPt
  $cfgW = "$repoW/configs/model/helix_v2_1b.yaml"
  $tokW = "$repoW/tokenizer/helix_v2_tokenizer.model"
  $outDiagW = "$repoW/eval/results/capability/${tag}_diag.json"

  Write-Log "run generation/top-k/margin diagnosis for $tag"
  wsl.exe env AURALIS_USE_MAMBA_KERNEL=1 PYTHONPATH="$repoW/src" $WslPython "$repoW/scripts/eval/diagnose_checkpoint_generation.py" `
    --model-config $cfgW `
    --checkpoint $ckptW `
    --tokenizer $tokW `
    --output $outDiagW `
    --device cuda `
    --max-new-tokens 40 `
    --top-k 12
  if ($LASTEXITCODE -ne 0) { throw "diagnose_checkpoint_generation failed for $tag" }

  Write-Log "run capability probes for $tag"
  wsl.exe env AURALIS_USE_MAMBA_KERNEL=1 PYTHONPATH="$repoW/src" $WslPython "$repoW/scripts/eval/run_capability_probes.py" `
    --model-config $cfgW `
    --checkpoint $ckptW `
    --tokenizer $tokW `
    --results-dir "$repoW/eval/results/capability" `
    --tag $tag `
    --device cuda `
    --max-new-tokens 32
  if ($LASTEXITCODE -ne 0) { throw "run_capability_probes failed for $tag" }

  $reportDir = "\\BITBASTION\Auralis\AuralisV2\reports\local_3090_checkpoint_tests"
  foreach ($suffix in @(".json", ".md", "_diag.json", "_diag.md")) {
    $src = Join-Path "$LocalRepo\eval\results\capability" "$tag$suffix"
    if (Test-Path $src) {
      Copy-Item -Force $src (Join-Path $reportDir "$tag$suffix")
    }
  }

  "tested $(Get-Date -Format o)" | Set-Content -Encoding UTF8 $donePath
  Write-Log "done $tag"
}

Ensure-LocalLayout
Sync-CodeIfRequested

Write-Log "watching $SourceCheckpointDir poll=${PollSeconds}s once=$Once min_step=$MinStep step_mod=$StepModulo include_best=$IncludeBest best_mod=$BestStepModulo"
do {
  try {
    $candidates = @(Get-Candidates)
    if ($candidates.Count -eq 0) {
      Write-Log "no checkpoint candidates yet"
    } else {
      foreach ($cand in $candidates) {
        Run-Diagnosis $cand
      }
    }
  } catch {
    Write-Log "ERROR: $($_.Exception.Message)"
  }

  if ($Once) { break }
  Start-Sleep -Seconds $PollSeconds
} while ($true)
