param(
    # Phases à exécuter (défaut : les 3, dans l'ordre P6 -> P3 -> P4)
    [string[]]$Phases = @("P6", "P3", "P4")
)

# P3/P4/P6 — tests d'hygiène des 3 prédictions SYNTHESIS restantes (§6).
# Phase 1 = ctrl uniquement. Les ablations (P3/P4 phase 2) se lancent
# APRÈS agrégation, sur les seuls seeds very_good identifiés ici.
#
#   P6 : bonus_energy=160 (max_pop=60, vcost=0.05), 16k ticks, seeds 1-15.
#        Prédiction : chute abrupte du taux very_good (étouffement énergétique).
#        Pré-enregistré : confirmée si very_good durci <= 1/15 ; réfutée si >= 3/15.
#   P3 : vcost=0.03 (baseline params), 15k ticks, seeds 1-15 + very_good
#        phase D hors plage (24, 25, 31, 40, 42) pour enrichir la strate.
#        Prédiction : aucun very_good ne montrera Δcl ablation < -3 (phase 2).
#   P4 : max_pop=80, bonus=70, vcost {0.04, 0.05}, 15k ticks, seeds 1-15.
#        Prédiction : seuil 0.045 invariant -> effet causal à 0.05 seulement (phase 2).
#
# Idempotent : skip si le JSON final existe ; nettoie les dossiers partiels.

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$root = "C:\Users\Wilfred\Documents\IA Inst\AetherLife\aetherlife_pkg"
Set-Location $root
& "$root\.venv\Scripts\Activate.ps1"

$inv = [System.Globalization.CultureInfo]::InvariantCulture

function Invoke-Run {
    param(
        [string]$OutDir,
        [string]$LogPath,
        [int]$Seed,
        [int]$Ticks,
        [string]$VCost,        # déjà formaté culture invariante
        [string[]]$ExtraArgs
    )
    if (Test-Path "$OutDir\overnight_v8b1_seed$Seed.json") {
        Write-Host "[skip] $OutDir already complete"
        return
    }
    if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
    $t0 = Get-Date
    Write-Host "[run]  $OutDir  start=$t0"
    & python scripts/overnight_v8b1.py `
        --ticks $Ticks `
        --regime coordination_collective `
        --vocalize-cost $VCost `
        --seed $Seed `
        --out-dir $OutDir `
        --device cuda @ExtraArgs *>&1 | Tee-Object -FilePath $LogPath | Out-Null
    $dt = (Get-Date) - $t0
    if (Test-Path "$OutDir\overnight_v8b1_seed$Seed.json") {
        Write-Host ("[ok]   {0}  dur={1:N0}s" -f $OutDir, $dt.TotalSeconds)
    } else {
        Write-Host ("[FAIL] {0}  dur={1:N0}s  see {1}" -f $OutDir, $dt.TotalSeconds, $LogPath)
    }
}

$tStart = Get-Date
Write-Host "[P346] start=$tStart  phases=$($Phases -join ',')"

# ── P6 : étouffement énergétique bonus=160 ──────────────────────────────
if ($Phases -contains "P6") {
    $base = "$root\results\v8c3p6_b160"
    $logDir = "$base\_logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    foreach ($s in 1..15) {
        Invoke-Run -OutDir "$base\seed$s" -LogPath "$logDir\seed$s.log" `
            -Seed $s -Ticks 16000 -VCost ((0.05).ToString($inv)) `
            -ExtraArgs @("--bonus-energy-override", "160")
    }
}

# ── P3 : panel ctrl vcost=0.03 ──────────────────────────────────────────
if ($Phases -contains "P3") {
    $base = "$root\results\v8c3p3_v03"
    $logDir = "$base\_logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $seeds = @(1..15) + @(24, 25, 31, 40, 42)
    foreach ($s in $seeds) {
        Invoke-Run -OutDir "$base\seed$s" -LogPath "$logDir\seed$s.log" `
            -Seed $s -Ticks 15000 -VCost ((0.03).ToString($inv)) `
            -ExtraArgs @()
    }
}

# ── P4 : invariance du seuil à (max_pop=80, bonus=70) ──────────────────
if ($Phases -contains "P4") {
    foreach ($cost in @(0.04, 0.05)) {
        $costStr = $cost.ToString($inv)
        $costTag = "v{0:00}" -f [int]([math]::Round($cost * 100))
        $base = "$root\results\v8c3p4_$costTag"
        $logDir = "$base\_logs"
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        foreach ($s in 1..15) {
            Invoke-Run -OutDir "$base\seed$s" -LogPath "$logDir\seed$s.log" `
                -Seed $s -Ticks 15000 -VCost $costStr `
                -ExtraArgs @("--max-pop-override", "80", "--bonus-energy-override", "70")
        }
    }
}

$tEnd = Get-Date
Write-Host "[P346] done=$tEnd  total=$([int]($tEnd-$tStart).TotalMinutes)min"
