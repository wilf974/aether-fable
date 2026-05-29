param(
    [int]$Start = 1,
    [int]$End   = 15,
    [double[]]$Costs = @(0.1, 0.2)
)

# P2 — test d'étouffement (signalisation coûteuse, prédiction SYNTHESIS §6).
# Hypothèse : à vcost >= 0.1, l'émergence linguistique est étouffée
#   (vocalize_total chute, cl_trend s'effondre, very_good < 14%, extinctions ↑).
# Ctrl only (pas d'ablation) — phase 1. Ablation ciblée en suivi si besoin.
# Baseline params conservés (max_pop=60, bonus=100) — SEUL vcost varie.

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$root = "C:\Users\Wilfred\Documents\IA Inst\AetherLife\aetherlife_pkg"
Set-Location $root
& "$root\.venv\Scripts\Activate.ps1"

$tStart = Get-Date
Write-Host "[P2] start=$tStart  costs=$($Costs -join ',')  seeds=$Start..$End"

$inv = [System.Globalization.CultureInfo]::InvariantCulture

foreach ($cost in $Costs) {
    # IMPORTANT locale FR : forcer la culture invariante sinon 0.1 -> "0,1"
    # (virgule) casse argparse float() côté Python.
    $costStr = $cost.ToString($inv)
    # tag répertoire : 0.1 -> v10, 0.2 -> v20
    $costTag = "v{0:00}" -f [int]([math]::Round($cost * 100))
    $base    = "$root\results\v8c3p2_$costTag"
    $logDir  = "$base\_logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null

    foreach ($s in $Start..$End) {
        $outDir = "$base\seed$s"
        $log    = "$logDir\seed$s.log"

        if (Test-Path "$outDir\overnight_v8b1_seed$s.json") {
            Write-Host "[skip] cost=$cost seed=$s already complete"
            continue
        }
        # repartir propre si dossier vide laissé par une interruption
        if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }

        $t0 = Get-Date
        Write-Host "[run]  cost=$cost seed=$s  start=$t0"
        & python scripts/overnight_v8b1.py `
            --ticks 16000 `
            --regime coordination_collective `
            --vocalize-cost $costStr `
            --seed $s `
            --out-dir $outDir `
            --device cuda *>&1 | Tee-Object -FilePath $log | Out-Null
        $dt = (Get-Date) - $t0
        if (Test-Path "$outDir\overnight_v8b1_seed$s.json") {
            Write-Host ("[ok]   cost={0} seed={1}  dur={2:N0}s" -f $cost, $s, $dt.TotalSeconds)
        } else {
            Write-Host ("[FAIL] cost={0} seed={1}  dur={2:N0}s  see $log" -f $cost, $s, $dt.TotalSeconds)
        }
    }
}

$tEnd = Get-Date
Write-Host "[P2] done=$tEnd  total=$([int]($tEnd-$tStart).TotalMinutes)min"
