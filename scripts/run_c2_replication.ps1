# C2 — REPLICATION CONFIRMATOIRE N=30 de l'effet diversite d'affinite -> survie.
#
# Pre-enregistrement : docs/preregistrations/2026-06-10-c2-replication-N30-prereg.md
# (critere VERROUILLE avant collecte des seeds 11-30).
#
# Protocole IDENTIQUE a C2 original (run_c2_affinity.ps1) :
#   regime coordination_collective, 16k ticks, --n-initial-affinities {1,2,4}, cuda.
# Reutilise results/c2_aff{k}/seed{s}/ -> les seeds 1-10 deja collectes sont SKIP
# (idempotent). Seuls les 20 nouveaux seeds (11-30) x 3 k = 60 runs s'executent.
#
# Durci contre les pertes GPU transitoires (cf. crash 2026-06-10) :
#   Wait-ForGpu (re-test CUDA toutes les 30s) + retry x3 par run.
#
# Agregation : python scripts/aggregate_c2.py results/c2_aff1/seed* \
#                  results/c2_aff2/seed* results/c2_aff4/seed*
param(
    [int]$Start = 1,
    [int]$End = 30,
    [int[]]$Ks = @(1, 2, 4),
    [int]$Ticks = 16000,
    [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Wilfred\Documents\IA Inst\AetherLife\aetherlife_pkg"
Set-Location $root

$MaxAttempts = 3
$GpuWaitMaxSeconds = 600

function Test-GpuReady {
    try {
        & ".venv\Scripts\python.exe" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Wait-ForGpu {
    $waited = 0
    while (-not (Test-GpuReady)) {
        if ($waited -ge $GpuWaitMaxSeconds) {
            Write-Host ("[gpu]  toujours indisponible apres {0}s — on tente quand meme" -f $waited)
            return
        }
        Write-Host "[gpu]  indisponible — attente 30s puis re-test..."
        Start-Sleep -Seconds 30
        $waited += 30
    }
}

$tStart = Get-Date
Write-Host "[C2-REPLIC] start=$tStart  seeds=$Start..$End  k=$($Ks -join ',')"

foreach ($k in $Ks) {
    for ($s = $Start; $s -le $End; $s++) {
        $outDir = "results\c2_aff$k\seed$s"
        $report = "$outDir\overnight_v8b1_seed$s.json"
        if (Test-Path $report) {
            Write-Host "[skip] seed$s k$k (deja fait)"
            continue
        }
        $done = $false
        for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
            if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
            Wait-ForGpu
            $t0 = Get-Date
            $tag = if ($attempt -gt 1) { " (tentative $attempt/$MaxAttempts)" } else { "" }
            Write-Host "[run]  seed$s k$k  start=$t0$tag"
            & ".venv\Scripts\python.exe" "scripts\overnight_v8b1.py" `
                --ticks $Ticks --seed $s --device $Device `
                --regime coordination_collective `
                --n-initial-affinities $k `
                --out-dir $outDir
            $dt = (Get-Date) - $t0
            if (Test-Path $report) {
                Write-Host ("[ok]   seed$s k$k  dur={0:N0}s" -f $dt.TotalSeconds)
                $done = $true
                break
            }
            Write-Host ("[FAIL] seed$s k$k  dur={0:N0}s  tentative={1}/{2}" `
                -f $dt.TotalSeconds, $attempt, $MaxAttempts)
            if ($attempt -lt $MaxAttempts) { Start-Sleep -Seconds 15 }
        }
        if (-not $done) {
            Write-Host "[GIVEUP] seed$s k$k  abandon apres $MaxAttempts tentatives"
        }
    }
}

$tEnd = Get-Date
Write-Host "[C2-REPLIC] done=$tEnd  total=$([int]($tEnd-$tStart).TotalMinutes)min"
