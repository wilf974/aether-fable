# C2 — test causal diversite d'affinite. 10 seeds x {1,2,4}, design apparie.
# Idempotent : skip un (seed,k) si son report existe deja.
param(
    [int]$Start = 1,
    [int]$End = 10,
    [int[]]$Ks = @(1, 2, 4),
    [int]$Ticks = 16000,
    [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Wilfred\Documents\IA Inst\AetherLife\aetherlife_pkg"
Set-Location $root
foreach ($k in $Ks) {
    for ($s = $Start; $s -le $End; $s++) {
        $outDir = "results\c2_aff$k\seed$s"
        $report = "$outDir\overnight_v8b1_seed$s.json"
        if (Test-Path $report) {
            Write-Host "SKIP seed$s k$k (deja fait)"
            continue
        }
        Write-Host "RUN seed$s k$k $(Get-Date -Format HH:mm:ss)"
        & ".venv\Scripts\python.exe" "scripts\overnight_v8b1.py" `
            --ticks $Ticks --seed $s --device $Device `
            --regime coordination_collective `
            --n-initial-affinities $k `
            --out-dir $outDir
    }
}
Write-Host "C2 BATCH DONE $(Get-Date -Format HH:mm:ss)"
