# Generalite portfolio effect : k {1,4} x n_seed_points {4,8,16} x seeds.
# Idempotent : skip une cellule (seed,k,n) si son report existe deja.
param(
    [int]$Start = 1,
    [int]$End = 8,
    [int[]]$Ks = @(1, 4),
    [int[]]$Ns = @(4, 8, 16),
    [int]$Ticks = 16000,
    [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$root = "C:\Users\Wilfred\Documents\IA Inst\AetherLife\aetherlife_pkg"
Set-Location $root
foreach ($n in $Ns) {
    foreach ($k in $Ks) {
        for ($s = $Start; $s -le $End; $s++) {
            $outDir = "results\topo_n${n}_k${k}\seed$s"
            $report = "$outDir\overnight_v8b1_seed$s.json"
            if (Test-Path $report) {
                Write-Host "SKIP seed$s k$k n$n (deja fait)"
                continue
            }
            Write-Host "RUN seed$s k$k n$n $(Get-Date -Format HH:mm:ss)"
            & ".venv\Scripts\python.exe" "scripts\overnight_v8b1.py" `
                --ticks $Ticks --seed $s --device $Device `
                --regime coordination_collective `
                --n-initial-affinities $k `
                --n-seed-points $n `
                --out-dir $outDir
        }
    }
}
Write-Host "TOPOLOGY BATCH DONE $(Get-Date -Format HH:mm:ss)"
