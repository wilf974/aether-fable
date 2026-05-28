param(
    [int]$Start = 32,
    [int]$End   = 50
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$root = "C:\Users\Wilfred\Documents\IA Inst\AetherLife\aetherlife_pkg"
Set-Location $root
& "$root\.venv\Scripts\Activate.ps1"

$logDir = "$root\results\v8c3d\_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$tStart = Get-Date
Write-Host "[phase D resume] start=$tStart  seeds=$Start..$End"

foreach ($s in $Start..$End) {
    $outDir = "$root\results\v8c3d\seed$s"
    $log    = "$logDir\seed$s.log"

    if (Test-Path "$outDir\overnight_v8b1_seed$s.json") {
        Write-Host "[skip] seed=$s already complete"
        continue
    }

    # repartir propre si dossier vide laissé par une interruption
    if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }

    $t0 = Get-Date
    Write-Host "[run]  seed=$s  start=$t0"
    & python scripts/overnight_v8b1.py `
        --ticks 16000 `
        --regime coordination_collective `
        --vocalize-cost 0.05 `
        --seed $s `
        --out-dir $outDir `
        --device cuda *>&1 | Tee-Object -FilePath $log | Out-Null
    $dt = (Get-Date) - $t0
    if (Test-Path "$outDir\overnight_v8b1_seed$s.json") {
        Write-Host ("[ok]   seed={0}  dur={1:N0}s" -f $s, $dt.TotalSeconds)
    } else {
        Write-Host ("[FAIL] seed={0}  dur={1:N0}s  see $log" -f $s, $dt.TotalSeconds)
    }
}

$tEnd = Get-Date
Write-Host "[phase D resume] done=$tEnd  total=$([int]($tEnd-$tStart).TotalMinutes)min"
