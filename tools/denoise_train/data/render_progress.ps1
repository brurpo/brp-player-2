# Progress of the noisy-variant renders (render_all.sh): frames done per scene and
# variant, overall percentage, and an estimate of the time left from the speed so far.
#   powershell -File tools\denoise_train\data\render_progress.ps1
$root = "D:\denoise_data\src\3D_RENDERS"
$scenes = "blender-5.2-splash", "lone-monk_cycles_and_exposure-node_demo", "Blenderman", "barbershop_interior", "thejunkshopsplashscreen"
$variants = foreach ($s in "16", "64", "128", "1024T") { foreach ($k in "static", "random") { "${s}_${k}_seed" } }
$perVariant = 25
$total = $scenes.Count * $variants.Count * $perVariant
$done = 0
$first = $null
$last = $null
foreach ($sc in $scenes) {
    $line = "{0,-42}" -f $sc
    $n = 0
    foreach ($v in $variants) {
        $files = Get-ChildItem "$root\$sc\$v\*.exr" -ErrorAction SilentlyContinue | Where-Object { $_.Length -gt 0 }
        $c = ($files | Measure-Object).Count
        $n += $c
        foreach ($f in $files) {
            if (-not $first -or $f.LastWriteTime -lt $first) { $first = $f.LastWriteTime }
            if (-not $last -or $f.LastWriteTime -gt $last) { $last = $f.LastWriteTime }
        }
        $line += " {0,2}" -f $c
    }
    $done += $n
    $line += "   {0,3}/{1}" -f $n, ($variants.Count * $perVariant)
    Write-Output $line
}
Write-Output ("{0,-42} {1}" -f "", (($variants | ForEach-Object { $_.Split('_')[0] + $_.Split('_')[1][0] }) -join " "))
$pct = [math]::Round(100 * $done / $total, 1)
Write-Output ""
Write-Output "Overall: $done / $total frames ($pct%)"
if ($done -gt 1 -and $first) {
    $elapsed = (Get-Date) - $first
    $rate = $done / [math]::Max($elapsed.TotalSeconds, 1)
    $left = [TimeSpan]::FromSeconds(($total - $done) / $rate)
    Write-Output ("Running {0:hh\:mm}, about {1:N1} s per frame, roughly {2:hh\:mm} left (heavier scenes come later: treat as a lower bound)" -f $elapsed, (1 / $rate), $left)
    Write-Output ("Last frame written {0:HH:mm:ss}" -f $last)
}
$running = Get-Process blender -ErrorAction SilentlyContinue
Write-Output ("Blender running: " + $(if ($running) { "yes" } else { "no" }))
