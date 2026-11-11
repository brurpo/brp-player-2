# Copies the Maya and Blender OCIO configs the player offers ("(maya)", "(blender)"
# entries) from local installs into third_party/ocio-configs/{maya,blender}; the build
# copies them next to the executable (ocio/). DESIGN.md §7d.
#
#   powershell -ExecutionPolicy Bypass -File tools/fetch_ocio_configs.ps1 `
#       -Maya "D:\maya\Maya2027\resources\OCIO-configs\Maya2022-default" `
#       -Blender "D:\blender-5.2.3\5.2\datafiles\colormanagement"
param([Parameter(Mandatory)][string]$Maya, [Parameter(Mandatory)][string]$Blender)
$ErrorActionPreference = "Stop"
$dst = Join-Path (Split-Path -Parent $PSScriptRoot) "third_party/ocio-configs"
New-Item -ItemType Directory -Force $dst | Out-Null
foreach ($pair in @(@("maya", $Maya), @("blender", $Blender))) {
    $to = Join-Path $dst $pair[0]
    if (Test-Path $to) { Remove-Item -Recurse -Force $to }
    Copy-Item -Recurse $pair[1] $to
    Write-Host "$($pair[0]): $($pair[1]) -> $to"
}
