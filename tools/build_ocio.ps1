# Builds OpenColorIO (BSD-3) for the player into third_party/ocio/<version>:
# include/, lib/OpenColorIO.lib, bin/OpenColorIO_2_5.dll. Its dependencies (yaml-cpp,
# expat, pystring, minizip-ng, zlib, Imath; MIT/BSD/zlib) are fetched and linked in
# statically by OCIO's own CMake. Needs CMake, Git and Visual Studio 2022 (MSVC).
#
#   powershell -ExecutionPolicy Bypass -File tools/build_ocio.ps1
param([string]$Version = "2.5.2")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "third_party/ocio/src-$Version"
$build = Join-Path $root "third_party/ocio/build-$Version"
$prefix = Join-Path $root "third_party/ocio/$Version"
if (-not (Test-Path $src)) {
    git clone --depth 1 --branch "v$Version" https://github.com/AcademySoftwareFoundation/OpenColorIO.git $src
}
cmake -S $src -B $build -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_INSTALL_PREFIX="$prefix" `
    -DBUILD_SHARED_LIBS=ON `
    -DOCIO_INSTALL_EXT_PACKAGES=ALL `
    -DOCIO_BUILD_APPS=OFF -DOCIO_BUILD_TESTS=OFF -DOCIO_BUILD_GPU_TESTS=OFF `
    -DOCIO_BUILD_PYTHON=OFF -DOCIO_BUILD_DOCS=OFF -DOCIO_BUILD_OPENFX=OFF -DOCIO_BUILD_JAVA=OFF
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --build $build --config Release --target install --parallel
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
Write-Host "OpenColorIO $Version installed in $prefix"
