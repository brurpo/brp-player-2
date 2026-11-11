#!/bin/sh
# libplacebo reference render of one frame with a user shader (via FFmpeg's libplacebo filter).
#   tools/plref.sh <video> <frame> <shader.glsl|none> <out.gbrp16> [frames_before]
# Output: raw planar G, B, R, 16-bit little endian, full range, source size.
set -e
cd "$(dirname "$0")/.."
FF="$(grep FFMPEG_DIR .cargo/config.toml | cut -d'"' -f2)"
E="$(cygpath "$FF")/bin/ffmpeg.exe"
video="$1"; n="$2"; shader="$3"; out="$4"; before="${5:-0}"
start=$((n - before))
sel="select=between(n\,$start\,$n)"
if [ "$shader" = none ]; then pl="libplacebo=format=gbrp16le"; else
  dir="$(cygpath -m "$(dirname "$shader")")"; base="$(basename "$shader")"
  cd "$dir"; pl="libplacebo=format=gbrp16le:custom_shader_path=$base"; fi
"$E" -hide_banner -v error -init_hw_device vulkan -i "$video" -vf "$sel,$pl" -fps_mode passthrough -f rawvideo -pix_fmt gbrp16le -y "$out.all"
size=$(stat -c %s "$out.all"); count=$((before + 1)); one=$((size / count))
tail -c $one "$out.all" > "$out"; rm -f "$out.all"
echo "reference: $out ($one bytes)"
