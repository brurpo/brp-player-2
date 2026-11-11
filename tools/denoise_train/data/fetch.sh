#!/usr/bin/env bash
# Training data for the stable denoiser (PLAN.md §3): downloads what sources.csv lists
# into $DENOISE_DATA/raw, resumable (curl -C -), checked by size and, where known,
# SHA-256. Unpacks into $DENOISE_DATA/src/<name>.
#
#   DENOISE_DATA=/path/to/data bash tools/denoise_train/data/fetch.sh [name substring]...
#
# With names, only the matching rows (e.g. "REDS validation"). Keeps the archives
# unless KEEP_ZIPS=0 (then each is deleted once unpacked and checked).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
data="${DENOISE_DATA:?set DENOISE_DATA to the folder for the training data}"
mkdir -p "$data/raw" "$data/src"
py="$(cd "$here/../../.." && pwd)/third_party/pyref/venv/Scripts/python.exe"
[ -x "$py" ] || py=python3

# (tr: Python on Windows ends lines with CR, which would stick to the checksum)
"$py" -I - "$here/sources.csv" "$@" <<'EOF' | tr -d '\r' | while IFS=$'\t' read -r name file url bytes sha; do
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
want = sys.argv[2:]
for r in rows:
    if want and not any(w.lower() in r["name"].lower() for w in want):
        continue
    # Excluded sources, and Netflix's raw clips (netflix.py fetches and cuts those).
    if r["use"] == "excluded" or "opencontent.netflix" in r["url"]:
        continue
    print("\t".join([r["name"], r["file"], r["url"], r["bytes"], r["sha256"]]))
EOF
  out="$data/raw/$file"
  echo "== $name ($((bytes / 1000000)) MB)"
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$have" != "$bytes" ]; then
    curl -L --fail --retry 5 --retry-delay 10 -C - -o "$out" "$url"
  fi
  have=$(stat -c %s "$out")
  [ "$have" = "$bytes" ] || { echo "size $have, expected $bytes: incomplete, run again to resume"; exit 1; }
  if [ -n "$sha" ]; then
    echo "$sha  $out" | sha256sum -c --quiet || { echo "checksum mismatch: delete $out and run again"; exit 1; }
  else
    echo "  sha256 (record it in sources.csv): $(sha256sum "$out" | cut -d' ' -f1)"
  fi
  dest="$data/src/$(echo "$name" | tr ' ()' '___' | tr -s _)"
  if [ ! -e "$dest/.done" ]; then
    mkdir -p "$dest"
    case "$file" in
      *.zip) unzip -q -o "$out" -d "$dest" ;;
      *) cp "$out" "$dest/" ;;
    esac
    touch "$dest/.done"
  fi
  [ "${KEEP_ZIPS:-1}" = 0 ] && rm -f "$out"
  echo "  ready: $dest"
done
