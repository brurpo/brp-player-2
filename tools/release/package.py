"""Release packaging: the app zip (GitHub release) and the components (Google Drive).

  python tools/release/package.py drive   # build the component zips for Drive
  python tools/release/package.py app     # build the app zip (needs drive_ids.json)

Everything goes to dist/:
  dist/BRP_PLAYER_2_ASSETS/v<version>/<component>-<cversion>.zip
                               upload into the Drive folder BRP_PLAYER_2_ASSETS (shared
                               once, "anyone with the link"; versions inherit it)
  dist/drive/components.json   name, dir, version, file, sha256, size of every component
  dist/drive_ids.json          you fill in: component file name -> Drive file id
  dist/brp-player-<version>-windows-x64.zip            the GitHub release asset

Components keep their version until their content changes (same SHA-256): a release
re-uses the Drive file of the last one that had it, so each Drive version folder only
holds what is new or changed (dist/drive/published.json remembers what was uploaded).

The app zip holds what the app needs to start: brp-player.exe, FFmpeg's LGPL DLLs,
the VC++ runtime, shaders, manifest.json, the licenses. Not in it (components,
downloaded on first start): the GPL ffmpeg.exe (x264/x265), the AI models, the NNEDI3
weights (GPL) and the RVM model (GPL).
"""
import hashlib, json, os, re, shutil, sys, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIST = os.path.join(ROOT, "dist")
REL = os.path.join(ROOT, "target", "release")
REPO = "brurpo/brp-player-2"
# Drive: one folder shared once ("anyone with the link can view"); each release adds
# a v<version> folder inside it, which inherits the sharing.
ASSETS = "BRP_PLAYER_2_ASSETS"
FF_LGPL = os.path.join(ROOT, "third_party", "ffmpeg-lgpl", "7.1")
FF_GPL = os.environ.get("BRP_FFMPEG_GPL_DIR") or os.path.join(
    os.environ["LOCALAPPDATA"], "Microsoft", "WinGet", "Packages",
    "BtbN.FFmpeg.GPL.Shared.7.1_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "ffmpeg-n7.1.5-12-g1fdbca85aa-win64-gpl-shared-7.1")
FF_VERSION = "7.1.5-12"
DATE = (2026, 1, 1, 0, 0, 0)  # fixed zip timestamps: same content, same checksum


def version():
    s = open(os.path.join(ROOT, "Cargo.toml"), encoding="utf-8").read()
    return re.search(r'\[workspace\.package\][^\[]*?version\s*=\s*"([^"]+)"', s, re.S).group(1)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_hash(files):
    """Content hash of (relative name, file) pairs: decides a component's version."""
    h = hashlib.sha256()
    for rel, path in sorted(files):
        h.update(rel.replace("\\", "/").encode() + b"\0" + sha256(path).encode())
    return h.hexdigest()[:12]


def write_zip(out, files, store_ext=(".bin", ".dll", ".exe", ".zip")):
    """files: [(name in zip, path)], deterministic order and timestamps."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with zipfile.ZipFile(tmp, "w") as z:
        for rel, path in sorted(files):
            zi = zipfile.ZipInfo(rel.replace("\\", "/"), DATE)
            zi.external_attr = 0o644 << 16
            zi.compress_type = zipfile.ZIP_STORED if path.lower().endswith(store_ext) and os.path.getsize(path) > (8 << 20) else zipfile.ZIP_DEFLATED
            with open(path, "rb") as f:
                z.writestr(zi, f.read(), compresslevel=6)
    os.replace(tmp, out)


def walk(d, skip=lambda rel: False):
    out = []
    for base, _, names in os.walk(d):
        for n in names:
            p = os.path.join(base, n)
            rel = os.path.relpath(p, d)
            if not skip(rel):
                out.append((rel, p))
    return out


def components():
    """(name, dir in the install, files, license note)."""
    out = []
    gpl_bin = os.path.join(FF_GPL, "bin")
    ff = [(n, os.path.join(gpl_bin, n)) for n in os.listdir(gpl_bin) if n == "ffmpeg.exe" or n.endswith(".dll")]
    ff.append(("LICENSE.txt", os.path.join(FF_GPL, "LICENSE.txt")))
    out.append(("ffmpeg-gpl", "ffmpeg-gpl", ff))
    models = os.path.join(REL, "models")
    for m in sorted(os.listdir(models)):
        files = walk(os.path.join(models, m), skip=lambda rel: rel.endswith(".safetensors"))
        out.append((f"models-{m}", f"models/{m}", files))
    out.append(("nnedi3", "nnedi3", [("nnedi3_weights.bin", os.path.join(ROOT, "crates", "brp-render", "assets", "nnedi3_weights.bin"))]))
    return out


def drive():
    ver = version()
    pub_path = os.path.join(DIST, "drive", "published.json")
    published = json.load(open(pub_path)) if os.path.exists(pub_path) else {}
    listing = []
    for name, d, files in components():
        cver = tree_hash(files)
        prev = published.get(name)
        if prev and prev["version"] == cver:
            listing.append(prev)
            print(f"{name}: unchanged ({prev['folder']}/{prev['file']})")
            continue
        folder = f"v{ver}"
        fname = f"{name}-{cver}.zip"
        out = os.path.join(DIST, ASSETS, folder, fname)
        write_zip(out, files)
        e = {"name": name, "dir": d, "version": cver, "folder": folder, "file": fname, "sha256": sha256(out), "size": os.path.getsize(out)}
        listing.append(e)
        print(f"{name}: {folder}/{fname} {e['size'] / 1e6:.0f} MB")
    json.dump(listing, open(os.path.join(DIST, "drive", "components.json"), "w"), indent=1)
    ids = os.path.join(DIST, "drive_ids.json")
    known = json.load(open(ids)) if os.path.exists(ids) else {}
    for e in listing:
        known.setdefault(e["file"], "")
    json.dump(known, open(ids, "w"), indent=1)
    print(f"\nUpload dist/{ASSETS}/v{ver}/ into the Drive folder {ASSETS} (new files only), then put their ids in {ids}")


def app():
    ver = version()
    listing = json.load(open(os.path.join(DIST, "drive", "components.json")))
    ids = json.load(open(os.path.join(DIST, "drive_ids.json")))
    comps = []
    for e in listing:
        fid = ids.get(e["file"], "")
        if not fid:
            sys.exit(f"no Drive id for {e['file']} in dist/drive_ids.json")
        comps.append({k: e[k] for k in ("name", "dir", "version", "sha256", "size")} | {"url": f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"})
    manifest = {"version": ver, "repo": REPO, "components": comps}
    stage = os.path.join(DIST, "stage")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    json.dump(manifest, open(os.path.join(stage, "manifest.json"), "w"), indent=1)
    top = f"brp-player-{ver}"
    files = [(f"{top}/brp-player.exe", os.path.join(REL, "brp-player.exe")), (f"{top}/manifest.json", os.path.join(stage, "manifest.json"))]
    for n in os.listdir(os.path.join(FF_LGPL, "bin")):
        if n.endswith(".dll"):
            files.append((f"{top}/{n}", os.path.join(FF_LGPL, "bin", n)))
    for n in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        files.append((f"{top}/{n}", os.path.join(os.environ["SystemRoot"], "System32", n)))
    for rel, p in walk(os.path.join(REL, "shaders")):
        files.append((f"{top}/shaders/{rel}", p))
    for n in ("LICENSE.md", "THIRD_PARTY_NOTICES.md", "README.md"):
        files.append((f"{top}/{n}", os.path.join(ROOT, n)))
    files.append((f"{top}/licenses/ffmpeg-lgpl/LICENSE.txt", os.path.join(FF_LGPL, "LICENSE.txt")))
    out = os.path.join(DIST, f"brp-player-{ver}-windows-x64.zip")
    write_zip(out, files)
    print(f"{out}: {os.path.getsize(out) / 1e6:.0f} MB, {len(files)} files")
    # What was uploaded for this release, for the next one to re-use.
    pub = {e["name"]: e for e in listing}
    json.dump(pub, open(os.path.join(DIST, "drive", "published.json"), "w"), indent=1)


if __name__ == "__main__":
    {"drive": drive, "app": app}[sys.argv[1]]()
