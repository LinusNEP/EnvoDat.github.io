#!/usr/bin/env python
"""
Download EnvoDat programmatically, e.g. for training or evaluation. 

Usage:
    python download_envodat.py --scene mu-hall --format yolo --out ./data
    python download_envodat.py --all --out ./data           
    python download_envodat.py --list                       
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from urllib.parse import urljoin

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install pyyaml")

DEFAULT_MANIFEST = os.path.join(os.path.dirname(__file__), "download_manifest.yaml")
PROJECT_PAGE = "https://sites.google.com/view/envodat/download"


def load_manifest(path):
    if not os.path.exists(path):
        sys.exit(
            f"Manifest not found: {path}\n"
            f"Create it (see the docstring) or download manually from:\n  {PROJECT_PAGE}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_url(entry, base_url):
    url = entry["url"]
    if url.startswith("http"):
        return url
    return urljoin(base_url.rstrip("/") + "/", url)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    def _hook(count, block_size, total):
        if total > 0:
            pct = min(100, count * block_size * 100 // total)
            print(f"\r  {pct:3d}%  {os.path.basename(dest)}", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)
    print()


def verify(path, expected_sha):
    if not expected_sha:
        print(f"  (no checksum in manifest; skipping verification for {os.path.basename(path)})")
        return True
    actual = sha256(path)
    ok = actual.lower() == expected_sha.lower()
    print(f"  checksum {'OK' if ok else 'MISMATCH'}: {os.path.basename(path)}")
    if not ok:
        print(f"    expected {expected_sha}\n    got      {actual}")
    return ok


def select(entries, scene, fmt, take_all):
    if take_all:
        return entries
    out = [e for e in entries
           if (scene is None or e.get("scene") == scene)
           and (fmt is None or e.get("format") == fmt)]
    return out


def main():
    ap = argparse.ArgumentParser(description="Download & verify EnvoDat archives.")
    ap.add_argument("--scene", help="Scene id, e.g. mu-hall")
    ap.add_argument("--format", dest="fmt", help="Annotation format, e.g. yolo, coco, voc")
    ap.add_argument("--all", action="store_true", help="Download every manifest entry")
    ap.add_argument("--out", default="./data", help="Output directory")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--list", action="store_true", help="List manifest entries and exit")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip files already present with a matching checksum")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    base_url = manifest.get("base_url", "")
    entries = manifest.get("files", [])

    if args.list:
        for e in entries:
            print(f"  scene={e.get('scene'):<12} format={e.get('format'):<8} "
                  f"sha256={'yes' if e.get('sha256') else 'no'}")
        return

    if not (args.all or args.scene or args.fmt):
        ap.error("specify --scene/--format, or --all (or use --list).")

    targets = select(entries, args.scene, args.fmt, args.all)
    if not targets:
        sys.exit("No manifest entries matched your selection.")

    failures = 0
    for e in targets:
        url = resolve_url(e, base_url)
        fname = e.get("filename") or os.path.basename(url.split("?")[0])
        dest = os.path.join(args.out, fname)

        if args.skip_existing and os.path.exists(dest) and verify(dest, e.get("sha256")):
            print(f"  skip (already verified): {fname}")
            continue

        print(f"Downloading {fname} from {url}")
        try:
            download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR downloading {fname}: {exc}")
            failures += 1
            continue
        if not verify(dest, e.get("sha256")):
            failures += 1

    if failures:
        sys.exit(f"\nCompleted with {failures} failure(s).")
    print("\nAll downloads completed and verified.")


if __name__ == "__main__":
    main()
