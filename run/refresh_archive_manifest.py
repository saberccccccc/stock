"""Write a deterministic manifest for a retained archive directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", default="archive/experiments_202606")
    args = parser.parse_args(argv)
    archive_dir = ROOT / args.archive_dir
    rows = []
    for path in sorted(archive_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir():
            continue
        size = directory_size(path)
        rows.append(
            {
                "name": path.name,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": size,
                "size_gb": round(size / (1024 ** 3), 6),
            }
        )
    manifest = archive_dir / "MIGRATION_MANIFEST_20260711.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "path", "size_bytes", "size_gb"])
        writer.writeheader()
        writer.writerows(rows)
    print({"archive_dir": str(archive_dir), "entries": len(rows), "manifest": str(manifest)}, flush=True)


if __name__ == "__main__":
    main()
