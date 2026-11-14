"""Generate a synthetic photo library for performance testing.

    python scripts/make_test_library.py <dest> --count 20000

Photos get varied sizes, EXIF dates spread over several years, and a share of
exact duplicates placed in a second folder.
"""

from __future__ import annotations

import argparse
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw

SIZES = [(2000, 1500), (1500, 2000), (2400, 1350), (1800, 1800), (3000, 1000)]


def make_one(args: tuple[str, int, int]) -> None:
    dest, index, seed = args
    rng = random.Random(seed * 1_000_003 + index)
    w, h = rng.choice(SIZES)
    # Low-resolution noise scaled up gives photo-like file sizes without slow per-pixel work.
    noise = Image.frombytes("RGB", (w // 20, h // 20), rng.randbytes((w // 20) * (h // 20) * 3))
    img = noise.resize((w, h), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img)
    color = tuple(rng.randrange(256) for _ in range(3))
    draw.ellipse((w * 0.2, h * 0.2, w * 0.8, h * 0.8), fill=color)
    draw.text((w * 0.45, h * 0.48), str(index), fill=(255, 255, 255))

    taken = datetime(2015, 1, 1) + timedelta(minutes=rng.randrange(10 * 365 * 24 * 60))
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = taken.strftime("%Y:%m:%d %H:%M:%S")
    exif[271], exif[272] = "Pixora", "Synthetic 1"
    path = Path(dest) / f"{taken.year}" / f"IMG_{index:06d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=85, exif=exif)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dest", type=Path)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--duplicates", type=float, default=0.05, help="share copied again")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    originals = args.dest / "photos"
    with ProcessPoolExecutor() as pool:
        list(pool.map(make_one, [(str(originals), i, args.seed) for i in range(args.count)], chunksize=50))

    files = sorted(originals.rglob("*.jpg"))
    copies = args.dest / "backup-copies"
    copies.mkdir(parents=True, exist_ok=True)
    for src in random.Random(args.seed).sample(files, int(len(files) * args.duplicates)):
        shutil.copy2(src, copies / f"copy_{src.name}")
    print(f"{len(files)} photos and {int(len(files) * args.duplicates)} duplicate files in {args.dest}")


if __name__ == "__main__":
    main()
