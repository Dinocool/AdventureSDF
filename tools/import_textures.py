#!/usr/bin/env python3
"""Import PBR texture zips into assets/textures/.

Each source zip (e.g. "Cobble Stone.zip") contains numbered variant folders, each
with a full PBR set as uncompressed BMPs plus a preview PNG. This converts every
map to PNG (near-lossless; BMPs are uncompressed), renames each to its role, and
writes a per-material `material.ron` manifest.

Map roles:
  *_diffuseOriginal.bmp -> diffuse.png    (base color; the N.png is just a thumbnail)
  *_normal.bmp          -> normal.png
  *_metallic.bmp        -> metallic.png
  *_smoothness.bmp      -> roughness.png  (INVERTED: roughness = 1 - smoothness)
  *_ao.bmp              -> ao.png
  *_height.bmp          -> height.png
  *_edge.bmp            -> edge.png        (edge-wear mask)
  N.png                 -> preview.png     (material picker thumbnail)
"""

import io
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

# Source zip STEMS to import (the destination slug is derived by `slugify`). The full library the user
# dropped in Downloads — terrain materials map onto these in `assets/worldgen/biomes.ron`.
NAMES = [
    "Cobble Stone", "Sand", "Ground", "Grass", "Snow", "Snow Ground", "Groomed Snow", "Snowy Grass",
    "Ice", "Mud", "Beach", "Desert", "Stone Wall", "Cave Wall", "Cave Floor", "Tree Bark", "Roots",
    "Swamp", "Fall Ground", "Burned Earth", "Fire Grass", "Lava", "Volcano", "Hell", "Flesh",
    "Monster Skin", "Skull Floor", "Pile of Gold", "Water", "Bush_Hedge", "Snowy Hedge_Bush",
    "Wall_with_plants", "Damaged Wall", "Horror Walls", "Indoor Walls", "Wood Planks",
    "Magical Wood Planks", "Damaged Parquet", "Tiles", "Floor", "Roof", "Snow Roofs", "Metal",
    "Metal Plates", "Sci-Fi", "Alien Planet", "Alien Floor", "Mystical", "Magical Forrest", "Fur",
]


def slugify(name: str) -> str:
    """`"Stone Wall"` -> `"stone_wall"`, `"Wall_with_plants"` -> `"wall_with_plants"`."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# (source zip stem, destination material slug)
MATERIALS = [(n, slugify(n)) for n in NAMES]

# suffix in source -> (role filename stem, invert?)
MAP_ROLES = {
    "diffuseOriginal": ("diffuse", False),
    "normal": ("normal", False),
    "metallic": ("metallic", False),
    "smoothness": ("roughness", True),  # roughness = 1 - smoothness
    "ao": ("ao", False),
    "height": ("height", False),
    "edge": ("edge", False),
}

SRC_DIR = Path("C:/Users/Aesthetic/Downloads")
# Relative to THIS repo (the script lives in <repo>/tools/), so running the worktree copy writes into the
# worktree's assets — never the main checkout.
DEST_ROOT = Path(__file__).resolve().parent.parent / "assets" / "textures"

# A variant MAP file like "12_normal.bmp" OR "1_diffuseOriginal.png" -> ("12", "normal"). Some packs ship
# the diffuse (and occasionally other maps) as PNG rather than BMP, so match both extensions.
MAP_RE = re.compile(r"^(\d+)_([A-Za-z]+)\.(?:bmp|png)$")
# The bare numbered preview thumbnail, e.g. "12.png" (NOT a `_suffix` map).
PNG_RE = re.compile(r"^(\d+)\.png$")


def convert(data: bytes, invert: bool) -> bytes:
    img = Image.open(io.BytesIO(data))
    if invert:
        # Invert only the luminance channels; drop alpha if present.
        img = img.convert("L")
        img = ImageOps.invert(img)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def import_material(zip_stem: str, slug: str) -> int:
    zip_path = SRC_DIR / f"{zip_stem}.zip"
    if not zip_path.exists():
        print(f"  SKIP missing {zip_path}", flush=True)
        return 0

    dest_mat = DEST_ROOT / slug
    variants: dict[str, dict[str, str]] = {}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            base = name.rsplit("/", 1)[-1]
            if not base:
                continue
            # which variant folder? path like "Cobble Stone/12/12_normal.bmp"
            parts = [p for p in name.split("/") if p]
            if len(parts) < 2:
                continue
            variant = parts[-2]

            m = MAP_RE.match(base)
            if m:
                vnum, suffix = m.group(1), m.group(2)
                if suffix not in MAP_ROLES:
                    continue  # unknown map, skip
                role, invert = MAP_ROLES[suffix]
                out_dir = dest_mat / vnum
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{role}.png"
                variants.setdefault(vnum, {})[role] = f"{role}.png"
                if out_path.exists():
                    continue  # already imported (idempotent re-run only fills gaps)
                out_path.write_bytes(convert(zf.read(name), invert))
                print(f"  {slug}/{vnum}/{role}.png", flush=True)
                continue

            p = PNG_RE.match(base)
            if p:
                vnum = p.group(1)
                out_dir = dest_mat / vnum
                out_dir.mkdir(parents=True, exist_ok=True)
                variants.setdefault(vnum, {})["preview"] = "preview.png"
                preview_path = out_dir / "preview.png"
                if preview_path.exists():
                    continue
                # Re-encode the preview through PIL too (consistent, stripped).
                preview_path.write_bytes(convert(zf.read(name), False))

    write_manifest(dest_mat, zip_stem, slug, variants)
    return len(variants)


def write_manifest(dest_mat: Path, name: str, slug: str, variants: dict) -> None:
    # Sort variant ids numerically.
    ids = sorted(variants.keys(), key=lambda s: int(s))
    lines = [
        "// Auto-generated by tools/import_textures.py. Lists the PBR variants for",
        "// this material; each variant folder holds role-named PNG maps.",
        "(",
        f'    name: "{name}",',
        f'    slug: "{slug}",',
        "    variants: [",
    ]
    for vid in ids:
        roles = variants[vid]
        have = ", ".join(sorted(roles.keys()))
        lines.append(f'        (id: {vid}, dir: "{vid}"),  // {have}')
    lines.append("    ],")
    lines.append(")")
    (dest_mat / "material.ron").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {slug}/material.ron ({len(ids)} variants)", flush=True)


def main() -> int:
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    for zip_stem, slug in MATERIALS:
        print(f"== {zip_stem} -> {slug} ==", flush=True)
        total += import_material(zip_stem, slug)
    print(f"DONE: {total} variants imported across {len(MATERIALS)} materials.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
