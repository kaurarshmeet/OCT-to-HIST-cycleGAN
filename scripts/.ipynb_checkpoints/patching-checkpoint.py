"""
patching.py
Authors: Arshmeet Kaur
Description:
    Extracts non-overlapping 256x256 patches from paired OCT and H&E .tif images.
    Images are trimmed to the nearest multiple of 256 before patching.
    Saves a patch_pairs.json mapping each original OCT patch to its corresponding
    H&E patch (for stitching purposes only).

    Optionally applies geometric data augmentation independently to each domain:
        - Horizontal flip
        - Vertical flip
        - 90°, 180°, 270° rotations
    Since CycleGAN samples trainA and trainB independently, augmentations are
    applied per-domain and do not need to be paired. Augmented patches are
    training-only and excluded from patch_pairs.json.

Usage:
    # Without augmentation (e.g. test set)
    python patching.py --inputA /path/to/testA --inputB /path/to/testB --output /path/to/output

    # With augmentation (e.g. train set)
    python patching.py --inputA /path/to/trainA --inputB /path/to/trainB --output /path/to/output --augment
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image


PATCH_SIZE = 256

AUGMENTATIONS = [
    ("aug_hflip",  lambda p: np.fliplr(p).copy()),
    ("aug_vflip",  lambda p: np.flipud(p).copy()),
    ("aug_rot90",  lambda p: np.rot90(p, k=1).copy()),
    ("aug_rot180", lambda p: np.rot90(p, k=2).copy()),
    ("aug_rot270", lambda p: np.rot90(p, k=3).copy()),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def trim_and_patch(image: np.ndarray, patch_size: int):
    """
    Trim image to nearest multiple of patch_size and extract non-overlapping patches.
    Returns list of (patch, row_idx, col_idx) tuples.
    """
    h, w = image.shape[:2]
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    image = image[:new_h, :new_w]

    patches = []
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append((patch, i // patch_size, j // patch_size))

    return patches


def extract_sample_id(filename: str):
    """
    Extract sample number from filename.
    e.g. 'silver_3_oct.tif' -> '3'
    """
    match = re.search(r'silver_(\d+)_', filename)
    if match:
        return match.group(1)
    return None


def save_patches(patches, out_dir, sample_id, augment=False):
    """
    Save patches to output directory.
    If augment=True, also save all geometric augmentations independently.
    """
    for patch, row, col in patches:
        patch_name = f"silver_{sample_id}_patch_{row}_{col}.jpg"
        Image.fromarray(patch).save(out_dir / patch_name)

        if augment:
            for aug_label, aug_fn in AUGMENTATIONS:
                aug_name = f"silver_{sample_id}_patch_{row}_{col}_{aug_label}.jpg"
                Image.fromarray(aug_fn(patch)).save(out_dir / aug_name)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_pairs(inputA_dir, inputB_dir, output_dir, augment=False):
    """
    For each paired OCT/H&E image:
        - Extract patches from both
        - Verify patch counts match
        - Save patches (+ augmentations if requested) independently per domain
        - Build patch_pairs.json from original patches only (for stitching)
    """
    inputA_dir = Path(inputA_dir)
    inputB_dir = Path(inputB_dir)
    output_dir = Path(output_dir)

    out_A = output_dir / "trainA"
    out_B = output_dir / "trainB"
    out_A.mkdir(parents=True, exist_ok=True)
    out_B.mkdir(parents=True, exist_ok=True)

    oct_files = sorted(inputA_dir.glob("*.tif"))
    if not oct_files:
        print("No .tif files found in inputA directory!")
        return

    patch_pairs    = {}
    total_original = 0

    for oct_file in oct_files:
        sample_id = extract_sample_id(oct_file.name)
        if sample_id is None:
            print(f"Could not extract sample ID from {oct_file.name}, skipping.")
            continue

        he_file = inputB_dir / f"silver_{sample_id}_he.tif"
        if not he_file.exists():
            print(f"WARNING: No matching H&E file found for {oct_file.name}, skipping.")
            continue

        print(f"\nProcessing pair: silver_{sample_id}")
        print(f"  OCT: {oct_file.name}")
        print(f"  H&E: {he_file.name}")

        oct_img = np.array(Image.open(oct_file))
        he_img  = np.array(Image.open(he_file))

        print(f"  OCT shape: {oct_img.shape}")
        print(f"  H&E shape: {he_img.shape}")

        oct_patches = trim_and_patch(oct_img, PATCH_SIZE)
        he_patches  = trim_and_patch(he_img,  PATCH_SIZE)

        if len(oct_patches) != len(he_patches):
            print(f"  WARNING: Patch count mismatch! OCT={len(oct_patches)}, H&E={len(he_patches)}. Skipping.")
            continue

        print(f"  Original patches: {len(oct_patches)}")

        # Build patch_pairs.json from original patches only
        for (_, row, col) in oct_patches:
            patch_name = f"silver_{sample_id}_patch_{row}_{col}.jpg"
            patch_pairs[str(Path("trainA") / patch_name)] = str(Path("trainB") / patch_name)

        # Save patches independently per domain
        save_patches(oct_patches, out_A, sample_id, augment)
        save_patches(he_patches,  out_B, sample_id, augment)

        total_original += len(oct_patches)

        if augment:
            print(f"  Augmented patches per domain: {len(oct_patches) * len(AUGMENTATIONS)}")

    # Save patch pairs (original only — used for stitching)
    pairs_path = output_dir / "patch_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(patch_pairs, f, indent=2)

    n_aug = len(AUGMENTATIONS)
    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Original patch pairs:           {total_original}")
    if augment:
        print(f"  Augmented patches (per domain): {total_original * n_aug}")
        print(f"  Total patches per domain:       {total_original * (1 + n_aug)}")
    print(f"  patch_pairs.json saved to {pairs_path}")
    print(f"  (original patches only — for stitching)")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract paired OCT/H&E patches for CycleGAN training.")
    parser.add_argument("--inputA", type=str, required=True, help="Path to OCT .tif files")
    parser.add_argument("--inputB", type=str, required=True, help="Path to H&E .tif files")
    parser.add_argument("--output", type=str, required=True, help="Path to save output patches")
    parser.add_argument("--augment", action="store_true",    help="Apply geometric augmentation independently per domain")
    args = parser.parse_args()

    print(f"Input OCT dir:  {args.inputA}")
    print(f"Input H&E dir:  {args.inputB}")
    print(f"Output dir:     {args.output}")
    print(f"Patch size:     {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Augmentation:   {'ON' if args.augment else 'OFF'}\n")

    process_pairs(
        inputA_dir=args.inputA,
        inputB_dir=args.inputB,
        output_dir=args.output,
        augment=args.augment
    )