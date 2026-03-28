"""
patch.py
Authors: Arshmeet Kaur
Description:
    Extracts non-overlapping 256x256 patches from paired OCT and H&E .tif images.
    Images are trimmed to the nearest multiple of 256 before patching.
    Saves a patch_pairs.json mapping each OCT patch to its corresponding H&E patch.

Usage:
    python patch.py --trainA /path/to/trainA --trainB /path/to/trainB --output /path/to/output

Output structure:
    output/
    ├── trainA/         <- OCT patches
    ├── trainB/         <- H&E patches
    └── patch_pairs.json
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image


PATCH_SIZE = 256


def trim_and_patch(image: np.ndarray, patch_size: int):
    """
    Trim image to nearest multiple of patch_size and extract non-overlapping patches.
    Returns list of (patch, row_idx, col_idx) tuples.
    """
    h, w = image.shape[:2]

    # Trim to nearest multiple of patch_size
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


def process_pairs(trainA_dir, trainB_dir, output_dir):
    """
    For each paired OCT/H&E image:
    - Extract patches from both
    - Verify patch counts match
    - Save patches to output directories
    - Build patch pair mapping
    """
    trainA_dir = Path(trainA_dir)
    trainB_dir = Path(trainB_dir)
    output_dir = Path(output_dir)

    # Create output directories
    out_A = output_dir / "trainA"
    out_B = output_dir / "trainB"
    out_A.mkdir(parents=True, exist_ok=True)
    out_B.mkdir(parents=True, exist_ok=True)

    # Find all OCT files
    oct_files = sorted(trainA_dir.glob("*.tif"))
    if not oct_files:
        print("No .tif files found in trainA directory!")
        return

    patch_pairs = {}
    total_patches = 0

    for oct_file in oct_files:
        sample_id = extract_sample_id(oct_file.name)
        if sample_id is None:
            print(f"Could not extract sample ID from {oct_file.name}, skipping.")
            continue

        # Find corresponding H&E file
        he_file = trainB_dir / f"silver_{sample_id}_he.tif"
        if not he_file.exists():
            print(f"WARNING: No matching H&E file found for {oct_file.name}, skipping.")
            continue

        print(f"\nProcessing pair: silver_{sample_id}")
        print(f"  OCT: {oct_file.name}")
        print(f"  H&E: {he_file.name}")

        # Load images
        oct_img = np.array(Image.open(oct_file))
        he_img  = np.array(Image.open(he_file))

        print(f"  OCT shape: {oct_img.shape}")
        print(f"  H&E shape: {he_img.shape}")

        # Extract patches
        oct_patches = trim_and_patch(oct_img, PATCH_SIZE)
        he_patches  = trim_and_patch(he_img,  PATCH_SIZE)

        # Verify patch counts match
        if len(oct_patches) != len(he_patches):
            print(f"  WARNING: Patch count mismatch! OCT={len(oct_patches)}, H&E={len(he_patches)}. Skipping.")
            continue

        print(f"  Patches: {len(oct_patches)} per image")

        # Save patches and build mapping
        for (oct_patch, row, col), (he_patch, _, _) in zip(oct_patches, he_patches):
            patch_name = f"silver_{sample_id}_patch_{row}_{col}.jpg"

            oct_patch_path = out_A / patch_name
            he_patch_path  = out_B / patch_name

            Image.fromarray(oct_patch).save(oct_patch_path)
            Image.fromarray(he_patch).save(he_patch_path)

            # Map OCT patch -> H&E patch (relative paths)
            patch_pairs[str(Path("trainA") / patch_name)] = str(Path("trainB") / patch_name)

        total_patches += len(oct_patches)
        print(f"  Saved {len(oct_patches)} patch pairs.")

    # Save patch pairs dictionary
    pairs_path = output_dir / "patch_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(patch_pairs, f, indent=2)

    print(f"\nDone! {total_patches} total patch pairs saved.")
    print(f"Patch pairs mapping saved to {pairs_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract paired OCT/H&E patches for CycleGAN training.")
    parser.add_argument("--trainA",  type=str, required=True, help="Path to OCT .tif files (trainA)")
    parser.add_argument("--trainB",  type=str, required=True, help="Path to H&E .tif files (trainB)")
    parser.add_argument("--output",  type=str, required=True, help="Path to save output patches")
    args = parser.parse_args()

    print(f"Input OCT dir:  {args.trainA}")
    print(f"Input H&E dir:  {args.trainB}")
    print(f"Output dir:     {args.output}")
    print(f"Patch size:     {PATCH_SIZE}x{PATCH_SIZE}\n")

    process_pairs(
        trainA_dir=args.trainA,
        trainB_dir=args.trainB,
        output_dir=args.output
    )