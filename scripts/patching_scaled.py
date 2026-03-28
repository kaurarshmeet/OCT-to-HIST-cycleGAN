"""
patching_scaled.py

Description:
    Extracts non-overlapping 256x256 patches from paired OCT and H&E .tif images.
    Before patching, OCT images are downscaled to match the spatial dimensions of
    their corresponding H&E image, ensuring patches represent the same physical area.
    Images are trimmed to the nearest multiple of 256 before patching.
    Patches are saved as .png to preserve bit depth.

    For perfectly coregistered (test/gold) images:
        - Builds patch_pairs.json mapping each OCT patch to its H&E patch (for stitching)

    For training/silver images:
        - Patches are saved independently per domain
        - No patch count checks or patch_pairs.json are created
        - Augmentation applied per domain independently

Usage:

    # Test / gold (pixel-aligned)
    python patching_scaled.py --inputA /path/to/testA \
                               --inputB /path/to/testB \
                               --output /path/to/output \
                               --train false \
                               --coreg_status gold

    # Training / silver (not perfectly aligned)
    python patching_scaled.py --inputA /path/to/trainA \
                               --inputB /path/to/trainB \
                               --output /path/to/output \
                               --train true \
                               --coreg_status silver \
                               [--augment]
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

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
    match = re.search(r'_(\d+)_', filename)
    if match:
        return match.group(1)
    return None


def scale_oct_to_he(oct_img: np.ndarray, he_img: np.ndarray) -> np.ndarray:
    """
    Downscale OCT image to match H&E image dimensions.
    Handles both grayscale (2D) and RGB (3D) OCT images.
    Uses LANCZOS resampling for high quality downscaling.
    Preserves original dtype after resampling.
    """
    he_h, he_w = he_img.shape[:2]
    oct_h, oct_w = oct_img.shape[:2]

    # Check if scaling is actually needed
    if oct_h == he_h and oct_w == he_w:
        print(f"  OCT and H&E already same size — no scaling needed.")
        return oct_img

    scale_h = he_h / oct_h
    scale_w = he_w / oct_w
    print(f"  Scaling OCT: ({oct_h}x{oct_w}) -> ({he_h}x{he_w}) "
          f"[scale_h={scale_h:.3f}, scale_w={scale_w:.3f}]")

    original_dtype = oct_img.dtype

    # PIL requires uint8 for LANCZOS resampling
    # For 16-bit, normalize to uint8, resample, then restore scale
    if oct_img.dtype == np.uint16 or str(oct_img.dtype) == '>u2':
        oct_max = oct_img.max()
        oct_normalized = (oct_img.astype(np.float32) / (oct_max + 1e-8) * 255).astype(np.uint8)
        pil_img = Image.fromarray(oct_normalized)
        pil_resized = pil_img.resize((he_w, he_h), Image.LANCZOS)
        resized = (np.array(pil_resized).astype(np.float32) / 255.0 * oct_max).astype(np.uint16)
    else:
        pil_img = Image.fromarray(oct_img)
        pil_resized = pil_img.resize((he_w, he_h), Image.LANCZOS)
        resized = np.array(pil_resized)

    print(f"  OCT dtype after scaling: {resized.dtype}")
    return resized


def save_patch(patch: np.ndarray, path: Path):
    """
    Save a patch as PNG, preserving bit depth.
    Handles 8-bit (uint8) and 16-bit (uint16) arrays correctly.
    """
    if patch.dtype == np.uint16 or str(patch.dtype) == '>u2':
        patch = patch.astype(np.uint16)
        pil_img = Image.fromarray(patch)
    elif patch.dtype == np.uint8:
        pil_img = Image.fromarray(patch)
    else:
        # Fallback: normalize to uint8
        patch = ((patch - patch.min()) / (patch.max() - patch.min() + 1e-8) * 255).astype(np.uint8)
        pil_img = Image.fromarray(patch)
    pil_img.save(path, format="PNG")


def save_patches(patches, out_dir, prefix, sample_id, augment=False):
    """
    Save patches to output directory.
    If augment=True, also save geometric augmentations independently.
    """
    for patch, row, col in patches:
        patch_name = f"{prefix}_{sample_id}_patch_{row}_{col}.png"
        save_patch(patch, out_dir / patch_name)

        if augment:
            for aug_label, aug_fn in AUGMENTATIONS:
                aug_name = f"{prefix}_{sample_id}_patch_{row}_{col}_{aug_label}.png"
                save_patch(aug_fn(patch), out_dir / aug_name)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_pairs(inputA_dir, inputB_dir, output_dir, train=False, coreg_status="gold", augment=False):
    """
    For each paired OCT/H&E image:
        - Scale OCT to match H&E dimensions
        - Extract patches from both
        - Verify patch counts match (test/gold only)
        - Save patches independently per domain
        - Build patch_pairs.json (test/gold only)
    """
    inputA_dir = Path(inputA_dir)
    inputB_dir = Path(inputB_dir)
    output_dir = Path(output_dir)

    folder_suffix = "train" if train else "test"
    out_A = output_dir / f"{folder_suffix}A"
    out_B = output_dir / f"{folder_suffix}B"
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

        he_candidates = list(inputB_dir.glob(f"*_{sample_id}_he.tif"))
        if not he_candidates:
            print(f"WARNING: No H&E file found for {oct_file.name}, skipping.")
            continue
        he_file = he_candidates[0]

        print(f"\nProcessing pair: {coreg_status}_{sample_id}")
        print(f"  OCT: {oct_file.name}")
        print(f"  H&E: {he_file.name}")

        oct_img = np.array(Image.open(oct_file))
        he_img  = np.array(Image.open(he_file))

        print(f"  OCT shape (original): {oct_img.shape}, dtype: {oct_img.dtype}")
        print(f"  H&E shape:            {he_img.shape}, dtype: {he_img.dtype}")

        # Scale OCT to match H&E dimensions
        oct_img = scale_oct_to_he(oct_img, he_img)
        print(f"  OCT shape (scaled):   {oct_img.shape}")

        oct_patches = trim_and_patch(oct_img, PATCH_SIZE)
        he_patches  = trim_and_patch(he_img,  PATCH_SIZE)

        # For test/gold: enforce matching patch counts
        if not train:
            if len(oct_patches) != len(he_patches):
                print(f"  WARNING: Patch count mismatch after scaling! "
                      f"OCT={len(oct_patches)}, H&E={len(he_patches)}. Skipping.")
                continue

        print(f"  Original patches: {len(oct_patches)}")

        # Build patch_pairs.json only for test/gold
        if not train:
            for (_, row, col) in oct_patches:
                patch_name = f"{coreg_status}_{sample_id}_patch_{row}_{col}.png"
                patch_pairs[str(Path(f"{folder_suffix}A") / patch_name)] = \
                            str(Path(f"{folder_suffix}B") / patch_name)

        # Save patches independently per domain
        save_patches(oct_patches, out_A, coreg_status, sample_id, augment)
        save_patches(he_patches,  out_B, coreg_status, sample_id, augment)
        total_original += len(oct_patches)

        if augment:
            print(f"  Augmented patches per domain: {len(oct_patches) * len(AUGMENTATIONS)}")

    # Save patch_pairs.json only for test/gold
    if not train and patch_pairs:
        pairs_path = output_dir / "patch_pairs.json"
        with open(pairs_path, "w") as f:
            json.dump(patch_pairs, f, indent=2)
        print(f"\npatch_pairs.json saved to {pairs_path}")

    n_aug = len(AUGMENTATIONS)
    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Original patch count:           {total_original}")
    if augment:
        print(f"  Augmented patches (per domain): {total_original * n_aug}")
        print(f"  Total patches per domain:       {total_original * (1 + n_aug)}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract scaled OCT/H&E patches for CycleGAN.")
    parser.add_argument("--inputA",       type=str, required=True, help="Path to OCT .tif files")
    parser.add_argument("--inputB",       type=str, required=True, help="Path to H&E .tif files")
    parser.add_argument("--output",       type=str, required=True, help="Path to save output patches")
    parser.add_argument("--augment",      action="store_true",     help="Apply geometric augmentation per domain")
    parser.add_argument("--train",        type=str, required=True, help="true if training (silver), false if test/gold")
    parser.add_argument("--coreg_status", type=str, required=True, help="gold or silver")
    args = parser.parse_args()

    train_flag = args.train.lower() == "true"

    print(f"Input OCT dir:  {args.inputA}")
    print(f"Input H&E dir:  {args.inputB}")
    print(f"Output dir:     {args.output}")
    print(f"Patch size:     {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Augmentation:   {'ON' if args.augment else 'OFF'}")
    print(f"Training data:  {'YES (silver)' if train_flag else 'NO (gold)'}")
    print(f"Coreg status:   {args.coreg_status}\n")

    process_pairs(
        inputA_dir=args.inputA,
        inputB_dir=args.inputB,
        output_dir=args.output,
        train=train_flag,
        coreg_status=args.coreg_status,
        augment=args.augment
    )