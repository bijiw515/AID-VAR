#!/usr/bin/env python3
"""
Unified FID evaluation script with consistent preprocessing.
"""

import os
import torch
import numpy as np
import argparse
from pathlib import Path
import subprocess
from typing import Optional, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='Unified FID evaluation with consistent preprocessing')

    parser.add_argument('--generated_samples', type=str, required=True,
                        help='Path to generated samples (NPZ file or directory)')
    parser.add_argument('--reference_stats', type=str,
                        default='/path/to/imagenet_stats.npz',
                        help='Path to ImageNet-1K reference statistics')

    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for feature extraction (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading (default: 8)')

    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for results (default: auto-generated)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for evaluation (default: 256)')
    parser.add_argument('--normalize_method', type=str, default='imagenet',
                        choices=['imagenet', 'zero_one', 'neg_one_one'],
                        help='Normalization method (default: imagenet)')
    parser.add_argument('--resize_method', type=str, default='bicubic',
                        choices=['bicubic', 'bilinear', 'nearest'],
                        help='Resize interpolation method (default: bicubic)')

    return parser.parse_args()

def load_samples(sample_path: str, verbose: bool = False) -> np.ndarray:
    """Load samples from an NPZ file or a directory of PNGs."""
    sample_path = Path(sample_path)

    if sample_path.is_file() and sample_path.suffix == '.npz':
        if verbose:
            print(f"Loading NPZ file: {sample_path}")

        data = np.load(sample_path)

        if 'samples' in data:
            samples = data['samples']
        elif 'arr_0' in data:
            samples = data['arr_0']
        else:
            key = list(data.keys())[0]
            samples = data[key]
            if verbose:
                print(f"Using key '{key}' from NPZ file")

        if verbose:
            print(f"Loaded {len(samples)} samples, shape: {samples.shape}")

        return samples

    elif sample_path.is_dir():
        if verbose:
            print(f"Loading PNG images from directory: {sample_path}")

        import PIL.Image as PImage

        image_files = sorted(list(sample_path.glob('*.png')))
        if not image_files:
            raise ValueError(f"No PNG files found in {sample_path}")

        if verbose:
            print(f"Found {len(image_files)} PNG files")

        samples = []
        for img_path in image_files:
            img = PImage.open(img_path).convert('RGB')
            img_np = np.array(img, dtype=np.float32) / 255.0
            samples.append(img_np)

        samples = np.stack(samples, axis=0)
        if verbose:
            print(f"Loaded {len(samples)} samples, shape: {samples.shape}")

        return samples

    else:
        raise ValueError(f"Invalid sample path: {sample_path}")

def normalize_samples(samples: np.ndarray, method: str = 'imagenet') -> np.ndarray:
    if method == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (samples - mean) / std

    elif method == 'zero_one':
        return np.clip(samples, 0.0, 1.0)

    elif method == 'neg_one_one':
        return np.clip(samples * 2.0 - 1.0, -1.0, 1.0)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

def resize_samples(samples: np.ndarray, target_size: int, method: str = 'bicubic') -> np.ndarray:
    if samples.shape[1] == target_size and samples.shape[2] == target_size:
        return samples

    import torch
    import torch.nn.functional as F

    samples_torch = torch.from_numpy(samples).permute(0, 3, 1, 2)

    if method == 'bicubic':
        mode = 'bicubic'
    elif method == 'bilinear':
        mode = 'bilinear'
    elif method == 'nearest':
        mode = 'nearest'
    else:
        raise ValueError(f"Unknown resize method: {method}")

    resized = F.interpolate(samples_torch, size=(target_size, target_size),
                           mode=mode, align_corners=False if mode != 'nearest' else None)

    return resized.permute(0, 2, 3, 1).numpy()

def compute_fid_stats(samples: np.ndarray, batch_size: int = 50,
                      device: str = 'cuda', verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FID feature statistics using standard Inception-v3."""
    if verbose:
        print(f"Computing FID statistics for {len(samples)} samples...")

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        raise ImportError("Please install torchmetrics: pip install torchmetrics")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    samples_torch = torch.from_numpy(samples).permute(0, 3, 1, 2).to(device)

    features = []
    num_batches = (len(samples) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(samples))
        batch = samples_torch[start_idx:end_idx]

        with torch.no_grad():
            batch_features = fid.inception(batch)
            features.append(batch_features.cpu())

        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_batches} batches")

    all_features = torch.cat(features, dim=0).numpy()

    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features, rowvar=False)

    if verbose:
        print(f"Feature statistics computed: mu.shape={mu.shape}, sigma.shape={sigma.shape}")

    return mu, sigma

def compute_fid_score(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute the FID score from two sets of statistics."""
    from scipy import linalg

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid)

def main():
    args = parse_args()

    print("="*80)
    print("Unified FID Evaluation")
    print("Standardized preprocessing + consistent evaluation")
    print("="*80)
    print(f"Generated samples: {args.generated_samples}")
    print(f"Reference stats: {args.reference_stats}")
    print(f"Image size: {args.image_size}")
    print(f"Normalization: {args.normalize_method}")
    print(f"Resize method: {args.resize_method}")
    print(f"Device: {args.device}")
    print("="*80)

    try:
        print("\nStep 1: Loading generated samples...")
        generated_samples = load_samples(args.generated_samples, verbose=args.verbose)

        print("\nStep 2: Preprocessing samples...")

        if generated_samples.shape[1] != args.image_size or generated_samples.shape[2] != args.image_size:
            print(f"Resizing from {generated_samples.shape[1:3]} to {args.image_size}x{args.image_size}")
            generated_samples = resize_samples(generated_samples, args.image_size, args.resize_method)

        print(f"Normalizing with method: {args.normalize_method}")
        generated_samples = normalize_samples(generated_samples, args.normalize_method)

        print(f"Preprocessed samples shape: {generated_samples.shape}")
        print(f"Value range: [{generated_samples.min():.3f}, {generated_samples.max():.3f}]")

        print("\nStep 3: Computing FID statistics for generated samples...")
        gen_mu, gen_sigma = compute_fid_stats(generated_samples, args.batch_size, args.device, args.verbose)

        print("\nStep 4: Loading reference statistics...")
        if os.path.exists(args.reference_stats):
            ref_stats = np.load(args.reference_stats)
            ref_mu = ref_stats['mu']
            ref_sigma = ref_stats['sigma']
            print(f"Reference stats loaded: mu.shape={ref_mu.shape}, sigma.shape={ref_sigma.shape}")
        else:
            print(f"Warning: Reference stats not found at {args.reference_stats}")
            print("Please provide ImageNet-1K reference statistics or compute them separately.")
            return

        print("\nStep 5: Computing FID score...")
        fid_score = compute_fid_score(gen_mu, gen_sigma, ref_mu, ref_sigma)

        print("\n" + "="*80)
        print("FID EVALUATION RESULTS")
        print("="*80)
        print(f"FID Score: {fid_score:.4f}")
        print(f"Generated samples: {len(generated_samples)}")
        print(f"Reference dataset: ImageNet-1K")
        print(f"Image size: {args.image_size}x{args.image_size}")
        print(f"Preprocessing: {args.normalize_method} normalization + {args.resize_method} resize")
        print("="*80)

        if args.output_file:
            output_path = args.output_file
        else:
            sample_name = Path(args.generated_samples).stem
            output_path = f"fid_results_{sample_name}.txt"

        with open(output_path, 'w') as f:
            f.write(f"FID Evaluation Results\n")
            f.write(f"=====================\n")
            f.write(f"Generated samples: {args.generated_samples}\n")
            f.write(f"Reference stats: {args.reference_stats}\n")
            f.write(f"FID Score: {fid_score:.4f}\n")
            f.write(f"Sample count: {len(generated_samples)}\n")
            f.write(f"Image size: {args.image_size}x{args.image_size}\n")
            f.write(f"Normalization: {args.normalize_method}\n")
            f.write(f"Resize method: {args.resize_method}\n")

        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
