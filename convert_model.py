"""
Lightweight model conversion tool

Usage:
    python convert_model.py --input models/model.pth --output models/face_embedder.pth

This script extracts the state_dict (model weights) from a full PyTorch checkpoint
and saves only the weights to reduce size for inference.
"""

import argparse
import os
import torch


def bytes_to_mb(n):
    return n / (1024 * 1024)


def main(input_path: str, output_path: str, force: bool = False):
    print("🔄 Converting checkpoint to lightweight weights-only file...")

    if not os.path.exists(input_path):
        print(f"❌ Input checkpoint not found: {input_path}")
        return 2

    # Load checkpoint (map to cpu to avoid GPU requirements)
    print(f"📂 Loading {input_path}...")
    data = torch.load(input_path, map_location='cpu')

    # Determine what we have and extract weights
    if isinstance(data, dict) and 'model_state_dict' in data:
        weights = data['model_state_dict']
        print("📋 Detected full checkpoint (contains metadata). Extracting 'model_state_dict'...")
    else:
        weights = data
        print("📋 Detected weights-only checkpoint (saving as-is)...")

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Avoid overwriting unless forced
    if os.path.exists(output_path) and not force:
        print(f"⚠️ Output already exists: {output_path} (use --force to overwrite)")
        return 3

    # Save weights-only file
    torch.save(weights, output_path)

    old_size = bytes_to_mb(os.path.getsize(input_path))
    new_size = bytes_to_mb(os.path.getsize(output_path))

    print("\n✅ Conversion complete!")
    print(f"📦 Old {os.path.basename(input_path)}:  {old_size:.2f} MB")
    print(f"📦 New {os.path.basename(output_path)}: {new_size:.2f} MB")
    print(f"💾 Saved: {old_size - new_size:.2f} MB ({((old_size - new_size) / old_size * 100):.1f}%)")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert full checkpoint to weights-only file')
    parser.add_argument('--input', type=str, default='models/model.pth', help='Path to full checkpoint')
    parser.add_argument('--output', type=str, default='models/face_embedder.pth', help='Path to save weights-only file')
    parser.add_argument('--force', action='store_true', help='Overwrite output if exists')
    args = parser.parse_args()

    raise SystemExit(main(args.input, args.output, args.force))
