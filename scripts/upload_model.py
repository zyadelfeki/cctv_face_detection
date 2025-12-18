#!/usr/bin/env python3
"""
Script to help upload model.pth to the correct location

This script copies model.pth from Downloads to models/ directory
and verifies it's valid.

Usage:
    python scripts/upload_model.py
    python scripts/upload_model.py --source /path/to/model.pth
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def find_model_in_downloads():
    """Try to find model.pth in common download locations"""
    home = Path.home()
    
    # Common download locations
    search_paths = [
        home / 'Downloads' / 'model.pth',
        home / 'downloads' / 'model.pth',
        Path.cwd() / 'model.pth',
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def verify_model(model_path):
    """Verify the model file is valid PyTorch checkpoint"""
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'embedding_size']
        for key in required_keys:
            if key not in checkpoint:
                print(f"  ⚠️ Warning: Missing key '{key}' in checkpoint")
        
        print(f"\n✅ Model file is valid!")
        print(f"   Embedding size: {checkpoint.get('embedding_size', 'Unknown')}")
        print(f"   Identities: {len(checkpoint.get('identities', []))}")
        print(f"   Val accuracy: {checkpoint.get('val_accuracy', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error: Model file appears invalid")
        print(f"   {e}")
        return False


def copy_model(source_path, dest_dir):
    """Copy model to destination directory"""
    # Create destination directory
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_path = dest_dir / 'model.pth'
    
    # Check if destination already exists
    if dest_path.exists():
        response = input(f"\n⚠️  {dest_path} already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("\n❌ Aborted")
            return False
    
    # Copy file
    print(f"\n💾 Copying model...")
    print(f"   From: {source_path}")
    print(f"   To:   {dest_path}")
    
    try:
        shutil.copy2(source_path, dest_path)
        print(f"\n✅ Model copied successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error copying file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Upload model.pth to project models/ directory'
    )
    parser.add_argument(
        '--source', 
        type=str, 
        help='Path to model.pth file (auto-detects if not specified)'
    )
    parser.add_argument(
        '--dest-dir',
        type=str,
        default='models',
        help='Destination directory (default: models)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify the model, do not copy'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Model Upload Script")
    print("="*60)
    
    # Find source model
    if args.source:
        source_path = Path(args.source)
    else:
        print("\n🔍 Searching for model.pth in common locations...")
        source_path = find_model_in_downloads()
    
    # Check if source exists
    if not source_path or not source_path.exists():
        print(f"\n❌ Error: Model file not found!")
        print("\n💡 Instructions:")
        print("   1. Download model.pth from Google Colab")
        print("   2. Save it to your Downloads folder")
        print("   3. Run this script again")
        print("\n   Or specify path manually:")
        print("   python scripts/upload_model.py --source /path/to/model.pth")
        return 1
    
    print(f"\n✅ Found model: {source_path}")
    print(f"   Size: {source_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Verify model
    print("\n🔍 Verifying model file...")
    if not verify_model(source_path):
        return 1
    
    # If verify-only mode, stop here
    if args.verify_only:
        print("\n✅ Verification complete (--verify-only mode)\n")
        return 0
    
    # Copy model
    if not copy_model(source_path, args.dest_dir):
        return 1
    
    # Final instructions
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\n🚀 Next steps:\n")
    print("   1. Test the model:")
    print("      python scripts/test_model.py\n")
    print("   2. Run CCTV recognition:")
    print("      python scripts/integrate_cctv.py --source 0\n")
    print("   3. Process a video:")
    print("      python scripts/integrate_cctv.py --source video.mp4\n")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
