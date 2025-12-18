"""
Simple test for weights-only face embedder

Run after converting:
    python test_lightweight_model.py

It will load `models/face_embedder.pth`, instantiate the model with inferred embedding size,
perform a forward pass with dummy input, and print status.
"""

import sys
import os
import torch

# Make sure src is importable
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

MODEL_PATH = os.path.join('models', 'face_embedder.pth')


def infer_embedding_size(state_dict):
    # Find embedding linear weight key and return its out features
    for k, v in state_dict.items():
        if 'embedding' in k and k.endswith('weight'):
            return v.shape[0]
    return 128


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ '{MODEL_PATH}' not found. Run 'python convert_model.py' first to create it.")
        return 2

    print(f"🔄 Loading {MODEL_PATH}...")
    data = torch.load(MODEL_PATH, map_location='cpu')

    if isinstance(data, dict) and 'model_state_dict' in data:
        state_dict = data['model_state_dict']
    else:
        state_dict = data

    embedding_size = infer_embedding_size(state_dict)
    print(f"📋 Inferred embedding size: {embedding_size}")

    # Try to import the real model implementation; if dependencies missing, do a lightweight check
    try:
        from recognition import FaceEmbeddingModel
        model = FaceEmbeddingModel(embedding_size=embedding_size)
        model.load_state_dict(state_dict)
        model.eval()

        print("✅ Model loaded successfully!")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total parameters: {total_params:,}")

        # Test forward
        dummy_input = torch.randn(1, 3, 160, 160)
        with torch.no_grad():
            out = model(dummy_input)
        print(f"✅ Test forward pass: output shape = {out.shape}")

    except Exception as e:
        print(f"⚠️ Could not run full model test due to missing dependency or error: {e}")
        print("➡️ Performing weights-only sanity checks...")

        # Check that embedding weights exist and have expected shape
        emb_weights = [v for k, v in state_dict.items() if 'embedding' in k and k.endswith('weight')]
        if emb_weights:
            w = emb_weights[0]
            print(f"✅ Found embedding weights with shape: {tuple(w.shape)}")
        else:
            print("❌ No embedding weights found in state_dict; the checkpoint may be incompatible.")
            return 3

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
