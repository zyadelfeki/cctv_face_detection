import torch
import torch.nn as nn
from torchvision import models
import argparse

class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=128):  # Default, but we'll auto-detect
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)  # This varies!
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.embedding(features)

def infer_embedding_size(state_dict):
    """Auto-detect embedding size from weights"""
    for key in state_dict.keys():
        if 'embedding.3.weight' in key or 'embedding.6.weight' in key:
            return state_dict[key].shape[0]  # Output dim
    return 128  # Default fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='models/face_embedder.pth')
    args = parser.parse_args()
    
    print(f"🔄 Loading {args.model_path}...")
    
    # Load weights
    state_dict = torch.load(args.model_path, map_location='cpu')
    
    # Auto-detect embedding size
    embedding_size = infer_embedding_size(state_dict)
    print(f"📋 Detected embedding size: {embedding_size}")
    
    # Create model with correct architecture
    model = FaceEmbeddingModel(embedding_size=embedding_size)
    
    try:
        model.load_state_dict(state_dict)
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"🎯 Embedding size: {embedding_size}")
        
        # Test forward pass
        print("\n🧪 Testing forward pass...")
        dummy_input = torch.randn(1, 3, 160, 160)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Output shape: {output.shape}")
            print(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}]")
            
        print("\n🎉 All tests passed! Model is ready for inference.")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n🔍 Model architecture details:")
        for key, value in state_dict.items():
            if 'embedding' in key:
                print(f"   {key}: {value.shape}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
