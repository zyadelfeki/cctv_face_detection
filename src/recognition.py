"""
Face Recognition Module for CCTV System
Uses trained triplet loss model for identity recognition with unknown detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np


class FaceEmbeddingModel(nn.Module):
    """
    Face Recognition Model with Triplet Loss
    - Pretrained ResNet50 backbone
    - Custom embedding head (128D)
    - L2 normalized embeddings
    """
    
    def __init__(self, embedding_size=128):
        super().__init__()
        
        # Load ResNet50 (weights will be loaded from checkpoint)
        resnet = models.resnet50(pretrained=False)
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)
        )
        
        self.embedding_size = embedding_size
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Get embeddings
        embeddings = self.embedding(features)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


def _load_state_dict_from_checkpoint(path: str, map_location='cpu'):
    """Load a checkpoint (full or weights-only) and return (state_dict, embedding_size, identities).

    - If checkpoint is a dict with 'model_state_dict' it extracts it and reads metadata.
    - If it's already a state_dict (weights-only), it returns it as-is and attempts to infer
      embedding size by inspecting any embedding layer weight shapes.
    """
    data = torch.load(path, map_location=map_location)

    if isinstance(data, dict) and 'model_state_dict' in data:
        state_dict = data['model_state_dict']
        embedding_size = data.get('embedding_size', None)
        identities = data.get('identities', [])
    else:
        state_dict = data
        embedding_size = None
        identities = []

    # Try to infer embedding size if not present
    if embedding_size is None:
        inferred = 128
        for k, v in state_dict.items():
            if 'embedding' in k and k.endswith('weight'):
                inferred = v.shape[0]
                break
        embedding_size = inferred

    return state_dict, embedding_size, identities


class FaceRecognitionSystem:
    """
    Production-ready face recognition system with unknown detection
    
    Features:
    - Load trained model from checkpoint
    - Register known identities with multiple images
    - Predict identity with confidence score
    - Unknown detection with configurable threshold
    """
    
    def __init__(self, model_path: str, threshold: float = 0.6, device: Optional[str] = None):
        """
        Initialize face recognition system
        
        Args:
            model_path: Path to trained model.pth file
            threshold: Similarity threshold for unknown detection (0.0-1.0)
                      Higher = stricter (fewer false positives)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load checkpoint / weights (supports full checkpoint or weights-only)
        print(f"📦 Loading model from: {model_path}")
        state_dict, embedding_size, identities = _load_state_dict_from_checkpoint(model_path, map_location=self.device)

        # Initialize model
        self.model = FaceEmbeddingModel(
            embedding_size=embedding_size
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load metadata
        self.identities = identities
        self.threshold = threshold
        self.embeddings_db = {}
        
        # Image preprocessing transform (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Print info
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   Embedding size: {self.model.embedding_size}")
        print(f"   Training identities: {len(self.identities)}")
        print(f"   Threshold: {self.threshold}")
        print(f"   Registered identities: {len(self.embeddings_db)}")
    
    def register_identity(self, identity_name: str, image_paths: List[str]):
        """
        Register a known identity with multiple face images
        
        Args:
            identity_name: Name/ID of the person (e.g., 'criminal_1')
            image_paths: List of paths to face images of this person
        
        Example:
            recognizer.register_identity('John_Doe', [
                'faces/john_1.jpg',
                'faces/john_2.jpg',
                'faces/john_3.jpg'
            ])
        """
        embeddings = []
        valid_images = 0
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    emb = self.model(img_tensor)
                    embeddings.append(emb.cpu())
                
                valid_images += 1
            except Exception as e:
                print(f"   ⚠️ Failed to process {img_path}: {e}")
                continue
        
        if valid_images == 0:
            print(f"   ❌ No valid images for {identity_name}")
            return
        
        # Average embedding (more robust than single image)
        avg_embedding = torch.cat(embeddings).mean(dim=0)
        self.embeddings_db[identity_name] = avg_embedding
        
        print(f"   ✅ Registered: {identity_name} ({valid_images} images)")
    
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Predict identity from PIL Image
        
        Args:
            image: PIL Image of a face (cropped)
        
        Returns:
            (identity_name, confidence): 
                - identity_name: Matched person or 'Unknown'
                - confidence: Cosine similarity score (0.0-1.0)
        
        Example:
            identity, conf = recognizer.predict(face_img)
            if identity != 'Unknown':
                print(f"Detected: {identity} (confidence: {conf:.2f})")
        """
        # Process image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            query_emb = self.model(img_tensor).cpu()
        
        # Compare with all registered identities
        best_match = None
        best_similarity = -1
        
        for identity, db_emb in self.embeddings_db.items():
            # Cosine similarity (both embeddings are L2 normalized)
            similarity = F.cosine_similarity(
                query_emb, 
                db_emb.unsqueeze(0)
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = identity
        
        # Check threshold for unknown detection
        if best_similarity >= self.threshold:
            return best_match, best_similarity
        else:
            return 'Unknown', best_similarity
    
    def predict_from_path(self, image_path: str) -> Tuple[str, float]:
        """
        Predict identity from image file path
        
        Args:
            image_path: Path to face image file
        
        Returns:
            (identity_name, confidence)
        """
        image = Image.open(image_path).convert('RGB')
        return self.predict(image)
    
    def predict_batch(self, images: List[Image.Image]) -> List[Tuple[str, float]]:
        """
        Predict identities for multiple images in batch
        
        Args:
            images: List of PIL Images
        
        Returns:
            List of (identity_name, confidence) tuples
        """
        results = []
        
        # Stack images into batch
        img_tensors = torch.stack([
            self.transform(img) for img in images
        ]).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            query_embs = self.model(img_tensors).cpu()
        
        # Compare each query with database
        for query_emb in query_embs:
            best_match = None
            best_similarity = -1
            
            for identity, db_emb in self.embeddings_db.items():
                similarity = F.cosine_similarity(
                    query_emb.unsqueeze(0), 
                    db_emb.unsqueeze(0)
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = identity
            
            if best_similarity >= self.threshold:
                results.append((best_match, best_similarity))
            else:
                results.append(('Unknown', best_similarity))
        
        return results
    
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Get raw embedding vector for an image
        
        Args:
            image: PIL Image
        
        Returns:
            128D embedding vector as numpy array
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor).cpu().numpy()[0]
        
        return embedding
    
    def set_threshold(self, threshold: float):
        """Update similarity threshold"""
        self.threshold = threshold
        print(f"✅ Threshold updated to: {threshold}")
    
    def get_registered_identities(self) -> List[str]:
        """Get list of registered identity names"""
        return list(self.embeddings_db.keys())
    
    def clear_database(self):
        """Clear all registered identities"""
        self.embeddings_db.clear()
        print("✅ Database cleared")


def load_recognizer(model_path: str = 'models/model.pth', threshold: float = 0.6) -> FaceRecognitionSystem:
    """
    Convenience function to load face recognition system
    
    Args:
        model_path: Path to model checkpoint
        threshold: Similarity threshold
    
    Returns:
        Initialized FaceRecognitionSystem
    """
    return FaceRecognitionSystem(model_path, threshold)
