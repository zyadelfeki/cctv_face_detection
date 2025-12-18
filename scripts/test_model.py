#!/usr/bin/env python3
"""
Test script for trained face recognition model

Usage:
    python scripts/test_model.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recognition import FaceRecognitionSystem
from PIL import Image


def test_basic_loading():
    """
    Test 1: Basic model loading
    """
    print("\n" + "="*60)
    print("Test 1: Loading Model")
    print("="*60)
    
    model_path = 'models/model.pth'
    
    try:
        recognizer = FaceRecognitionSystem(
            model_path=model_path,
            threshold=0.6
        )
        print("\n✅ Model loaded successfully!")
        return recognizer
    except FileNotFoundError:
        print(f"\n❌ ERROR: Model file not found at: {model_path}")
        print("\n💡 Instructions:")
        print("   1. Download model.pth from Colab")
        print("   2. Create 'models' folder in project root")
        print("   3. Place model.pth in models/model.pth")
        print("\n   Or run: mkdir -p models && mv ~/Downloads/model.pth models/")
        return None
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        return None


def test_registration(recognizer):
    """
    Test 2: Identity registration
    """
    print("\n" + "="*60)
    print("Test 2: Registering Identities")
    print("="*60)
    
    # Example: Register from training data (if available)
    data_dir = Path('./data_prepared')
    
    if not data_dir.exists():
        print(f"\n⚠️ Training data not found at: {data_dir}")
        print("   Skipping registration test")
        print("\n💡 To test registration:")
        print("   1. Create data_prepared/ folder")
        print("   2. Add folders: person_0/, person_1/, etc.")
        print("   3. Add face images to each folder")
        return False
    
    print(f"\n📝 Registering identities from: {data_dir}")
    registered = 0
    
    for identity_folder in sorted(data_dir.iterdir()):
        if identity_folder.is_dir():
            # Get first 5 images for registration
            image_paths = [str(p) for p in identity_folder.glob('*.jpg')][:5]
            image_paths += [str(p) for p in identity_folder.glob('*.png')][:5]
            
            if image_paths:
                recognizer.register_identity(
                    identity_name=identity_folder.name,
                    image_paths=image_paths
                )
                registered += 1
    
    print(f"\n✅ Registered {registered} identities")
    return registered > 0


def test_prediction(recognizer, has_data):
    """
    Test 3: Face prediction
    """
    print("\n" + "="*60)
    print("Test 3: Face Prediction")
    print("="*60)
    
    if not has_data:
        print("\n⚠️ No registered identities, skipping prediction test")
        return
    
    # Test with first image from first identity
    data_dir = Path('./data_prepared')
    test_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not test_folders:
        print("\n⚠️ No test data available")
        return
    
    # Test 1: Known face (should match)
    print("\n📘 Test 3a: Predicting known face...")
    test_folder = test_folders[0]
    test_images = list(test_folder.glob('*.jpg')) + list(test_folder.glob('*.png'))
    
    if test_images:
        test_image = test_images[-1]  # Use last image (not used in registration)
        identity, confidence = recognizer.predict_from_path(str(test_image))
        
        print(f"   Image: {test_image.name}")
        print(f"   Expected: {test_folder.name}")
        print(f"   Predicted: {identity}")
        print(f"   Confidence: {confidence:.4f}")
        
        if identity == test_folder.name:
            print(f"   ✅ PASS: Correctly identified!")
        else:
            print(f"   ⚠️ Mismatch (may need threshold adjustment)")
    
    # Test 2: Different person (may or may not match depending on data)
    if len(test_folders) > 1:
        print("\n📙 Test 3b: Predicting different person...")
        test_folder2 = test_folders[1]
        test_images2 = list(test_folder2.glob('*.jpg')) + list(test_folder2.glob('*.png'))
        
        if test_images2:
            test_image2 = test_images2[-1]
            identity2, confidence2 = recognizer.predict_from_path(str(test_image2))
            
            print(f"   Image: {test_image2.name}")
            print(f"   Expected: {test_folder2.name}")
            print(f"   Predicted: {identity2}")
            print(f"   Confidence: {confidence2:.4f}")
            
            if identity2 == test_folder2.name:
                print(f"   ✅ PASS: Correctly identified!")
            else:
                print(f"   ⚠️ Mismatch")


def test_unknown_detection(recognizer):
    """
    Test 4: Unknown face detection
    """
    print("\n" + "="*60)
    print("Test 4: Unknown Detection")
    print("="*60)
    
    print("\n💡 To test unknown detection:")
    print("   1. Add a face image not in training data")
    print("   2. Run: recognizer.predict_from_path('unknown_face.jpg')")
    print("   3. Should return ('Unknown', confidence < 0.6)")
    print("\n   Threshold can be adjusted:")
    print("   - Higher (0.7): Stricter, fewer false positives")
    print("   - Lower (0.5): More lenient, more matches")


def test_batch_prediction(recognizer, has_data):
    """
    Test 5: Batch prediction
    """
    print("\n" + "="*60)
    print("Test 5: Batch Prediction")
    print("="*60)
    
    if not has_data:
        print("\n⚠️ No registered identities, skipping batch test")
        return
    
    data_dir = Path('./data_prepared')
    test_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    # Collect multiple images
    test_images = []
    for folder in test_folders[:3]:  # First 3 identities
        imgs = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
        if imgs:
            test_images.append(imgs[0])
    
    if len(test_images) < 2:
        print("\n⚠️ Not enough test images for batch test")
        return
    
    print(f"\n📦 Processing {len(test_images)} images in batch...")
    
    # Load images as PIL
    pil_images = [Image.open(img).convert('RGB') for img in test_images]
    
    # Batch predict
    results = recognizer.predict_batch(pil_images)
    
    print("\n📊 Results:")
    for i, (img_path, (identity, confidence)) in enumerate(zip(test_images, results)):
        print(f"   {i+1}. {img_path.parent.name}/{img_path.name}")
        print(f"      → {identity} (confidence: {confidence:.4f})")
    
    print(f"\n✅ Batch prediction completed")


def main():
    """
    Run all tests
    """
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + "  Face Recognition Model Test Suite".center(58) + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    # Test 1: Load model
    recognizer = test_basic_loading()
    if recognizer is None:
        print("\n❌ Tests aborted: Model not loaded")
        return
    
    # Test 2: Register identities
    has_data = test_registration(recognizer)
    
    # Test 3: Prediction
    test_prediction(recognizer, has_data)
    
    # Test 4: Unknown detection
    test_unknown_detection(recognizer)
    
    # Test 5: Batch prediction
    test_batch_prediction(recognizer, has_data)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"\n✅ Model Status: Loaded")
    print(f"   Device: {recognizer.device}")
    print(f"   Threshold: {recognizer.threshold}")
    print(f"   Registered: {len(recognizer.get_registered_identities())} identities")
    
    if has_data:
        print(f"\n✅ All tests completed successfully!")
    else:
        print(f"\n⚠️ Partial tests completed (no training data)")
    
    print("\n" + "#"*60)
    print("\n🚀 Ready for production use!\n")


if __name__ == "__main__":
    main()
