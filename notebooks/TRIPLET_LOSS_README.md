# 🔐 Face Recognition with Triplet Loss & Unknown Detection

## 📚 Overview

Complete implementation of a **Closed-Set Face Recognition System** with **Unknown Detection** capability using:
- **Transfer Learning** with ResNet50
- **Triplet Loss** for metric learning
- **Cosine Similarity** for verification
- **Threshold-based Unknown Detection**

---

## 🎯 Problem Statement

### Requirements
- **Known Identities:** 10 criminals (persons)
- **Images per Identity:** 20 face images
- **Unknown Detection:** Classify faces not in database as "Unknown"

### Architecture
- **Backbone:** ResNet50 (pretrained on ImageNet)
- **Embedding Size:** 128D
- **Loss Function:** Triplet Loss with margin=0.5
- **Distance Metric:** Cosine Similarity
- **Unknown Threshold:** 0.6 (adjustable)

---

## 💾 Files Generated

```
notebooks/
├── triplet_loss_face_recognition.ipynb  # Main training notebook
├── TRIPLET_LOSS_README.md              # This guide
└── checkpoints/
    ├── model.pth                       # Best model weights
    └── training_history.png            # Training curves
```

---

## 🛠️ Setup Instructions

### 1. Data Preparation

Organize your dataset in the following structure:

```
data/
├── person_0/
│   ├── img_00.jpg
│   ├── img_01.jpg
│   │   ...
│   └── img_19.jpg
├── person_1/
│   ├── img_00.jpg
│   │   ...
│   └── img_19.jpg
...
└── person_9/
    ├── img_00.jpg
    │   ...
    └── img_19.jpg
```

**Requirements:**
- 10 folders (one per person)
- Each folder contains exactly 20 images
- Supported formats: `.jpg`, `.png`, `.jpeg`

### 2. Install Dependencies

```bash
pip install torch torchvision pillow matplotlib tqdm scikit-learn
```

### 3. Run the Notebook

Open `triplet_loss_face_recognition.ipynb` and run all cells sequentially.

---

## 📊 Training Process

### 1. Dataset Loading
```python
TripletFaceDataset(
    data_dir='./data',
    transform=train_transform
)
```
- Automatically loads all 10 identities
- Creates triplets: (anchor, positive, negative)
- Applies data augmentation

### 2. Model Architecture
```python
FaceEmbeddingModel(
    embedding_size=128,
    pretrained=True
)
```
- ResNet50 backbone (frozen early layers)
- Custom embedding head (2048 → 512 → 128)
- L2-normalized embeddings

### 3. Triplet Loss
```python
TripletLoss(margin=0.5)
```
- Formula: `L = max(0, ||a - p||² - ||a - n||² + margin)`
- Pulls positive pairs together
- Pushes negative pairs apart

### 4. Training Loop
```python
for epoch in range(50):
    train_loss = train_epoch(...)
    val_loss, val_acc = validate(...)
    # Save best model
```
- 50 epochs
- Batch size: 16
- Adam optimizer (lr=0.0001)
- StepLR scheduler (decay every 10 epochs)

---

## 🔍 Inference & Unknown Detection

### 1. Load Trained Model
```python
checkpoint = torch.load('checkpoints/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 2. Register Known Identities
```python
recognition_system = FaceRecognitionSystem(
    model=model,
    identities=checkpoint['identities'],
    threshold=0.6
)

# Register each identity
for identity in identities:
    recognition_system.register_identity(
        identity_name=identity,
        image_paths=[...],  # All 20 images
        transform=val_transform
    )
```

### 3. Predict on New Image
```python
identity, confidence = recognition_system.predict(
    image_path='test_image.jpg',
    transform=val_transform
)

if identity == 'Unknown':
    print(f"Unknown person (confidence: {confidence:.4f})")
else:
    print(f"Identity: {identity} (confidence: {confidence:.4f})")
```

---

## ⚙️ Hyperparameters

### Model
| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_size` | 128 | Dimensionality of face embeddings |
| `backbone` | ResNet50 | Pretrained CNN backbone |
| `frozen_layers` | First 30 | Only fine-tune last 20 layers |

### Training
| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 50 | Number of training epochs |
| `batch_size` | 16 | Samples per batch |
| `learning_rate` | 0.0001 | Initial learning rate |
| `margin` | 0.5 | Triplet loss margin |
| `optimizer` | Adam | Optimization algorithm |

### Inference
| Parameter | Value | Description |
|-----------|-------|-------------|
| `threshold` | 0.6 | Cosine similarity threshold |
| `distance_metric` | Cosine | Similarity measure |
| `unknown_class` | Yes | Enable unknown detection |

---

## 📊 Performance Metrics

### Training Metrics
- **Loss:** Triplet Loss (lower is better)
- **Validation Accuracy:** Percentage of correct verifications

### Inference Metrics
- **True Positive Rate (TPR):** Known faces correctly identified
- **False Positive Rate (FPR):** Unknown faces incorrectly accepted
- **Threshold:** Adjust to balance TPR/FPR trade-off

### Threshold Selection
```python
# Lower threshold (0.4-0.5): More strict, fewer false positives
# Higher threshold (0.6-0.7): More lenient, fewer false negatives
# Recommended: 0.6 for balanced performance
```

---

## 🔧 Troubleshooting

### Issue: Low Training Accuracy
**Solutions:**
- Increase training epochs (50 → 100)
- Reduce learning rate (0.0001 → 0.00005)
- Increase margin (0.5 → 0.7)
- Add more data augmentation

### Issue: Too Many "Unknown" Predictions
**Solutions:**
- Lower threshold (0.6 → 0.5 or 0.4)
- Register more images per identity
- Retrain with better quality images

### Issue: False Positives (Wrong Identity)
**Solutions:**
- Raise threshold (0.6 → 0.7 or 0.8)
- Increase training epochs
- Use harder triplet mining

### Issue: GPU Out of Memory
**Solutions:**
- Reduce batch size (16 → 8)
- Use smaller backbone (ResNet50 → ResNet18)
- Reduce embedding size (128 → 64)

---

## 🚀 Deployment

### 1. Export Model for Production
```python
# Save as TorchScript
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model_traced.pt')

# Save as ONNX
torch.onnx.export(model, example_input, 'model.onnx')
```

### 2. Integration Example
```python
import torch
from PIL import Image

# Load model
model = torch.jit.load('model_traced.pt')
model.eval()

# Load embeddings database
embeddings_db = torch.load('embeddings_db.pt')

# Predict
def recognize_face(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(img)
    
    # Compare with database
    best_match, best_score = find_closest(embedding, embeddings_db)
    
    if best_score >= 0.6:
        return best_match
    else:
        return "Unknown"
```

---

## 📚 Key Concepts

### Triplet Loss
- **Goal:** Learn embeddings where same-person distances are small
- **Components:**
  - **Anchor:** Reference image
  - **Positive:** Same person as anchor
  - **Negative:** Different person
- **Loss Formula:** `max(0, d(a,p) - d(a,n) + margin)`

### Transfer Learning
- **Pretrained Weights:** ResNet50 trained on ImageNet
- **Fine-tuning:** Only train last 20 layers
- **Benefits:** Faster convergence, better generalization

### Unknown Detection
- **Method:** Threshold-based classification
- **Metric:** Cosine similarity
- **Decision Rule:** If `similarity < threshold` → Unknown

---

## ✅ Expected Results

### After Training
- **Training Loss:** ≈ 0.1 - 0.3
- **Validation Accuracy:** > 85%
- **Best Model:** Saved at `checkpoints/model.pth`

### Inference Performance
- **Known Faces:** 90-95% correctly identified
- **Unknown Faces:** 85-90% correctly rejected
- **Average Inference Time:** < 50ms per image (GPU)

---

## 📝 Model Output Format

### Saved Checkpoint (`model.pth`)
```python
{
    'epoch': 42,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'val_accuracy': 0.923,
    'embedding_size': 128,
    'identities': ['person_0', 'person_1', ..., 'person_9']
}
```

### Prediction Output
```python
identity, confidence = recognition_system.predict('test.jpg')
# Output: ('person_3', 0.847) or ('Unknown', 0.523)
```

---

## 🎓 Next Steps

1. **Improve Dataset Quality:**
   - Add more diverse images per person
   - Include various lighting conditions
   - Add different poses and expressions

2. **Experiment with Hyperparameters:**
   - Try different margins (0.3, 0.5, 0.7)
   - Test various thresholds (0.5, 0.6, 0.7)
   - Adjust learning rate schedule

3. **Advanced Techniques:**
   - Hard negative mining
   - Online triplet mining
   - Center Loss + Triplet Loss
   - ArcFace / CosFace losses

4. **Production Deployment:**
   - Convert to ONNX for cross-platform
   - Optimize with TensorRT
   - Deploy with FastAPI/Flask
   - Add face detection pipeline (MTCNN/RetinaFace)

---

## 💬 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs and validation curves
3. Adjust hyperparameters based on your data

---

**Built with PyTorch ❤️**

**Ready to train your face recognition system!** 🚀