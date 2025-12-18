#!/usr/bin/env python3
"""
CCTV Face Detection & Recognition Integration

Combines YOLOv8 face detection with trained recognition model
for real-time criminal identification in CCTV footage.

Usage:
    # Webcam
    python scripts/integrate_cctv.py --source 0
    
    # Video file
    python scripts/integrate_cctv.py --source video.mp4
    
    # RTSP stream
    python scripts/integrate_cctv.py --source rtsp://camera_ip/stream
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: ultralytics not installed")
    print("Run: pip install ultralytics")
    sys.exit(1)

from src.recognition import FaceRecognitionSystem


class CCTVFaceRecognition:
    """
    Integrated CCTV Face Detection & Recognition System
    """
    
    def __init__(self, 
                 detector_model: str = 'yolov8n-face.pt',
                 recognition_model: str = 'models/model.pth',
                 threshold: float = 0.6,
                 conf_threshold: float = 0.5):
        """
        Initialize CCTV system
        
        Args:
            detector_model: Path to YOLOv8 face detection model
            recognition_model: Path to trained recognition model
            threshold: Recognition similarity threshold
            conf_threshold: Detection confidence threshold
        """
        print("🚀 Initializing CCTV Face Recognition System...")
        
        # Load face detector
        print(f"\n📍 Loading face detector: {detector_model}")
        try:
            self.detector = YOLO(detector_model)
            print("✅ Face detector loaded")
        except:
            print("⚠️ YOLOv8 face model not found, using default YOLOv8n")
            self.detector = YOLO('yolov8n.pt')
        
        # Load face recognizer
        print(f"\n🧠 Loading face recognizer: {recognition_model}")
        self.recognizer = FaceRecognitionSystem(
            model_path=recognition_model,
            threshold=threshold
        )
        
        self.conf_threshold = conf_threshold
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.recognition_count = 0
        
        print("\n✅ System initialized!\n")
    
    def register_criminals(self, data_dir: str):
        """
        Register known criminals from image directory
        
        Args:
            data_dir: Path to directory with criminal face images
                     Structure: data_dir/criminal_name/*.jpg
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"⚠️ Warning: Data directory not found: {data_dir}")
            return 0
        
        print(f"\n📝 Registering criminals from: {data_dir}")
        registered = 0
        
        for criminal_folder in sorted(data_path.iterdir()):
            if criminal_folder.is_dir():
                # Get all images
                image_paths = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_paths.extend([str(p) for p in criminal_folder.glob(ext)])
                
                if image_paths:
                    self.recognizer.register_identity(
                        identity_name=criminal_folder.name,
                        image_paths=image_paths[:10]  # Use up to 10 images
                    )
                    registered += 1
        
        print(f"\n✅ Registered {registered} criminals\n")
        return registered
    
    def process_frame(self, frame):
        """
        Process a single frame: detect and recognize faces
        
        Args:
            frame: BGR image from cv2
        
        Returns:
            Annotated frame with detections and identities
        """
        self.frame_count += 1
        annotated_frame = frame.copy()
        
        # Detect faces
        results = self.detector(frame, conf=self.conf_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Crop face
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    continue
                
                self.detection_count += 1
                
                # Convert to PIL for recognition
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                
                # Recognize
                identity, similarity = self.recognizer.predict(face_pil)
                
                # Color coding
                if identity != 'Unknown':
                    color = (0, 0, 255)  # Red for known criminals
                    self.recognition_count += 1
                    label = f"{identity} ({similarity:.2f})"
                else:
                    color = (0, 255, 0)  # Green for unknown
                    label = f"Unknown ({similarity:.2f})"
                
                # Draw box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, 
                          (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (255, 255, 255), 2)
                
                # Alert if criminal detected
                if identity != 'Unknown':
                    print(f"⚠️ ALERT: {identity} detected! (Confidence: {similarity:.2f})")
        
        return annotated_frame
    
    def run(self, source=0, display=True, save_output=None):
        """
        Run CCTV processing on video source
        
        Args:
            source: Video source (0 for webcam, path for video file, RTSP URL)
            display: Show output window
            save_output: Path to save output video (optional)
        """
        print(f"\n🎥 Starting video processing...")
        print(f"   Source: {source}")
        print(f"   Display: {display}")
        print(f"   Save output: {save_output or 'No'}")
        print("\n⏸️  Press 'q' to quit\n")
        
        # Open video source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video source: {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            print(f"💾 Saving output to: {save_output}")
        
        # Processing loop
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ End of video")
                break
            
            # Process frame
            annotated_frame = self.process_frame(frame)
            
            # Add stats overlay
            elapsed = time.time() - start_time
            current_fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            stats_text = [
                f"Frame: {self.frame_count}",
                f"FPS: {current_fps:.1f}",
                f"Faces: {self.detection_count}",
                f"Criminals: {self.recognition_count}"
            ]
            
            y_offset = 30
            for text in stats_text:
                cv2.putText(annotated_frame, text, 
                          (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (255, 255, 255), 2)
                y_offset += 30
            
            # Display
            if display:
                cv2.imshow('CCTV Face Recognition', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Stopped by user")
                    break
            
            # Save
            if writer:
                writer.write(annotated_frame)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Final stats
        print("\n" + "="*60)
        print("Processing Complete")
        print("="*60)
        print(f"  Total frames: {self.frame_count}")
        print(f"  Faces detected: {self.detection_count}")
        print(f"  Criminals identified: {self.recognition_count}")
        print(f"  Processing time: {elapsed:.1f}s")
        print(f"  Average FPS: {current_fps:.1f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='CCTV Face Recognition System')
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0=webcam, path=file, rtsp://=stream)')
    parser.add_argument('--detector', type=str, default='yolov8n-face.pt',
                      help='YOLOv8 face detection model')
    parser.add_argument('--recognizer', type=str, default='models/model.pth',
                      help='Trained recognition model')
    parser.add_argument('--data-dir', type=str, default='data_prepared',
                      help='Directory with criminal face images')
    parser.add_argument('--threshold', type=float, default=0.6,
                      help='Recognition similarity threshold')
    parser.add_argument('--conf', type=float, default=0.5,
                      help='Detection confidence threshold')
    parser.add_argument('--no-display', action='store_true',
                      help='Disable display window')
    parser.add_argument('--save', type=str, default=None,
                      help='Save output video to path')
    
    args = parser.parse_args()
    
    # Convert source to int if numeric
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Initialize system
    system = CCTVFaceRecognition(
        detector_model=args.detector,
        recognition_model=args.recognizer,
        threshold=args.threshold,
        conf_threshold=args.conf
    )
    
    # Register criminals
    system.register_criminals(args.data_dir)
    
    # Run processing
    system.run(
        source=source,
        display=not args.no_display,
        save_output=args.save
    )


if __name__ == "__main__":
    main()
