"""
Texture analysis for detecting printed photos, screens, and masks.
Uses Local Binary Patterns (LBP) and frequency analysis.
"""

from typing import Dict, List, Tuple
import numpy as np
import cv2
from scipy import ndimage
from loguru import logger


class TextureAnalyzer:
    """
    Analyze face texture to detect spoofing attempts.
    
    Uses multiple techniques:
    - Local Binary Pattern (LBP) histogram analysis
    - Frequency domain analysis (detecting screen moiré patterns)
    - Color distribution analysis
    - Reflection/glare detection
    """
    
    def __init__(
        self,
        lbp_radius: int = 1,
        lbp_neighbors: int = 8,
        frequency_threshold: float = 0.15,
        reflection_threshold: float = 0.3
    ):
        """
        Initialize texture analyzer.
        
        Args:
            lbp_radius: LBP neighborhood radius
            lbp_neighbors: Number of LBP neighborhood points
            frequency_threshold: Threshold for screen moiré detection
            reflection_threshold: Threshold for abnormal reflection detection
        """
        self.lbp_radius = lbp_radius
        self.lbp_neighbors = lbp_neighbors
        self.frequency_threshold = frequency_threshold
        self.reflection_threshold = reflection_threshold
        
        # Reference LBP histogram for real faces (learned from training)
        self._reference_histogram: np.ndarray = None
        
    def compute_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Compute Local Binary Pattern image.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            LBP encoded image
        """
        rows, cols = gray_image.shape
        lbp = np.zeros_like(gray_image, dtype=np.uint8)
        
        for i in range(self.lbp_radius, rows - self.lbp_radius):
            for j in range(self.lbp_radius, cols - self.lbp_radius):
                center = gray_image[i, j]
                binary = 0
                
                # Sample points around center
                for n in range(self.lbp_neighbors):
                    angle = 2 * np.pi * n / self.lbp_neighbors
                    x = int(round(i + self.lbp_radius * np.cos(angle)))
                    y = int(round(j - self.lbp_radius * np.sin(angle)))
                    
                    if gray_image[x, y] >= center:
                        binary |= (1 << n)
                
                lbp[i, j] = binary
        
        return lbp
    
    def compute_lbp_histogram(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Compute normalized LBP histogram.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Normalized histogram
        """
        lbp = self.compute_lbp(gray_image)
        n_bins = 2 ** self.lbp_neighbors
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        return hist
    
    def analyze_frequency(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze frequency domain for moiré patterns and screen artifacts.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Dictionary with frequency analysis results
        """
        # Apply FFT
        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Log magnitude for better visualization
        magnitude_log = np.log1p(magnitude)
        
        rows, cols = gray_image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Analyze high frequency content (edges of frequency space)
        mask_radius = min(rows, cols) // 4
        y, x = np.ogrid[:rows, :cols]
        mask = (x - ccol) ** 2 + (y - crow) ** 2 > mask_radius ** 2
        
        high_freq_energy = np.sum(magnitude_log[mask])
        total_energy = np.sum(magnitude_log)
        
        high_freq_ratio = high_freq_energy / (total_energy + 1e-7)
        
        # Detect periodic patterns (moiré)
        # Look for strong peaks in frequency domain
        threshold = np.mean(magnitude_log) + 3 * np.std(magnitude_log)
        peaks = magnitude_log > threshold
        # Exclude DC component
        peaks[crow-5:crow+5, ccol-5:ccol+5] = False
        
        periodic_score = np.sum(peaks) / (rows * cols)
        
        return {
            'high_freq_ratio': float(high_freq_ratio),
            'periodic_score': float(periodic_score),
            'likely_screen': periodic_score > self.frequency_threshold
        }
    
    def analyze_color_distribution(self, bgr_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze color distribution for signs of printed photos or screens.
        
        Args:
            bgr_image: BGR color image
            
        Returns:
            Dictionary with color analysis results
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
        
        # Analyze saturation - printed photos often have different saturation
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        # Analyze skin tone consistency in Cr channel
        cr = ycrcb[:, :, 1]
        cr_mean = np.mean(cr)
        cr_std = np.std(cr)
        
        # Check for unnaturally uniform colors
        uniformity_score = 1.0 - min(1.0, sat_std / 50.0)
        
        # Detect color banding (common in prints/screens)
        unique_colors = len(np.unique(bgr_image.reshape(-1, 3), axis=0))
        total_pixels = bgr_image.shape[0] * bgr_image.shape[1]
        color_diversity = unique_colors / total_pixels
        
        return {
            'saturation_mean': float(sat_mean),
            'saturation_std': float(sat_std),
            'cr_mean': float(cr_mean),
            'cr_std': float(cr_std),
            'uniformity_score': float(uniformity_score),
            'color_diversity': float(color_diversity),
            'likely_print': uniformity_score > 0.7 or color_diversity < 0.1
        }
    
    def detect_reflections(self, gray_image: np.ndarray) -> Dict[str, any]:
        """
        Detect abnormal reflections that might indicate a screen or glossy print.
        
        Args:
            gray_image: Grayscale input image
            
        Returns:
            Dictionary with reflection analysis results
        """
        # Find very bright spots
        _, bright_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
        
        # Calculate ratio of bright pixels
        bright_ratio = np.sum(bright_mask > 0) / bright_mask.size
        
        # Detect specular highlights using Laplacian
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        highlight_score = np.mean(np.abs(laplacian))
        
        # Check for rectangular bright regions (screen reflections)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_reflections = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box.astype(np.int32))
                cnt_area = cv2.contourArea(cnt)
                if box_area > 0 and cnt_area / box_area > 0.8:
                    rectangular_reflections += 1
        
        return {
            'bright_ratio': float(bright_ratio),
            'highlight_score': float(highlight_score),
            'rectangular_reflections': rectangular_reflections,
            'has_abnormal_reflection': bright_ratio > self.reflection_threshold or rectangular_reflections > 0
        }
    
    def analyze(self, face_image: np.ndarray) -> Dict[str, any]:
        """
        Perform complete texture analysis on a face image.
        
        Args:
            face_image: BGR face image (cropped)
            
        Returns:
            Dictionary with complete analysis results and liveness score
        """
        if face_image is None or face_image.size == 0:
            return {'error': 'Invalid image', 'is_live': False, 'confidence': 0.0}
        
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Run all analyses
        lbp_hist = self.compute_lbp_histogram(gray)
        freq_analysis = self.analyze_frequency(gray)
        color_analysis = self.analyze_color_distribution(face_image) if len(face_image.shape) == 3 else {}
        reflection_analysis = self.detect_reflections(gray)
        
        # Calculate overall liveness score
        spoof_indicators = 0
        total_checks = 0
        
        if freq_analysis.get('likely_screen', False):
            spoof_indicators += 1
        total_checks += 1
        
        if color_analysis.get('likely_print', False):
            spoof_indicators += 1
        total_checks += 1
        
        if reflection_analysis.get('has_abnormal_reflection', False):
            spoof_indicators += 1
        total_checks += 1
        
        # High periodic score suggests screen
        if freq_analysis.get('periodic_score', 0) > 0.05:
            spoof_indicators += 0.5
        total_checks += 1
        
        spoof_probability = spoof_indicators / total_checks
        is_live = spoof_probability < 0.4
        confidence = 1.0 - spoof_probability
        
        return {
            'is_live': is_live,
            'confidence': float(confidence),
            'spoof_probability': float(spoof_probability),
            'lbp_histogram': lbp_hist.tolist(),
            'frequency_analysis': freq_analysis,
            'color_analysis': color_analysis,
            'reflection_analysis': reflection_analysis
        }
    
    def set_reference_histogram(self, histogram: np.ndarray):
        """Set reference LBP histogram from real face training data."""
        self._reference_histogram = histogram.astype(np.float32)
        self._reference_histogram /= (self._reference_histogram.sum() + 1e-7)
    
    def compare_to_reference(self, face_image: np.ndarray) -> float:
        """
        Compare face texture to reference real face histogram.
        
        Args:
            face_image: Face image to analyze
            
        Returns:
            Similarity score (higher = more likely real)
        """
        if self._reference_histogram is None:
            logger.warning("No reference histogram set, returning default score")
            return 0.5
        
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        hist = self.compute_lbp_histogram(gray)
        
        # Chi-squared distance
        chi_sq = cv2.compareHist(
            hist.astype(np.float32),
            self._reference_histogram,
            cv2.HISTCMP_CHISQR
        )
        
        # Convert to similarity (lower chi-sq = more similar)
        similarity = 1.0 / (1.0 + chi_sq)
        return float(similarity)
