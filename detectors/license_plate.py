import cv2
import numpy as np
import re
from models.detection_models import ModelManager
from config.settings import Config

class LicensePlateDetector:
    def __init__(self):
        self.model_manager = ModelManager()
    
    def detect_license_plate(self, frame, vehicle_bbox):
        """Extract license plate text from vehicle with enhanced preprocessing"""
        try:
            ocr_reader = self.model_manager.get_ocr_reader()
            x1, y1, x2, y2 = vehicle_bbox
            
            # Extract vehicle ROI with some padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            vehicle_roi = frame[y1:y2, x1:x2]
            
            if vehicle_roi.size == 0:
                return ""
            
            # Focus on the lower part of vehicle where license plates are typically located
            height = vehicle_roi.shape[0]
            lower_roi = vehicle_roi[int(height*0.6):, :]
            
            # Try multiple preprocessing approaches
            license_candidates = []
            
            # Method 1: Original image
            candidates = self._extract_text_with_ocr(ocr_reader, lower_roi)
            license_candidates.extend(candidates)
            
            # Method 2: Enhanced preprocessing
            enhanced_roi = self._preprocess_for_ocr(lower_roi)
            candidates = self._extract_text_with_ocr(ocr_reader, enhanced_roi)
            license_candidates.extend(candidates)
            
            # Method 3: Contour-based license plate detection
            plate_regions = self._detect_plate_regions(lower_roi)
            for region in plate_regions:
                processed_region = self._preprocess_for_ocr(region)
                candidates = self._extract_text_with_ocr(ocr_reader, processed_region)
                license_candidates.extend(candidates)
            
            # Return the best candidate
            return self._select_best_license_plate(license_candidates)
            
        except Exception as e:
            print(f"License plate detection error: {e}")
            return ""
    
    def _preprocess_for_ocr(self, roi):
        """Enhanced preprocessing for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Resize for better OCR (if image is too small)
        height, width = processed.shape
        if height < 50 or width < 150:
            scale_factor = max(2, 150 // width)
            processed = cv2.resize(processed, (width * scale_factor, height * scale_factor),
                                 interpolation=cv2.INTER_CUBIC)
        
        return processed
    
    def _detect_plate_regions(self, roi):
        """Detect potential license plate regions using contours"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # License plates typically have aspect ratio between 2:1 and 5:1
                if 1.5 < aspect_ratio < 6.0 and w > 50 and h > 15:
                    # Extract the region
                    plate_region = roi[y:y+h, x:x+w]
                    if plate_region.size > 0:
                        plate_regions.append(plate_region)
        
        return plate_regions
    
    def _extract_text_with_ocr(self, ocr_reader, image):
        """Extract text using OCR with improved settings"""
        try:
            results = ocr_reader.readtext(
                image,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                width_ths=0.7,
                height_ths=0.7,
                paragraph=False
            )
            
            candidates = []
            for (bbox, text, conf) in results:
                if conf > Config.LICENSE_PLATE_CONFIDENCE_THRESHOLD:
                    clean_text = self._clean_license_text(text)
                    if self._is_valid_indian_license_plate(clean_text):
                        candidates.append((clean_text, conf))
            
            return candidates
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return []
    
    def _clean_license_text(self, text):
        """Clean and format license plate text"""
        # Remove non-alphanumeric characters
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Common OCR corrections for Indian license plates
        corrections = {
            'O': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'G': '6', 'B': '8'
        }
        
        # Apply corrections only to numeric parts
        result = ""
        for char in clean_text:
            if char.isdigit() or char in corrections.values():
                result += char
            elif char.isalpha():
                result += corrections.get(char, char)
            else:
                result += char
        
        return result
    
    def _is_valid_indian_license_plate(self, text):
        """Validate if text matches Indian license plate patterns"""
        if not text or len(text) < 6:
            return False
        
        # Indian license plate patterns:
        # Old format: XX00XX0000 (2 letters, 2 digits, 2 letters, 4 digits)
        # New format: XX00XX0000 (2 letters, 2 digits, 2 letters, 4 digits)
        # Simplified check for alphanumeric with reasonable length
        
        patterns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,5}$',  # Standard format
            r'^[A-Z]{1,3}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,5}$',  # Variations
            r'^[0-9]{2}[A-Z]{2}[0-9]{4}$',  # Some old formats
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        # Fallback: reasonable length with mix of letters and numbers
        if 6 <= len(text) <= 12:
            has_letters = any(c.isalpha() for c in text)
            has_numbers = any(c.isdigit() for c in text)
            return has_letters and has_numbers
        
        return False
    
    def _select_best_license_plate(self, candidates):
        """Select the best license plate candidate from all methods"""
        if not candidates:
            return ""
        
        # Remove duplicates and sort by confidence
        unique_candidates = {}
        for text, conf in candidates:
            if text not in unique_candidates or conf > unique_candidates[text]:
                unique_candidates[text] = conf
        
        if not unique_candidates:
            return ""
        
        # Return the candidate with highest confidence
        best_candidate = max(unique_candidates.items(), key=lambda x: x[1])
        return best_candidate[0] if best_candidate[1] > Config.LICENSE_PLATE_CONFIDENCE_THRESHOLD else ""
