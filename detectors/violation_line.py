import cv2
import numpy as np
from config.settings import Config

class ViolationLineDetector:
    @staticmethod
    def detect_zebra_crossing(frame):
        """Automatically detect zebra crossing and suggest violation line"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Focus on lower half of the image where crossings are typically located
            roi = gray[height//2:, :]
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            
            # Threshold to find white areas (zebra stripes)
            _, thresh = cv2.threshold(blurred, Config.ZEBRA_CROSSING_WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            zebra_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > Config.ZEBRA_CROSSING_MIN_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Zebra crossings are typically wide and not too tall
                    if 2.0 < aspect_ratio < 20.0:
                        # Adjust y coordinate to original frame coordinates
                        zebra_candidates.append((x, y + height//2, w, h))
            
            # Find the most suitable zebra crossing (largest area in lower portion)
            if zebra_candidates:
                best_crossing = max(zebra_candidates, key=lambda x: x[2] * x[3])
                x, y, w, h = best_crossing
                
                # Create violation line before the zebra crossing
                line_y = max(10, y - 20)  # 20 pixels before crossing
                line_start = (max(0, x - 50), line_y)
                line_end = (min(width, x + w + 50), line_y)
                
                return [line_start, line_end]
            
            # Fallback: Use Hough line detection for road markings
            return ViolationLineDetector._detect_road_markings(gray)
            
        except Exception as e:
            print(f"Error in zebra crossing detection: {e}")
            return None
    
    @staticmethod
    def _detect_road_markings(gray):
        """Fallback method using Hough line detection"""
        try:
            height, width = gray.shape
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Focus on horizontal lines in the lower half
            roi_edges = edges[height//2:, :]
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(roi_edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                horizontal_lines = []
                for rho, theta in lines[:10]:  # Check first 10 lines
                    # Filter for nearly horizontal lines
                    if abs(theta - np.pi/2) < 0.3:  # Within ~17 degrees of horizontal
                        y = int(rho / np.sin(theta)) + height//2
                        if height//2 < y < height - 50:  # In lower half but not at bottom
                            horizontal_lines.append(y)
                
                if horizontal_lines:
                    # Use the topmost horizontal line in the lower half
                    line_y = min(horizontal_lines)
                    return [(50, line_y), (width - 50, line_y)]
            
            # Ultimate fallback: horizontal line at 3/4 height
            fallback_y = int(height * 0.75)
            return [(50, fallback_y), (width - 50, fallback_y)]
            
        except Exception as e:
            print(f"Error in road marking detection: {e}")
            height, width = gray.shape
            fallback_y = int(height * 0.75)
            return [(50, fallback_y), (width - 50, fallback_y)]
    
    @staticmethod
    def visualize_detection(frame, detected_line):
        """Draw detected violation line for visualization"""
        if detected_line and len(detected_line) == 2:
            frame_copy = frame.copy()
            (x1, y1), (x2, y2) = detected_line
            cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame_copy, "AUTO-DETECTED VIOLATION LINE", (x1, y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return frame_copy
        return frame
