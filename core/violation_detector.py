import os
import cv2
import numpy as np
from datetime import datetime
from config.settings import Config
from detectors.traffic_light import TrafficLightDetector
from detectors.helmet import HelmetDetector
from detectors.license_plate import LicensePlateDetector
from detectors.speed import SpeedCalculator
from data.logger import ViolationLogger

# Try to import the new violation line detector, fallback if not available
try:
    from detectors.violation_line import ViolationLineDetector
    HAS_AUTO_DETECTION = True
except ImportError:
    print("ViolationLineDetector not found, using fallback auto-detection")
    HAS_AUTO_DETECTION = False

class ViolationDetector:
    def __init__(self):
        self.violation_line = None
        self.auto_detected_line = None
        self.traffic_light_detector = TrafficLightDetector()
        self.helmet_detector = HelmetDetector()
        self.license_plate_detector = LicensePlateDetector()
        self.speed_calculator = SpeedCalculator()
        self.logger = ViolationLogger()
        
        # Initialize line detector if available
        if HAS_AUTO_DETECTION:
            self.line_detector = ViolationLineDetector()
        else:
            self.line_detector = None
        
        # Ensure temp directory exists
        Config.ensure_temp_dir()
    
    def set_violation_line(self, point1, point2):
        """Set the violation line coordinates manually"""
        self.violation_line = (point1, point2)
        self.auto_detected_line = None  # Clear auto-detected line when manual is set
    
    def auto_detect_violation_line(self, frame):
        """Automatically detect violation line from frame"""
        if not self.line_detector:
            # Fallback auto-detection
            return self._simple_auto_detect(frame)
        
        detected_line = self.line_detector.detect_zebra_crossing(frame)
        if detected_line:
            self.auto_detected_line = detected_line
            # If no manual line is set, use the auto-detected one
            if not self.violation_line:
                self.violation_line = (tuple(detected_line[0]), tuple(detected_line[1]))
            return detected_line
        return None
    
    def _simple_auto_detect(self, frame):
        """Simple fallback auto-detection"""
        try:
            height, width = frame.shape[:2]
            # Place line at 75% of image height
            line_y = int(height * 0.75)
            line_start = (50, line_y)
            line_end = (width - 50, line_y)
            
            self.auto_detected_line = [line_start, line_end]
            if not self.violation_line:
                self.violation_line = (line_start, line_end)
            
            return [line_start, line_end]
        except Exception as e:
            print(f"Error in simple auto-detection: {e}")
            return None
    
    def parse_line_coordinates(self, line_coords):
        """Parse line coordinates from string input"""
        try:
            if not line_coords or line_coords.strip() == "":
                return None
            coords = eval(line_coords)
            if isinstance(coords, list) and len(coords) == 2:
                return coords
        except Exception as e:
            print(f"Error parsing coordinates: {e}")
        return None
    
    def is_crossing_line(self, vehicle_center):
        """Check if vehicle center crosses the violation line"""
        if not self.violation_line:
            return False
        
        (x1, y1), (x2, y2) = self.violation_line
        vx, vy = vehicle_center
        
        # Calculate distance from point to line
        if x1 == x2:  # Vertical line
            distance = abs(vx - x1)
            return distance < Config.LINE_CROSSING_TOLERANCE
        elif y1 == y2:  # Horizontal line
            distance = abs(vy - y1)
            return distance < Config.LINE_CROSSING_TOLERANCE
        else:  # Diagonal line
            # Calculate perpendicular distance from point to line
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            distance = abs(A * vx + B * vy + C) / np.sqrt(A**2 + B**2)
            return distance < Config.LINE_CROSSING_TOLERANCE
    
    def save_violation_screenshot(self, frame, vehicle_bbox, violation_type, timestamp):
        """Save screenshot of violation"""
        try:
            x1, y1, x2, y2 = vehicle_bbox
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(frame.shape[1], x2 + margin)
            y2 = min(frame.shape[0], y2 + margin)
            
            cropped = frame[y1:y2, x1:x2]
            
            timestamp_clean = timestamp.replace(':', '-').replace(' ', '_')
            filename = f"violation_{violation_type}_{timestamp_clean}.jpg"
            filepath = os.path.join(Config.TEMP_DIR, filename)
            
            cv2.imwrite(filepath, cropped)
            return filepath
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            return ""
    
    def get_violation_line_for_display(self):
        """Get the current violation line for display purposes"""
        if self.violation_line:
            return self.violation_line
        elif self.auto_detected_line:
            return (tuple(self.auto_detected_line[0]), tuple(self.auto_detected_line[1]))
        return None
    
    def clear_violation_logs(self):
        """Clear all violation logs"""
        return self.logger.clear_violations_csv()
    
    def reset_session(self):
        """Reset the current session data"""
        self.logger.violations_log = []
        if hasattr(self.speed_calculator, 'vehicle_tracks'):
            self.speed_calculator.vehicle_tracks = {}
        if hasattr(self.speed_calculator, 'frame_timestamps'):
            self.speed_calculator.frame_timestamps = {}
