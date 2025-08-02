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
from detectors.violation_line import ViolationLineDetector

class ViolationDetector:
    def __init__(self):
        self.violation_line = None
        self.traffic_light_detector = TrafficLightDetector()
        self.helmet_detector = HelmetDetector()
        self.license_plate_detector = LicensePlateDetector()
        self.speed_calculator = SpeedCalculator()
        self.logger = ViolationLogger()
        
        # Ensure temp directory exists
        Config.ensure_temp_dir()
    
    def set_violation_line(self, point1, point2):
        """Set the violation line coordinates"""
        self.violation_line = (point1, point2)
    
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
        
        # Vertical line
        if x1 == x2:
            return abs(vx - x1) < 10
        
        # Horizontal line
        if y1 == y2:
            return vy > y1 - 10 and vy < y1 + 10
        
        # Diagonal line
        slope = (y2 - y1) / (x2 - x1)
        expected_y = y1 + slope * (vx - x1)
        return abs(vy - expected_y) < Config.LINE_CROSSING_TOLERANCE
    
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
