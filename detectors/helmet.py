import cv2
import numpy as np
from config.settings import Config

class HelmetDetector:
    @staticmethod
    def detect_helmet(frame, person_bbox):
        """Detect if person is wearing a helmet"""
        x1, y1, x2, y2 = person_bbox
        person_roi = frame[y1:y2, x1:x2]
        person_height = y2 - y1
        head_region = person_roi[:int(person_height * 0.3), :]
        
        if head_region.size == 0:
            return False
            
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Define helmet color ranges
        helmet_colors = [
            ((0, 0, 0), (180, 255, 50)),      # Dark colors
            ((0, 0, 100), (180, 50, 255)),    # Bright colors
            ((100, 50, 50), (130, 255, 255))  # Blue range
        ]
        
        helmet_pixels = 0
        for lower, upper in helmet_colors:
            mask = cv2.inRange(hsv, lower, upper)
            helmet_pixels += cv2.countNonZero(mask)
        
        total_pixels = head_region.shape[0] * head_region.shape[1]
        helmet_ratio = helmet_pixels / total_pixels if total_pixels > 0 else 0
        
        return helmet_ratio > Config.HELMET_DETECTION_RATIO
