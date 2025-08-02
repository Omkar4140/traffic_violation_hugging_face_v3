import cv2
import numpy as np

class TrafficLightDetector:
    @staticmethod
    def detect_color(frame, bbox):
        """Detect traffic light color from bounding box"""
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return "unknown"
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        red_mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        red_mask = red_mask1 + red_mask2
        
        yellow_mask = cv2.inRange(hsv, (20, 50, 50), (30, 255, 255))
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Determine dominant color
        if red_pixels > max(yellow_pixels, green_pixels):
            return "red"
        elif yellow_pixels > green_pixels:
            return "yellow"
        elif green_pixels > 0:
            return "green"
        return "unknown"
