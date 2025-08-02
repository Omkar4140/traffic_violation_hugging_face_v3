import os
import cv2
import numpy as np
from typing import Tuple, Optional, List

class GeometryUtils:
    @staticmethod
    def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def is_point_near_line(point: Tuple[int, int], line_start: Tuple[int, int], 
                          line_end: Tuple[int, int], tolerance: int = 15) -> bool:
        """Check if a point is near a line within tolerance"""
        x1, y1 = line_start
        x2, y2 = line_end
        px, py = point
        
        # Vertical line
        if x1 == x2:
            return abs(px - x1) < tolerance
        
        # Horizontal line
        if y1 == y2:
            return abs(py - y1) < tolerance
        
        # Diagonal line - calculate perpendicular distance
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        distance = abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)
        return distance < tolerance

class FileUtils:
    @staticmethod
    def ensure_directory(directory_path: str) -> str:
        """Ensure directory exists, create if not"""
        os.makedirs(directory_path, exist_ok=True)
        return directory_path
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get file extension in lowercase"""
        return os.path.splitext(file_path)[1].lower()
    
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """Check if file is a video"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        return FileUtils.get_file_extension(file_path) in video_extensions
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """Check if file is an image"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        return FileUtils.get_file_extension(file_path) in image_extensions

class ValidationUtils:
    @staticmethod
    def validate_coordinates(coords_str: str) -> Optional[List[Tuple[int, int]]]:
        """Validate and parse coordinate string"""
        try:
            if not coords_str or coords_str.strip() == "":
                return None
            
            coords = eval(coords_str)
            if isinstance(coords, list) and len(coords) == 2:
                # Validate each coordinate is a tuple of 2 integers
                for coord in coords:
                    if not isinstance(coord, tuple) or len(coord) != 2:
                        return None
                    if not all(isinstance(x, (int, float)) for x in coord):
                        return None
                return [(int(x), int(y)) for x, y in coords]
        except Exception as e:
            print(f"Error validating coordinates: {e}")
        return None
    
    @staticmethod
    def validate_bbox(bbox: Tuple[int, int, int, int]) -> bool:
        """Validate bounding box coordinates"""
        x1, y1, x2, y2 = bbox
        return x1 < x2 and y1 < y2 and all(x >= 0 for x in bbox)

class ImageUtils:
    @staticmethod
    def resize_image(image: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        if width <= max_width and height <= max_height:
            return image
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def add_watermark(image: np.ndarray, text: str = "Traffic Violation Detector") -> np.ndarray:
        """Add watermark to image"""
        height, width = image.shape[:2]
        
        # Calculate text size and position
        font_scale = min(width, height) / 1000
        thickness = max(1, int(font_scale * 2))
        
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Position at bottom right
        x = width - text_width - 10
        y = height - 10
        
        # Add semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (x - 5, y - text_height - 5), 
                     (x + text_width + 5, y + 5), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
        
        # Add text
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), thickness)
        
        return image
