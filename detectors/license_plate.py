from models.detection_models import ModelManager
from config.settings import Config

class LicensePlateDetector:
    def __init__(self):
        self.model_manager = ModelManager()
    
    def detect_license_plate(self, frame, vehicle_bbox):
        """Extract license plate text from vehicle"""
        try:
            ocr_reader = self.model_manager.get_ocr_reader()
            x1, y1, x2, y2 = vehicle_bbox
            vehicle_roi = frame[max(0, y1):min(frame.shape[0], y2), 
                              max(0, x1):min(frame.shape[1], x2)]
            
            if vehicle_roi.size == 0:
                return ""
                
            results = ocr_reader.readtext(
                vehicle_roi, 
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            )
            
            for (bbox, text, conf) in results:
                if conf > Config.LICENSE_PLATE_CONFIDENCE_THRESHOLD and len(text.replace(' ', '')) >= 4:
                    clean_text = ''.join(c for c in text if c.isalnum())
                    if len(clean_text) >= 4:
                        return clean_text
            return ""
        except Exception as e:
            print(f"License plate detection error: {e}")
            return ""
