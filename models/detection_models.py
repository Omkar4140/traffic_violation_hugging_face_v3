from ultralytics import YOLO
import easyocr
from config.settings import Config

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.yolo_model = None
            self.ocr_reader = None
            self._initialized = True
    
    def load_models(self):
        """Load YOLO and OCR models"""
        if self.yolo_model is None:
            print("Loading YOLO model...")
            self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)
        
        if self.ocr_reader is None:
            print("Loading OCR model...")
            self.ocr_reader = easyocr.Reader(Config.OCR_LANGUAGES)
        
        return self.yolo_model, self.ocr_reader
    
    def get_yolo_model(self):
        if self.yolo_model is None:
            self.load_models()
        return self.yolo_model
    
    def get_ocr_reader(self):
        if self.ocr_reader is None:
            self.load_models()
        return self.ocr_reader
