import os

class Config:
    # Model paths
    YOLO_MODEL_PATH = "yolov8s.pt"
    OCR_LANGUAGES = ['en']
    
    # Detection thresholds
    VEHICLE_CONFIDENCE_THRESHOLD = 0.5
    PERSON_CONFIDENCE_THRESHOLD = 0.5
    TRAFFIC_LIGHT_CONFIDENCE_THRESHOLD = 0.3
    LICENSE_PLATE_CONFIDENCE_THRESHOLD = 0.3  # Lowered for better detection
    
    # Violation parameters
    SPEED_LIMIT_KMH = 40  # More realistic speed limit
    LINE_CROSSING_TOLERANCE = 15
    HELMET_DETECTION_RATIO = 0.15
    NEARBY_VEHICLE_DISTANCE = 100
    
    # File paths
    CSV_LOG_FILE = "violation_log.csv"  # Keep original filename
    TEMP_DIR = os.path.join(os.getcwd(), "temp")
    
    # Video processing
    PIXEL_TO_METER_RATIO = 0.05
    MAX_SPEED_KMH = 200
    MIN_SPEED_THRESHOLD = 5  # Minimum speed to consider for violations
    
    # Calibration
    CALIBRATION_DISTANCE = 3.5
    
    # Zebra crossing detection - with safer defaults
    ZEBRA_CROSSING_MIN_AREA = 1000
    ZEBRA_CROSSING_WHITE_THRESHOLD = 200
    
    @classmethod
    def ensure_temp_dir(cls):
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
        return cls.TEMP_DIR
    
    @classmethod
    def ensure_data_dir(cls):
        # For backward compatibility, don't create data subdirectory
        return os.getcwd()
