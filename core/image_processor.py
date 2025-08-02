import cv2
import os
from datetime import datetime
from models.detection_models import ModelManager
from config.settings import Config

class ImageProcessor:
    def __init__(self, violation_detector):
        self.detector = violation_detector
        self.model_manager = ModelManager()
    
    def process_image(self, image_path, enable_plate_detection=True):
        """Process a single image for violations with enhanced features"""
        frame = cv2.imread(image_path)
        if frame is None:
            return None, [], None
        
        # Auto-detect violation line if none is set
        if not self.detector.violation_line:
            auto_line = self.detector.auto_detect_violation_line(frame)
            if auto_line:
                print(f"Auto-detected violation line: {auto_line}")
        
        yolo_model = self.model_manager.get_yolo_model()
        output_frame = frame.copy()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frame_no = 0
        
        # Run YOLO detection
        results = yolo_model(frame)[0]
        vehicle_detections = []
        person_detections = []
        traffic_lights = []
        
        # Parse detections
        for box in results.boxes:
            cls_name = yolo_model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if cls_name in ["car", "truck", "bus", "motorbike", "bicycle"] and conf > Config.VEHICLE_CONFIDENCE_THRESHOLD:
                vehicle_detections.append((cls_name, (x1, y1, x2, y2), conf))
            elif cls_name == "person" and conf > Config.PERSON_CONFIDENCE_THRESHOLD:
                person_detections.append((x1, y1, x2, y2))
            elif cls_name == "traffic light" and conf > Config.TRAFFIC_LIGHT_CONFIDENCE_THRESHOLD:
                traffic_lights.append((x1, y1, x2, y2))
        
        # Detect traffic light state
        traffic_light_state = self._detect_traffic_light_state(frame, traffic_lights)
        
        # Process vehicle violations
        output_frame = self._process_vehicle_violations(
            frame, output_frame, vehicle_detections, traffic_light_state, 
            timestamp, frame_no, enable_plate_detection
        )
        
        # Process helmet violations
        output_frame = self._process_helmet_violations(
            frame, output_frame, person_detections, vehicle_detections, timestamp, frame_no
        )
        
        # Draw violation line with enhanced visibility
        self._draw_violation_line(output_frame)
        
        # Add enhanced info panel
        self._add_info_panel(output_frame, traffic_light_state, len(vehicle_detections), 
                           len(person_detections), len(self.detector.logger.violations_log))
        
        # Save processed image
        output_path = os.path.join(Config.TEMP_DIR, "processed_image.jpg")
        cv2.imwrite(output_path, output_frame)
        
        return output_path, self.detector.logger.get_violations_dataframe(), None
    
    def _detect_traffic_light_state(self, frame, traffic_lights):
        """Detect traffic light state"""
        for light_bbox in traffic_lights:
            state = self.detector.traffic_light_detector.detect_color(frame, light_bbox)
            if state != "unknown":
                return state
        return "unknown"
    
    def _process_vehicle_violations(self, frame, output_frame, vehicle_detections, 
                                  traffic_light_state, timestamp, frame_no, enable_plate_detection):
        """Process vehicle-related violations with enhanced detection"""
        for vehicle_type, bbox, conf in vehicle_detections:
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Enhanced license plate detection
            license_plate = ""
            if enable_plate_detection:
                license_plate = self.detector.license_plate_detector.detect_license_plate(frame, bbox)
            
            repeat_offender = self.detector.logger.is_repeat_offender(license_plate)
            
            # Draw bounding box with enhanced styling
            color = (0, 0, 255) if repeat_offender else (0, 255, 0)
            thickness = 3 if repeat_offender else 2
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Create enhanced label with background
            label = f"{vehicle_type} {conf:.2f}"
            if license_plate:
                label += f" [{license_plate}]"
            if repeat_offender:
                label += " [REPEAT OFFENDER]"
            
            # Draw label background for better visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output_frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(output_frame, label, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Check for red light violation
            if self.detector.violation_line and traffic_light_state == "red":
                if self.detector.is_crossing_line(center):
                    screenshot_path = self.detector.save_violation_screenshot(
                        frame, bbox, "red_light", timestamp)
                    self.detector.logger.log_violation(
                        timestamp, "red_light_violation", vehicle_type, 
                        conf, 0, license_plate, frame_no, 
                        screenshot_path, repeat_offender
                    )
                    
                    # Enhanced violation visualization
                    cv2.circle(output_frame, center, 25, (0, 0, 255), -1)
                    cv2.circle(output_frame, center, 30, (255, 255, 255), 3)
                    cv2.putText(output_frame, "RED LIGHT VIOLATION", (x1, y2 + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        return output_frame
    
    def _process_helmet_violations(self, frame, output_frame, person_detections, 
                                 vehicle_detections, timestamp, frame_no):
        """Process helmet-related violations with enhanced detection"""
        import numpy as np
        
        for person_bbox in person_detections:
            x1, y1, x2, y2 = person_bbox
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Find nearby motorcycle/bicycle
            nearby_vehicle = None
            nearby_bbox = None
            for vehicle_type, v_bbox, v_conf in vehicle_detections:
                if vehicle_type in ["motorbike", "bicycle"]:
                    v_x1, v_y1, v_x2, v_y2 = v_bbox
                    v_center = ((v_x1 + v_x2) // 2, (v_y1 + v_y2) // 2)
                    distance = np.sqrt((person_center[0] - v_center[0])**2 + 
                                     (person_center[1] - v_center[1])**2)
                    if distance < Config.NEARBY_VEHICLE_DISTANCE:
                        nearby_vehicle = vehicle_type
                        nearby_bbox = v_bbox
                        break
            
            if nearby_vehicle:
                has_helmet = self.detector.helmet_detector.detect_helmet(frame, person_bbox)
                
                if not has_helmet:
                    screenshot_path = self.detector.save_violation_screenshot(
                        frame, nearby_bbox, "no_helmet", timestamp)
                    self.detector.logger.log_violation(
                        timestamp, "no_helmet_violation", nearby_vehicle, 
                        0.8, 0, "", frame_no, screenshot_path, False
                    )
                    
                    # Enhanced helmet violation visualization
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                    cv2.rectangle(output_frame, (x1-5, y1-5), (x2+5, y2+5), (255, 255, 255), 2)
                    
                    # Add warning icon
                    cv2.putText(output_frame, "⚠", (x2 + 5, y1 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                    cv2.putText(output_frame, "NO HELMET", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 3)
        
        return output_frame
    
    def _draw_violation_line(self, output_frame):
        """Draw violation line with enhanced visibility"""
        line = self.detector.get_violation_line_for_display()
        if line:
            (x1, y1), (x2, y2) = line
            
            # Draw multi-layered line for maximum visibility
            cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 0), 8)      # Black outline
            cv2.line(output_frame, (x1, y1), (x2, y2), (255, 255, 0), 6)  # Yellow main
            cv2.line(output_frame, (x1, y1), (x2, y2), (255, 255, 255), 2) # White center
            
            # Add enhanced label
            label = "VIOLATION LINE"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            label_x = x1
            label_y = max(30, y1 - 20)
            
            # Label background
            cv2.rectangle(output_frame, (label_x - 5, label_y - 25), 
                         (label_x + label_size[0] + 5, label_y + 5), (0, 0, 0), -1)
            cv2.rectangle(output_frame, (label_x - 5, label_y - 25), 
                         (label_x + label_size[0] + 5, label_y + 5), (255, 255, 0), 2)
            cv2.putText(output_frame, label, (label_x, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    def _add_info_panel(self, output_frame, traffic_light_state, vehicle_count, 
                       person_count, violation_count):
        """Add enhanced information panel"""
        height, width = output_frame.shape[:2]
        
        # Create info panel background
        panel_height = 100
        cv2.rectangle(output_frame, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(output_frame, (0, 0), (width, panel_height), (255, 255, 255), 3)
        
        # Add title
        cv2.putText(output_frame, "TRAFFIC VIOLATION DETECTION", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add detection info
        info_text = f"Traffic Light: {traffic_light_state.upper()} | Vehicles: {vehicle_count} | Persons: {person_count}"
        cv2.putText(output_frame, info_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add violation count with color coding
        violation_color = (0, 255, 0) if violation_count == 0 else (0, 0, 255)
        violation_text = f"Violations Detected: {violation_count}"
        cv2.putText(output_frame, violation_text, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, violation_color, 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(output_frame, timestamp, (width - timestamp_size[0] - 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
