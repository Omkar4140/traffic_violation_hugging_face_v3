import cv2
import os
import numpy as np
from datetime import datetime
from models.detection_models import ModelManager
from config.settings import Config

class VideoProcessor:
    def __init__(self, violation_detector):
        self.detector = violation_detector
        self.model_manager = ModelManager()
    
    def process_video(self, video_path, enable_plate_detection=True):
        """Process video for violations"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, [], None
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Output video setup
        output_path = os.path.join(Config.TEMP_DIR, "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        yolo_model = self.model_manager.get_yolo_model()
        frame_count = 0
        vehicle_trackers = {}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                output_frame = frame.copy()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # Run YOLO detection
                results = yolo_model(frame)[0]
                current_detections = {}
                person_detections = []
                traffic_lights = []
                
                # Parse detections
                for box in results.boxes:
                    cls_name = yolo_model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if cls_name in ["car", "truck", "bus", "motorbike", "bicycle"] and conf > Config.VEHICLE_CONFIDENCE_THRESHOLD:
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        current_detections[len(current_detections)] = {
                            'type': cls_name, 'bbox': (x1, y1, x2, y2), 
                            'center': center, 'conf': conf
                        }
                    elif cls_name == "person" and conf > Config.PERSON_CONFIDENCE_THRESHOLD:
                        person_detections.append((x1, y1, x2, y2))
                    elif cls_name == "traffic light" and conf > Config.TRAFFIC_LIGHT_CONFIDENCE_THRESHOLD:
                        traffic_lights.append((x1, y1, x2, y2))
                
                # Detect traffic light state
                traffic_light_state = self._detect_traffic_light_state(frame, traffic_lights)
                
                # Process vehicle violations
                output_frame = self._process_vehicles(
                    frame, output_frame, current_detections, vehicle_trackers,
                    traffic_light_state, timestamp, frame_count, fps, enable_plate_detection
                )
                
                # Process helmet violations
                output_frame = self._process_helmet_violations(
                    frame, output_frame, person_detections, current_detections, timestamp, frame_count
                )
                
                # Draw violation line
                if self.detector.violation_line:
                    self._draw_violation_line(output_frame)
                
                # Add frame info
                cv2.putText(output_frame, f"Frame: {frame_count} | Light: {traffic_light_state.upper()}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(output_frame)
                frame_count += 1
                
        finally:
            cap.release()
            out.release()
        
        return output_path, self.detector.logger.get_violations_dataframe(), None
    
    def _detect_traffic_light_state(self, frame, traffic_lights):
        """Detect traffic light state"""
        for light_bbox in traffic_lights:
            state = self.detector.traffic_light_detector.detect_color(frame, light_bbox)
            if state != "unknown":
                return state
        return "unknown"
    
    def _process_vehicles(self, frame, output_frame, current_detections, vehicle_trackers,
                         traffic_light_state, timestamp, frame_count, fps, enable_plate_detection):
        """Process vehicle detections and violations"""
        for det_id, detection in current_detections.items():
            vehicle_type = detection['type']
            bbox = detection['bbox']
            center = detection['center']
            conf = detection['conf']
            x1, y1, x2, y2 = bbox
            
            # License plate detection
            license_plate = ""
            if enable_plate_detection:
                license_plate = self.detector.license_plate_detector.detect_license_plate(frame, bbox)
            
            repeat_offender = self.detector.logger.is_repeat_offender(license_plate)
            
            # Speed calculation
            speed = 0
            if det_id in vehicle_trackers:
                prev_center = vehicle_trackers[det_id]['center']
                speed = self.detector.speed_calculator.calculate_speed(prev_center, center, fps)
                
                # Check for speeding violation
                if speed > Config.SPEED_LIMIT_KMH:
                    screenshot_path = self.detector.save_violation_screenshot(
                        frame, bbox, "speeding", timestamp)
                    self.detector.logger.log_violation(
                        timestamp, "speeding_violation", vehicle_type, 
                        conf, speed, license_plate, frame_count, 
                        screenshot_path, repeat_offender
                    )
                    
                    cv2.putText(output_frame, f"SPEEDING: {speed:.1f} km/h", (x1, y2 + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Update tracker
            vehicle_trackers[det_id] = {'center': center, 'type': vehicle_type}
            
            # Draw bounding box
            color = (0, 0, 255) if repeat_offender else (0, 255, 0)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{vehicle_type} {conf:.2f}"
            if speed > 0:
                label += f" {speed:.1f}km/h"
            if license_plate:
                label += f" [{license_plate}]"
            if repeat_offender:
                label += " [REPEAT]"
            
            cv2.putText(output_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Check for red light violation
            if self.detector.violation_line and traffic_light_state == "red":
                if self.detector.is_crossing_line(center):
                    screenshot_path = self.detector.save_violation_screenshot(
                        frame, bbox, "red_light", timestamp)
                    self.detector.logger.log_violation(
                        timestamp, "red_light_violation", vehicle_type, 
                        conf, speed, license_plate, frame_count, 
                        screenshot_path, repeat_offender
                    )
                    
                    cv2.circle(output_frame, center, 15, (0, 0, 255), -1)
                    cv2.putText(output_frame, "RED LIGHT VIOLATION", (x1, y2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output_frame
    
    def _process_helmet_violations(self, frame, output_frame, person_detections, 
                                 current_detections, timestamp, frame_count):
        """Process helmet violations"""
        for person_bbox in person_detections:
            x1, y1, x2, y2 = person_bbox
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Find nearby motorcycle/bicycle
            nearby_vehicle = None
            nearby_bbox = None
            for detection in current_detections.values():
                if detection['type'] in ["motorbike", "bicycle"]:
                    v_center = detection['center']
                    distance = np.sqrt((person_center[0] - v_center[0])**2 + 
                                     (person_center[1] - v_center[1])**2)
                    if distance < Config.NEARBY_VEHICLE_DISTANCE:
                        nearby_vehicle = detection['type']
                        nearby_bbox = detection['bbox']
                        break
            
            if nearby_vehicle:
                has_helmet = self.detector.helmet_detector.detect_helmet(frame, person_bbox)
                
                if not has_helmet:
                    screenshot_path = self.detector.save_violation_screenshot(
                        frame, nearby_bbox, "no_helmet", timestamp)
                    self.detector.logger.log_violation(
                        timestamp, "no_helmet_violation", nearby_vehicle, 
                        0.8, 0, "", frame_count, screenshot_path, False
                    )
                    
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(output_frame, "NO HELMET", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        return output_frame
    
    def _draw_violation_line(self, output_frame):
        """Draw violation line on frame"""
        (x1, y1), (x2, y2) = self.detector.violation_line
        cv2.line(output_frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
