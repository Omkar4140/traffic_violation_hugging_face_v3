import cv2
import os
import numpy as np
from datetime import datetime, timedelta
from models.detection_models import ModelManager
from config.settings import Config

class VideoProcessor:
    def __init__(self, violation_detector):
        self.detector = violation_detector
        self.model_manager = ModelManager()
        self.vehicle_id_counter = 0
        self.active_vehicles = {}
    
    def process_video(self, video_path, enable_plate_detection=True):
        """Process video for violations with enhanced tracking and timeline markers"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, [], None
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Auto-detect violation line from first frame
        ret, first_frame = cap.read()
        if ret:
            auto_line = self.detector.auto_detect_violation_line(first_frame)
            if auto_line:
                print(f"Auto-detected violation line: {auto_line}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        # Output video setup
        output_path = os.path.join(Config.TEMP_DIR, "processed_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        yolo_model = self.model_manager.get_yolo_model()
        frame_count = 0
        violation_frames = []  # Store frame numbers with violations
        start_time = datetime.now()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate current timestamp
                current_time = start_time + timedelta(seconds=frame_count/fps)
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                output_frame = frame.copy()
                
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
                        
                        # Assign vehicle ID based on proximity to previous detections
                        vehicle_id = self._assign_vehicle_id(center, frame_count)
                        
                        current_detections[vehicle_id] = {
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
                has_violations = self._process_vehicles(
                    frame, output_frame, current_detections,
                    traffic_light_state, timestamp, frame_count, fps, enable_plate_detection
                )
                
                # Process helmet violations
                helmet_violations = self._process_helmet_violations(
                    frame, output_frame, person_detections, current_detections, timestamp, frame_count
                )
                
                # Track violation frames for timeline markers
                if has_violations or helmet_violations:
                    violation_frames.append(frame_count)
                
                # Draw violation line with enhanced visibility
                self._draw_violation_line(output_frame)
                
                # Add enhanced frame info
                self._add_frame_info(output_frame, frame_count, total_frames, traffic_light_state, 
                                   len(current_detections), len(violation_frames))
                
                # Add timeline markers for violations (enhanced visibility)
                if frame_count in violation_frames:
                    self._add_violation_marker(output_frame, width, height)
                
                out.write(output_frame)
                frame_count += 1
                
                # Progress indicator
                if frame_count % 30 == 0:  # Every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing: {progress:.1f}% ({frame_count}/{total_frames})")
                
        finally:
            cap.release()
            out.release()
            
            # Clean up old vehicle tracks
            current_vehicle_ids = list(current_detections.keys()) if 'current_detections' in locals() else []
            self.detector.speed_calculator.clear_old_tracks(current_vehicle_ids)
        
        print(f"Video processing complete. Found {len(violation_frames)} violation frames.")
        return output_path, self.detector.logger.get_violations_dataframe(), violation_frames
    
    def _assign_vehicle_id(self, center, frame_count):
        """Assign vehicle ID based on proximity to existing vehicles"""
        min_distance = float('inf')
        assigned_id = None
        
        # Check proximity to existing vehicles
        for vehicle_id, vehicle_data in self.active_vehicles.items():
            if frame_count - vehicle_data['last_seen'] < 10:  # Vehicle seen in last 10 frames
                distance = np.sqrt((center[0] - vehicle_data['center'][0])**2 + 
                                 (center[1] - vehicle_data['center'][1])**2)
                if distance < min_distance and distance < 50:  # Within 50 pixels
                    min_distance = distance
                    assigned_id = vehicle_id
        
        # Create new vehicle ID if no match found
        if assigned_id is None:
            assigned_id = self.vehicle_id_counter
            self.vehicle_id_counter += 1
        
        # Update vehicle tracking
        self.active_vehicles[assigned_id] = {
            'center': center,
            'last_seen': frame_count
        }
        
        return assigned_id
    
    def _detect_traffic_light_state(self, frame, traffic_lights):
        """Detect traffic light state"""
        for light_bbox in traffic_lights:
            state = self.detector.traffic_light_detector.detect_color(frame, light_bbox)
            if state != "unknown":
                return state
        return "unknown"
    
    def _process_vehicles(self, frame, output_frame, current_detections,
                         traffic_light_state, timestamp, frame_count, fps, enable_plate_detection):
        """Process vehicle detections and violations"""
        has_violations = False
        
        for vehicle_id, detection in current_detections.items():
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
            
            # Speed calculation with vehicle ID tracking
            speed = 0
            if vehicle_id in self.active_vehicles:
                prev_center = self.active_vehicles[vehicle_id].get('prev_center')
                if prev_center:
                    speed = self.detector.speed_calculator.calculate_speed(
                        vehicle_id, prev_center, center, fps, frame_timestamp=datetime.now().timestamp()
                    )
                    
                    # Check for speeding violation
                    if speed > Config.SPEED_LIMIT_KMH:
                        screenshot_path = self.detector.save_violation_screenshot(
                            frame, bbox, "speeding", timestamp)
                        self.detector.logger.log_violation(
                            timestamp, "speeding_violation", vehicle_type, 
                            conf, speed, license_plate, frame_count, 
                            screenshot_path, repeat_offender
                        )
                        has_violations = True
                        
                        # Enhanced speeding violation display
                        cv2.putText(output_frame, f"SPEEDING: {speed:.1f} km/h", (x1, y2 + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
                        cv2.rectangle(output_frame, (x1-5, y1-5), (x2+5, y2+5), (255, 0, 0), 3)
            
            # Update vehicle tracking
            if vehicle_id in self.active_vehicles:
                self.active_vehicles[vehicle_id]['prev_center'] = center
            
            # Draw bounding box with enhanced colors
            color = (0, 0, 255) if repeat_offender else (0, 255, 0)
            thickness = 3 if repeat_offender else 2
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Create enhanced label
            label = f"{vehicle_type} {conf:.2f}"
            if speed > 0:
                label += f" {speed:.1f}km/h"
            if license_plate:
                label += f" [{license_plate}]"
            if repeat_offender:
                label += " [REPEAT]"
            
            # Draw label with background for better visibility
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output_frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
            cv2.putText(output_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
                    has_violations = True
                    
                    # Enhanced red light violation display
                    cv2.circle(output_frame, center, 20, (0, 0, 255), -1)
                    cv2.putText(output_frame, "RED LIGHT VIOLATION", (x1, y2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        return has_violations
    
    def _process_helmet_violations(self, frame, output_frame, person_detections, 
                                 current_detections, timestamp, frame_count):
        """Process helmet violations"""
        has_violations = False
        
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
                    has_violations = True
                    
                    # Enhanced helmet violation display
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                    cv2.putText(output_frame, "NO HELMET", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 3)
        
        return has_violations
    
    def _draw_violation_line(self, output_frame):
        """Draw violation line with enhanced visibility"""
        line = self.detector.get_violation_line_for_display()
        if line:
            (x1, y1), (x2, y2) = line
            # Draw thicker line with multiple colors for better visibility
            cv2.line(output_frame, (x1, y1), (x2, y2), (0, 0, 0), 7)  # Black outline
            cv2.line(output_frame, (x1, y1), (x2, y2), (255, 255, 0), 5)  # Yellow line
            cv2.line(output_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)  # White center
            
            # Add label with background
            label = "VIOLATION LINE"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_x = x1
            label_y = y1 - 20
            cv2.rectangle(output_frame, (label_x, label_y - 20), 
                         (label_x + label_size[0], label_y), (0, 0, 0), -1)
            cv2.putText(output_frame, label, (label_x, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def _add_frame_info(self, output_frame, frame_count, total_frames, traffic_light_state, 
                       vehicle_count, violation_count):
        """Add enhanced frame information"""
        height, width = output_frame.shape[:2]
        
        # Create info panel background
        info_height = 80
        cv2.rectangle(output_frame, (0, 0), (width, info_height), (0, 0, 0), -1)
        cv2.rectangle(output_frame, (0, 0), (width, info_height), (255, 255, 255), 2)
        
        # Add information text
        info_lines = [
            f"Frame: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)",
            f"Traffic Light: {traffic_light_state.upper()} | Vehicles: {vehicle_count} | Violations: {violation_count}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(output_frame, line, (10, 25 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _add_violation_marker(self, output_frame, width, height):
        """Add enhanced violation marker for timeline visibility"""
        # Add bright red border around entire frame
        cv2.rectangle(output_frame, (0, 0), (width-1, height-1), (0, 0, 255), 8)
        
        # Add pulsing violation indicator in top-right corner
        marker_size = 30
        cv2.circle(output_frame, (width - 40, 40), marker_size, (0, 0, 255), -1)
        cv2.circle(output_frame, (width - 40, 40), marker_size//2, (255, 255, 255), -1)
        cv2.putText(output_frame, "!", (width - 47, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
