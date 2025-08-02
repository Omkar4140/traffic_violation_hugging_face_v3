import numpy as np
from config.settings import Config

class SpeedCalculator:
    def __init__(self):
        self.vehicle_tracks = {}  # Store tracking history for better speed calculation
        self.frame_timestamps = {}  # Store frame timestamps for accurate time calculation
    
    def calculate_speed(self, vehicle_id, prev_center, curr_center, fps, frame_timestamp=None):
        """Calculate vehicle speed with improved accuracy"""
        if prev_center is None or curr_center is None:
            return 0
        
        try:
            # Calculate pixel distance
            pixel_distance = np.sqrt(
                (curr_center[0] - prev_center[0])**2 + 
                (curr_center[1] - prev_center[1])**2
            )
            
            # Ignore very small movements (likely detection noise)
            if pixel_distance < 5:
                return 0
            
            # Convert to real-world distance
            real_distance = pixel_distance * Config.PIXEL_TO_METER_RATIO
            
            # Calculate time difference
            if frame_timestamp and vehicle_id in self.frame_timestamps:
                # Use actual timestamp if available
                time_diff = frame_timestamp - self.frame_timestamps[vehicle_id]
                if time_diff <= 0:
                    return 0
            else:
                # Fallback to FPS-based calculation
                time_diff = 1.0 / fps if fps > 0 else 1.0
            
            # Store current timestamp
            if frame_timestamp:
                self.frame_timestamps[vehicle_id] = frame_timestamp
            
            # Calculate speed in m/s then convert to km/h
            speed_mps = real_distance / time_diff
            speed_kmh = speed_mps * 3.6
            
            # Apply smoothing using tracking history
            smoothed_speed = self._smooth_speed(vehicle_id, speed_kmh)
            
            # Clamp to reasonable values
            final_speed = max(0, min(Config.MAX_SPEED_KMH, smoothed_speed))
            
            # Only return speed if it's above minimum threshold
            return final_speed if final_speed > Config.MIN_SPEED_THRESHOLD else 0
            
        except Exception as e:
            print(f"Speed calculation error: {e}")
            return 0
    
    def _smooth_speed(self, vehicle_id, current_speed):
        """Apply smoothing to reduce speed calculation noise"""
        if vehicle_id not in self.vehicle_tracks:
            self.vehicle_tracks[vehicle_id] = []
        
        # Add current speed to history
        self.vehicle_tracks[vehicle_id].append(current_speed)
        
        # Keep only last 5 readings for smoothing
        if len(self.vehicle_tracks[vehicle_id]) > 5:
            self.vehicle_tracks[vehicle_id] = self.vehicle_tracks[vehicle_id][-5:]
        
        # Calculate moving average
        speeds = self.vehicle_tracks[vehicle_id]
        
        if len(speeds) == 1:
            return current_speed
        
        # Remove outliers (speeds that are too different from the median)
        median_speed = np.median(speeds)
        filtered_speeds = [s for s in speeds if abs(s - median_speed) < median_speed * 0.5]
        
        if not filtered_speeds:
            return current_speed
        
        return np.mean(filtered_speeds)
    
    def reset_vehicle_tracking(self, vehicle_id):
        """Reset tracking for a specific vehicle"""
        if vehicle_id in self.vehicle_tracks:
            del self.vehicle_tracks[vehicle_id]
        if vehicle_id in self.frame_timestamps:
            del self.frame_timestamps[vehicle_id]
    
    def clear_old_tracks(self, current_vehicle_ids):
        """Clear tracking data for vehicles no longer in scene"""
        # Remove tracks for vehicles not seen in current frame
        old_ids = set(self.vehicle_tracks.keys()) - set(current_vehicle_ids)
        for old_id in old_ids:
            self.reset_vehicle_tracking(old_id)
    
    @staticmethod
    def estimate_pixel_to_meter_ratio(frame_height, assumed_road_width_meters=10):
        """Estimate pixel to meter ratio based on frame size and assumed road width"""
        # This is a rough estimation - in practice, you'd calibrate this
        # Assume the road takes up about 60% of the frame width
        estimated_road_pixels = frame_height * 0.6
        return assumed_road_width_meters / estimated_road_pixels
