import numpy as np
from config.settings import Config

class SpeedCalculator:
    def __init__(self):
        # Initialize tracking dictionaries for enhanced speed calculation
        self.vehicle_tracks = {}
        self.frame_timestamps = {}
    
    def calculate_speed(self, prev_center, curr_center, fps):
        """Calculate vehicle speed (backward compatible method)"""
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
            time_diff = 1.0 / fps if fps > 0 else 1.0
            
            # Calculate speed in m/s then convert to km/h
            speed_mps = real_distance / time_diff
            speed_kmh = speed_mps * 3.6
            
            # Apply reasonable limits
            final_speed = max(0, min(Config.MAX_SPEED_KMH, speed_kmh))
            
            # Only return speed if it's above minimum threshold
            return final_speed if final_speed > getattr(Config, 'MIN_SPEED_THRESHOLD', 5) else 0
            
        except Exception as e:
            print(f"Speed calculation error: {e}")
            return 0
    
    # Enhanced method for vehicle tracking (optional, backward compatible)
    def calculate_speed_enhanced(self, vehicle_id, prev_center, curr_center, fps, frame_timestamp=None):
        """Enhanced speed calculation with vehicle tracking"""
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
                time_diff = frame_timestamp - self.frame_timestamps[vehicle_id]
                if time_diff <= 0:
                    return 0
            else:
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
            return final_speed if final_speed > getattr(Config, 'MIN_SPEED_THRESHOLD', 5) else 0
            
        except Exception as e:
            print(f"Enhanced speed calculation error: {e}")
            return 0
    
    def _smooth_speed(self, vehicle_id, current_speed):
        """Apply smoothing to reduce speed calculation noise"""
        try:
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
        except Exception as e:
            print(f"Error in speed smoothing: {e}")
            return current_speed
    
    def reset_vehicle_tracking(self, vehicle_id):
        """Reset tracking for a specific vehicle"""
        try:
            if vehicle_id in self.vehicle_tracks:
                del self.vehicle_tracks[vehicle_id]
            if vehicle_id in self.frame_timestamps:
                del self.frame_timestamps[vehicle_id]
        except Exception as e:
            print(f"Error resetting vehicle tracking: {e}")
    
    def clear_old_tracks(self, current_vehicle_ids):
        """Clear tracking data for vehicles no longer in scene"""
        try:
            # Remove tracks for vehicles not seen in current frame
            old_ids = set(self.vehicle_tracks.keys()) - set(current_vehicle_ids)
            for old_id in old_ids:
                self.reset_vehicle_tracking(old_id)
        except Exception as e:
            print(f"Error clearing old tracks: {e}")
