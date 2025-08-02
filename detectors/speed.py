import numpy as np
from config.settings import Config

class SpeedCalculator:
    @staticmethod
    def calculate_speed(prev_center, curr_center, fps):
        """Calculate vehicle speed based on center movement"""
        if prev_center is None:
            return 0
        
        pixel_distance = np.sqrt(
            (curr_center[0] - prev_center[0])**2 + 
            (curr_center[1] - prev_center[1])**2
        )
        
        real_distance = pixel_distance * Config.PIXEL_TO_METER_RATIO
        speed_mps = real_distance * fps
        speed_kmh = speed_mps * 3.6
        
        return max(0, min(Config.MAX_SPEED_KMH, speed_kmh))
