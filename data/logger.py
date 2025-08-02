import os
import pandas as pd
from config.settings import Config

class ViolationLogger:
    def __init__(self):
        self.violations_log = []
        self.csv_file = Config.CSV_LOG_FILE
    
    def log_violation(self, timestamp, violation_type, vehicle_type, confidence, 
                     speed, license_plate, frame_no, screenshot_path, repeat_offender):
        """Log a violation to memory"""
        violation = {
            'timestamp': timestamp,
            'violation_type': violation_type,
            'vehicle_type': vehicle_type,
            'confidence': confidence,
            'speed': speed,
            'license_plate': license_plate,
            'frame_no': frame_no,
            'screenshot_path': screenshot_path,
            'repeat_offender': repeat_offender
        }
        self.violations_log.append(violation)
    
    def is_repeat_offender(self, license_plate):
        """Check if license plate has previous violations"""
        if not license_plate or license_plate == "":
            return False
        
        if os.path.exists(self.csv_file):
            try:
                df = pd.read_csv(self.csv_file)
                return license_plate in df['license_plate'].values
            except Exception as e:
                print(f"Error checking repeat offender: {e}")
                return False
        return False
    
    def save_violations_to_csv(self):
        """Save violations to CSV file"""
        if not self.violations_log:
            return None
            
        new_df = pd.DataFrame(self.violations_log)
        
        if os.path.exists(self.csv_file):
            try:
                existing_df = pd.read_csv(self.csv_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing CSV: {e}")
                combined_df = new_df
        else:
            combined_df = new_df
        
        try:
            combined_df.to_csv(self.csv_file, index=False)
            return self.csv_file
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return None
    
    def get_violations_dataframe(self):
        """Get violations as pandas DataFrame"""
        return pd.DataFrame(self.violations_log) if self.violations_log else pd.DataFrame()
