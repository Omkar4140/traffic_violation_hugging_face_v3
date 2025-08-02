import os
import pandas as pd
import shutil
from datetime import datetime
from config.settings import Config

class ViolationLogger:
    def __init__(self):
        self.violations_log = []
        self.csv_file = Config.CSV_LOG_FILE
        self.screenshot_dir = os.path.join(Config.TEMP_DIR, "violation_screenshots")
        
        # Ensure directories exist
        Config.ensure_data_dir()
        Config.ensure_temp_dir()
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Initialize CSV with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with proper headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            headers = [
                'timestamp', 'violation_type', 'vehicle_type', 'confidence',
                'speed', 'license_plate', 'frame_no', 'screenshot_path',
                'repeat_offender', 'screenshot_display'
            ]
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.csv_file, index=False)
    
    def log_violation(self, timestamp, violation_type, vehicle_type, confidence, 
                     speed, license_plate, frame_no, screenshot_path, repeat_offender):
        """Log a violation to memory with screenshot display handling"""
        
        # Copy screenshot to persistent location if it exists
        screenshot_display_path = ""
        if screenshot_path and os.path.exists(screenshot_path):
            # Create a unique filename
            timestamp_clean = timestamp.replace(':', '-').replace(' ', '_').replace('.', '_')
            screenshot_filename = f"{violation_type}_{timestamp_clean}_{frame_no}.jpg"
            persistent_screenshot_path = os.path.join(self.screenshot_dir, screenshot_filename)
            
            try:
                shutil.copy2(screenshot_path, persistent_screenshot_path)
                screenshot_display_path = persistent_screenshot_path
            except Exception as e:
                print(f"Error copying screenshot: {e}")
                screenshot_display_path = screenshot_path
        
        violation = {
            'timestamp': timestamp,
            'violation_type': violation_type,
            'vehicle_type': vehicle_type,
            'confidence': round(confidence, 3),
            'speed': round(speed, 1) if speed > 0 else 0,
            'license_plate': license_plate if license_plate else "N/A",
            'frame_no': frame_no,
            'screenshot_path': screenshot_path,
            'repeat_offender': repeat_offender,
            'screenshot_display': screenshot_display_path
        }
        self.violations_log.append(violation)
    
    def is_repeat_offender(self, license_plate):
        """Check if license plate has previous violations"""
        if not license_plate or license_plate == "" or license_plate == "N/A":
            return False
        
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                if 'license_plate' in df.columns:
                    # Check if license plate exists in previous records
                    existing_plates = df['license_plate'].astype(str).str.strip()
                    return license_plate.strip() in existing_plates.values
            return False
        except Exception as e:
            print(f"Error checking repeat offender: {e}")
            return False
    
    def save_violations_to_csv(self):
        """Save violations to CSV file with proper handling of existing data"""
        if not self.violations_log:
            return self.csv_file if os.path.exists(self.csv_file) else None
        
        try:
            new_df = pd.DataFrame(self.violations_log)
            
            # Load existing data if file exists
            if os.path.exists(self.csv_file):
                try:
                    existing_df = pd.read_csv(self.csv_file)
                    # Ensure both dataframes have the same columns
                    for col in new_df.columns:
                        if col not in existing_df.columns:
                            existing_df[col] = ""
                    for col in existing_df.columns:
                        if col not in new_df.columns:
                            new_df[col] = ""
                    
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                except Exception as e:
                    print(f"Error reading existing CSV: {e}")
                    combined_df = new_df
            else:
                combined_df = new_df
            
            # Sort by timestamp (most recent first)
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp', ascending=False)
            
            # Save to CSV
            combined_df.to_csv(self.csv_file, index=False)
            return self.csv_file
            
        except Exception as e:
            print(f"Error saving violations to CSV: {e}")
            return None
    
    def clear_violations_csv(self):
        """Clear all violations from CSV but keep the file structure"""
        try:
            # Keep only the headers
            headers = [
                'timestamp', 'violation_type', 'vehicle_type', 'confidence',
                'speed', 'license_plate', 'frame_no', 'screenshot_path',
                'repeat_offender', 'screenshot_display'
            ]
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.csv_file, index=False)
            
            # Clear in-memory log as well
            self.violations_log = []
            
            # Optionally clear screenshot directory
            try:
                if os.path.exists(self.screenshot_dir):
                    shutil.rmtree(self.screenshot_dir)
                    os.makedirs(self.screenshot_dir, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not clear screenshot directory: {e}")
            
            return True
        except Exception as e:
            print(f"Error clearing violations CSV: {e}")
            return False
    
    def get_violations_dataframe(self):
        """Get violations as pandas DataFrame for UI display"""
        if not self.violations_log:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.violations_log)
        
        # Reorder columns for better display
        display_columns = [
            'timestamp', 'violation_type', 'vehicle_type', 
            'license_plate', 'speed', 'confidence', 'repeat_offender',
            'frame_no', 'screenshot_display'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in display_columns if col in df.columns]
        df_display = df[available_columns].copy()
        
        # Format for better readability
        if 'confidence' in df_display.columns:
            df_display['confidence'] = df_display['confidence'].round(3)
        if 'speed' in df_display.columns:
            df_display['speed'] = df_display['speed'].apply(lambda x: f"{x:.1f} km/h" if x > 0 else "N/A")
        if 'repeat_offender' in df_display.columns:
            df_display['repeat_offender'] = df_display['repeat_offender'].apply(lambda x: "Yes" if x else "No")
        
        return df_display
    
    def get_csv_summary(self):
        """Get summary statistics from the CSV file"""
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                if len(df) > 0:
                    summary = {
                        'total_violations': len(df),
                        'violation_types': df['violation_type'].value_counts().to_dict() if 'violation_type' in df.columns else {},
                        'repeat_offenders': len(df[df['repeat_offender'] == True]) if 'repeat_offender' in df.columns else 0,
                        'latest_violation': df.iloc[0]['timestamp'] if 'timestamp' in df.columns else "N/A"
                    }
                    return summary
            return {'total_violations': 0, 'violation_types': {}, 'repeat_offenders': 0, 'latest_violation': 'N/A'}
        except Exception as e:
            print(f"Error getting CSV summary: {e}")
            return {'total_violations': 0, 'violation_types': {}, 'repeat_offenders': 0, 'latest_violation': 'N/A'}
