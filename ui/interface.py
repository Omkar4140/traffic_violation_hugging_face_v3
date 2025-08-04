import gradio as gr
import pandas as pd
import os
from data.dashboard import ViolationDashboard

class TrafficViolationInterface:
    def __init__(self, image_processor, video_processor):
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.dashboard = ViolationDashboard()
    
    def process_image(self, image_file, line_coordinates, enable_plate_detection):
        """Process uploaded image with enhanced features"""
        if image_file is None:
            return None, pd.DataFrame(), None, None, "Please upload an image first."
        
        try:
            # Parse and set violation line (manual override)
            line_coords = self.image_processor.detector.parse_line_coordinates(line_coordinates)
            if line_coords:
                self.image_processor.detector.set_violation_line(line_coords[0], line_coords[1])
                status_msg = f"✅ Using manual violation line: {line_coords}"
            else:
                # Try auto-detection
                import cv2
                frame = cv2.imread(image_file)
                if frame is not None:
                    auto_line = self.image_processor.detector.auto_detect_violation_line(frame)
                    if auto_line:
                        status_msg = f"🤖 Auto-detected violation line: {auto_line}"
                    else:
                        status_msg = "⚠️ No violation line detected. Red light detection disabled."
                else:
                    status_msg = "❌ Error reading image file."
            
            # Process image
            output_path, violations_df, _ = self.image_processor.process_image(
                image_file, enable_plate_detection
            )
            
            # Create dashboard and save CSV
            dashboard_path = None
            csv_path = None
            if not violations_df.empty:
                try:
                    dashboard_path = self.dashboard.create_dashboard(
                        self.image_processor.detector.logger.violations_log
                    )
                    csv_path = self.image_processor.detector.logger.save_violations_to_csv()
                    status_msg += f" | 🚨 Found {len(violations_df)} violations."
                except Exception as e:
                    print(f"Error creating dashboard: {e}")
                    status_msg += f" | Found {len(violations_df)} violations (dashboard error)."
            else:
                status_msg += " | ✅ No violations detected."
            
            return output_path, violations_df, dashboard_path, csv_path, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error processing image: {str(e)}"
            print(f"Image processing error: {e}")
            return None, pd.DataFrame(), None, None, error_msg
    
    def process_video(self, video_file, line_coordinates, enable_plate_detection):
        """Process uploaded video with enhanced features"""
        if video_file is None:
            return None, pd.DataFrame(), None, None, "Please upload a video first."
        
        try:
            # Parse and set violation line (manual override)
            line_coords = self.video_processor.detector.parse_line_coordinates(line_coordinates)
            if line_coords:
                self.video_processor.detector.set_violation_line(line_coords[0], line_coords[1])
                status_msg = f"✅ Using manual violation line: {line_coords}"
            else:
                status_msg = "🤖 Using auto-detected violation line (if found)."
            
            # Process video
            output_path, violations_df, violation_frames = self.video_processor.process_video(
                video_file, enable_plate_detection
            )
            
            # Create dashboard and save CSV
            dashboard_path = None
            csv_path = None
            if not violations_df.empty:
                try:
                    dashboard_path = self.dashboard.create_dashboard(
                        self.video_processor.detector.logger.violations_log
                    )
                    csv_path = self.video_processor.detector.logger.save_violations_to_csv()
                    frame_count = len(violation_frames) if violation_frames else 0
                    status_msg += f" | 🚨 Found {len(violations_df)} violations in {frame_count} frames."
                except Exception as e:
                    print(f"Error creating dashboard: {e}")
                    status_msg += f" | Found {len(violations_df)} violations (dashboard error)."
            else:
                status_msg += " | ✅ No violations detected."
            
            return output_path, violations_df, dashboard_path, csv_path, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error processing video: {str(e)}"
            print(f"Video processing error: {e}")
            return None, pd.DataFrame(), None, None, error_msg
    
    def clear_violation_logs(self):
        """Clear all violation logs"""
        try:
            # Clear from both processors
            success1 = self.image_processor.detector.clear_violation_logs()
            success2 = self.video_processor.detector.clear_violation_logs()
            
            if success1 and success2:
                return "✅ Violation logs cleared successfully!", pd.DataFrame()
            else:
                return "⚠️ Warning: Some logs may not have been cleared.", pd.DataFrame()
        except Exception as e:
            return f"❌ Error clearing logs: {str(e)}", pd.DataFrame()
    
    def get_csv_summary(self):
        """Get summary of violation logs"""
        try:
            summary = self.image_processor.detector.logger.get_csv_summary()
            summary_text = f"""
📊 **Violation Log Summary:**
- Total Violations: {summary['total_violations']}
- Repeat Offenders: {summary['repeat_offenders']}
- Latest Violation: {summary['latest_violation']}
**Violation Types:**
"""
            for vtype, count in summary['violation_types'].items():
                summary_text += f"\n- {vtype.replace('_', ' ').title()}: {count}"
            
            if summary['total_violations'] == 0:
                summary_text = "📊 **No violations recorded yet.**"
            
            return summary_text
        except Exception as e:
            return f"❌ Error getting summary: {str(e)}"

def create_interface(image_processor, video_processor):
    """Create enhanced Gradio interface"""
    interface = TrafficViolationInterface(image_processor, video_processor)
    
    with gr.Blocks(title="Enhanced Traffic Violation Detection System", theme=gr.themes.Soft()) as iface:
        
        gr.Markdown("# 🚦 Enhanced Traffic Violation Detection System")
        gr.Markdown("Upload images or videos to detect traffic violations with automatic violation line detection")
        
        # Add summary section at the top
        with gr.Row():
            summary_btn = gr.Button("📊 Get Log Summary", variant="secondary", size="sm")
            summary_output = gr.Markdown()
        
        with gr.Tabs():
            with gr.TabItem("📸 Image Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="Upload Traffic Image", type="filepath")
                        
                        with gr.Group():
                            gr.Markdown("### ⚙️ Violation Line Setup")
                            gr.Markdown("**Auto-detection enabled by default.** Manual input overrides auto-detection.")
                            line_input = gr.Textbox(
                                label="Manual Violation Line (Optional)", 
                                placeholder="[(x1,y1), (x2,y2)] - Leave empty for auto-detection",
                                info="Example: [(100,300), (500,300)] for horizontal line"
                            )
                        
                        with gr.Group():
                            gr.Markdown("### 🔧 Detection Options")
                            enable_plates = gr.Checkbox(
                                label="Enable Enhanced License Plate Detection", 
                                value=True,
                                info="Improved OCR for Indian license plates"
                            )
                        
                        process_img_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                        
                        # Status message
                        img_status = gr.Markdown("Ready to process images...")
                    
                    with gr.Column(scale=2):
                        output_image = gr.Image(label="Processed Image with Detections")
                        
                        with gr.Row():
                            with gr.Column():
                                violations_table = gr.Dataframe(
                                    label="🚨 Detected Violations",
                                    wrap=True,
                                    interactive=False
                                )
                                csv_download = gr.File(label="📥 Download Complete Log (CSV)")
                            with gr.Column():
                                dashboard_chart = gr.Image(label="📊 Violation Dashboard")
            
            with gr.TabItem("🎥 Video Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Traffic Video")
                        
                        with gr.Group():
                            gr.Markdown("### ⚙️ Violation Line Setup")
                            gr.Markdown("**Auto-detection enabled by default.** Manual input overrides auto-detection.")
                            line_input_video = gr.Textbox(
                                label="Manual Violation Line (Optional)", 
                                placeholder="[(x1,y1), (x2,y2)] - Leave empty for auto-detection",
                                info="Example: [(100,300), (500,300)] for horizontal line"
                            )
                        
                        with gr.Group():
                            gr.Markdown("### 🔧 Detection Options")
                            enable_plates_video = gr.Checkbox(
                                label="Enable Enhanced License Plate Detection", 
                                value=True,
                                info="Improved OCR for Indian license plates"
                            )
                        
                        process_vid_btn = gr.Button("🎬 Analyze Video", variant="primary", size="lg")
                        
                        # Status message
                        vid_status = gr.Markdown("Ready to process videos...")
                    
                    with gr.Column(scale=2):
                        output_video = gr.Video(
                            label="Processed Video with Enhanced Markers",
                            show_download_button=True
                        )
                        
                        with gr.Row():
                            with gr.Column():
                                violations_table_video = gr.Dataframe(
                                    label="🚨 Detected Violations",
                                    wrap=True,
                                    interactive=False
                                )
                                csv_download_video = gr.File(label="📥 Download Complete Log (CSV)")
                            with gr.Column():
                                dashboard_chart_video = gr.Image(label="📊 Violation Dashboard")
        
        # Management section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🗂️ Log Management")
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear All Violation Logs", variant="stop", size="sm")
                    refresh_btn = gr.Button("🔄 Refresh Summary", variant="secondary", size="sm")
                
                clear_status = gr.Markdown()
        
        # Enhanced feature description
        with gr.Accordion("📋 Enhanced Features & Usage Guide", open=False):
            gr.Markdown("""
            ### 🚀 **Enhanced Features:**
            
            #### 🤖 **Automatic Violation Line Detection**
            - Automatically detects zebra crossings and road markings
            - Creates violation line before crossings for red light detection
            - Manual input available as override option
            
            #### 🔍 **Improved License Plate Detection**
            - Enhanced OCR with preprocessing for Indian license plates
            - Better handling of varied fonts, backgrounds, and lighting
            - Supports multiple Indian license plate formats
            
            #### 🚗 **Fixed Speed Detection**
            - Accurate speed calculation with configurable limits
            - Eliminates false speeding violations
            - Default speed limit: 40 km/h
            
            #### 📊 **Enhanced Management**
            - Screenshots automatically saved and linked
            - Repeat offender detection based on license plates
            - CSV logs with complete violation history
            - Clear logs functionality
            
            ### 📖 **Usage Instructions:**
            
            1. **Upload** your traffic image or video
            2. **Violation Line**: Leave empty for auto-detection, or specify coordinates manually
            3. **Enable** license plate detection for complete logging
            4. **Process** and review detected violations
            5. **Download** CSV logs or clear history as needed
            
            ### 🔧 **Detection Types:**
            - **Red Light Violations**: Vehicles crossing violation line during red light
            - **No Helmet Violations**: Motorcycle/bicycle riders without helmets  
            - **Speeding Violations**: Vehicles exceeding speed limit
            - **Repeat Offender Tracking**: Based on license plate history
            
            ### 📝 **Line Format Examples:**
            - Horizontal: `[(50, 300), (650, 300)]`
            - Vertical: `[(400, 100), (400, 500)]`
            - Diagonal: `[(100, 200), (500, 400)]`
            """)
        
        # Event handlers
        process_img_btn.click(
            interface.process_image,
            inputs=[image_input, line_input, enable_plates],
            outputs=[output_image, violations_table, dashboard_chart, csv_download, img_status]
        )
        
        process_vid_btn.click(
            interface.process_video,
            inputs=[video_input, line_input_video, enable_plates_video],
            outputs=[output_video, violations_table_video, dashboard_chart_video, csv_download_video, vid_status]
        )
        
        clear_btn.click(
            interface.clear_violation_logs,
            outputs=[clear_status, violations_table]
        )
        
        summary_btn.click(
            interface.get_csv_summary,
            outputs=[summary_output]
        )
        
        refresh_btn.click(
            interface.get_csv_summary,
            outputs=[summary_output]
        )
    
    return iface
