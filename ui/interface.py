import gradio as gr
import pandas as pd
from data.dashboard import ViolationDashboard

class TrafficViolationInterface:
    def __init__(self, image_processor, video_processor):
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.dashboard = ViolationDashboard()
    
    def process_image(self, image_file, line_coordinates, enable_plate_detection):
        """Process uploaded image"""
        if image_file is None:
            return None, pd.DataFrame(), None, None
        
        # Parse and set violation line
        line_coords = self.image_processor.detector.parse_line_coordinates(line_coordinates)
        if line_coords:
            self.image_processor.detector.set_violation_line(line_coords[0], line_coords[1])
        
        # Process image
        output_path, violations_df, _ = self.image_processor.process_image(
            image_file, enable_plate_detection
        )
        
        # Create dashboard
        dashboard_path = None
        csv_path = None
        if not violations_df.empty:
            dashboard_path = self.dashboard.create_dashboard(
                self.image_processor.detector.logger.violations_log
            )
            csv_path = self.image_processor.detector.logger.save_violations_to_csv()
        
        return output_path, violations_df, dashboard_path, csv_path
    
    def process_video(self, video_file, line_coordinates, enable_plate_detection):
        """Process uploaded video"""
        if video_file is None:
            return None, pd.DataFrame(), None, None
        
        # Parse and set violation line
        line_coords = self.video_processor.detector.parse_line_coordinates(line_coordinates)
        if line_coords:
            self.video_processor.detector.set_violation_line(line_coords[0], line_coords[1])
        
        # Process video
        output_path, violations_df, _ = self.video_processor.process_video(
            video_file, enable_plate_detection
        )
        
        # Create dashboard
        dashboard_path = None
        csv_path = None
        if not violations_df.empty:
            dashboard_path = self.dashboard.create_dashboard(
                self.video_processor.detector.logger.violations_log
            )
            csv_path = self.video_processor.detector.logger.save_violations_to_csv()
        
        return output_path, violations_df, dashboard_path, csv_path

def create_interface(image_processor, video_processor):
    """Create Gradio interface"""
    interface = TrafficViolationInterface(image_processor, video_processor)
    
    with gr.Blocks(title="Traffic Violation Detection System", theme=gr.themes.Soft()) as iface:
        gr.Markdown("# 🚦 Enhanced Traffic Violation Detection System")
        gr.Markdown("Upload images or videos to detect traffic violations with dynamic violation line setup")
        
        with gr.Tabs():
            with gr.TabItem("📸 Image Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(label="Upload Traffic Image", type="filepath")
                        
                        with gr.Group():
                            gr.Markdown("### Violation Line Setup")
                            line_input = gr.Textbox(
                                label="Violation Line Coordinates", 
                                placeholder="[(x1,y1), (x2,y2)] - Two points defining the line",
                                info="Example: [(100,300), (500,300)] for horizontal line"
                            )
                        
                        with gr.Group():
                            gr.Markdown("### Detection Options")
                            enable_plates = gr.Checkbox(label="Enable License Plate Detection", value=True)
                        
                        process_img_btn = gr.Button("🔍 Analyze Image", variant="primary")
                    
                    with gr.Column(scale=2):
                        output_image = gr.Image(label="Processed Image")
                        
                        with gr.Row():
                            with gr.Column():
                                violations_table = gr.Dataframe(label="Detected Violations")
                                csv_download = gr.File(label="Download Complete Log (CSV)")
                            with gr.Column():
                                dashboard_chart = gr.Image(label="Violation Dashboard")
            
            with gr.TabItem("🎥 Video Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Traffic Video")
                        
                        with gr.Group():
                            gr.Markdown("### Violation Line Setup")
                            line_input_video = gr.Textbox(
                                label="Violation Line Coordinates", 
                                placeholder="[(x1,y1), (x2,y2)] - Two points defining the line",
                                info="Example: [(100,300), (500,300)] for horizontal line"
                            )
                        
                        with gr.Group():
                            gr.Markdown("### Detection Options")
                            enable_plates_video = gr.Checkbox(label="Enable License Plate Detection", value=True)
                        
                        process_vid_btn = gr.Button("🎬 Analyze Video", variant="primary")
                    
                    with gr.Column(scale=2):
                        output_video = gr.Video(label="Processed Video")
                        
                        with gr.Row():
                            with gr.Column():
                                violations_table_video = gr.Dataframe(label="Detected Violations")
                                csv_download_video = gr.File(label="Download Complete Log (CSV)")
                            with gr.Column():
                                dashboard_chart_video = gr.Image(label="Violation Dashboard")
        
        # Feature description
        with gr.Row():
            gr.Markdown("""
            ### 📋 Key Features:
            1. **Dynamic Violation Line**: Set custom line coordinates to detect red light violations
            2. **CSV Logging**: All violations appended to `violation_log.csv` with history preservation
            3. **Repeat Offender Detection**: License plate matching against previous violations
            4. **Screenshot Capture**: Cropped images saved for each violation
            5. **Dashboard**: Bar chart and pie chart showing violation distribution
            6. **Detections**: Red light violations, No helmet violations, Speeding (>60 km/h)
            
            **Line Format**: Use two points: `[(x1,y1), (x2,y2)]` to define violation line before zebra crossing
            """)
        
        # Event handlers
        process_img_btn.click(
            interface.process_image,
            inputs=[image_input, line_input, enable_plates],
            outputs=[output_image, violations_table, dashboard_chart, csv_download]
        )
        
        process_vid_btn.click(
            interface.process_video,
            inputs=[video_input, line_input_video, enable_plates_video],
            outputs=[output_video, violations_table_video, dashboard_chart_video, csv_download_video]
        )
    
    return iface
