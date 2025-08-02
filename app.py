import gradio as gr
from config.settings import Config
from core.violation_detector import ViolationDetector
from core.image_processor import ImageProcessor
from core.video_processor import VideoProcessor
from ui.interface import create_interface
from models.detection_models import ModelManager

def main():
    """Main application entry point with enhanced features"""
    print("🚦 Starting Enhanced Traffic Violation Detection System...")
    
    # Ensure all directories exist
    print("📁 Setting up directories...")
    Config.ensure_temp_dir()
    Config.ensure_data_dir()
    
    # Initialize model manager and load models
    print("📥 Loading AI models...")
    model_manager = ModelManager()
    model_manager.load_models()
    
    # Initialize core components
    print("🔧 Initializing enhanced components...")
    detector = ViolationDetector()
    image_processor = ImageProcessor(detector)
    video_processor = VideoProcessor(detector)
    
    print("🌐 Creating enhanced web interface...")
    # Create and launch interface
    iface = create_interface(image_processor, video_processor)
    
    print("🚀 Launching enhanced application...")
    print("✨ New Features:")
    print("   - Automatic violation line detection")
    print("   - Enhanced license plate recognition")
    print("   - Fixed speed calculation")
    print("   - Improved screenshot display")
    print("   - CSV management with clear functionality")
    print("   - Enhanced timeline markers for videos")
    
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main()
