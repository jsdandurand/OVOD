#!/usr/bin/env python3
"""
Startup script for ClipTracker Real-time Webcam Application
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_requirements():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import websockets
        from ml.clip_detection import ClipDetector
        from ml.clip_segmentation import ClipSegmenter
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_files():
    """Check if required files exist."""
    required_files = [
        "app/app_server.py", 
        "app/app_frontend.html",
        "ml/clip_detection.py",
        "ml/clip_segmentation.py"
    ]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files found")
    return True

def main():
    """Start the ClipTracker webcam application."""
    print("🎯 ClipTracker Real-time Webcam Application")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check files
    if not check_files():
        sys.exit(1)
    
    print("\n🚀 Starting server...")
    print("📹 Make sure your webcam is connected and working")
    print("🌐 The application will open in your browser automatically")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start server
        import uvicorn
        
        # Open browser after short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:8000")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run server
        uvicorn.run(
            "app.app_server:app", 
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            reload=False
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Thank you for using ClipTracker!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("Make sure port 8000 is available and try again.")

if __name__ == "__main__":
    main() 