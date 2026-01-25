#!/usr/bin/env python
"""
Stream Analyzer for 3D Print Failure Detection

Monitors an RTSP camera feed, periodically sends frames to the ML API
for failure detection, and can trigger a printer stop command when
failures are detected.
"""

import cv2
import requests
import numpy as np
import base64
import time
import os
import sys
import logging
import threading
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class PrinterController:
    """Handles communication with the printer to stop prints"""
    
    def __init__(self, webhook_url: str, api_key: str | None = None):
        self.webhook_url = webhook_url
        self.api_key = api_key
    
    def stop_print(self, reason: str = "Failure detected") -> bool:
        """
        Send stop command to the printer.
        
        Supports:
        - OctoPrint: POST /api/job with command=cancel
        - Moonraker/Klipper: POST /printer/emergency_stop
        - Generic webhook: POST with JSON body
        """
        if not self.webhook_url:
            logger.warning("No printer webhook configured - cannot stop print")
            return False
        
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['X-Api-Key'] = self.api_key
        
        try:
            # Detect printer type and send appropriate command
            if '/api/job' in self.webhook_url:
                # OctoPrint
                response = requests.post(
                    self.webhook_url,
                    json={'command': 'cancel'},
                    headers=headers,
                    timeout=10
                )
            elif 'emergency_stop' in self.webhook_url:
                # Moonraker/Klipper
                response = requests.post(
                    self.webhook_url,
                    headers=headers,
                    timeout=10
                )
            else:
                # Generic webhook
                response = requests.post(
                    self.webhook_url,
                    json={'action': 'stop', 'reason': reason},
                    headers=headers,
                    timeout=10
                )
            
            if response.ok:
                logger.info(f"Successfully sent stop command to printer")
                return True
            else:
                logger.error(f"Failed to stop printer: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending stop command: {e}")
            return False


class StreamAnalyzer:
    """Analyzes RTSP stream for 3D print failures"""
    
    def __init__(
        self,
        rtsp_url: str,
        ml_api_url: str,
        detection_threshold: float = 0.40,
        analysis_interval: float = 10.0,
        consecutive_failures_to_stop: int = 3,
        printer_controller: Optional[PrinterController] = None,
        snapshot_dir: str = "/app/snapshots",
        max_snapshot_size_mb: float = 200.0
    ):
        self.rtsp_url = rtsp_url
        self.ml_api_url = ml_api_url.rstrip('/')
        self.detection_threshold = detection_threshold
        self.analysis_interval = analysis_interval
        self.consecutive_failures_to_stop = consecutive_failures_to_stop
        self.printer_controller = printer_controller
        self.snapshot_dir = snapshot_dir
        self.max_snapshot_size_bytes = int(max_snapshot_size_mb * 1024 * 1024)
        
        self.consecutive_failures = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        
        # Frame buffer for low-latency access (keeps latest 5 frames)
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=5)
        self.frame_lock = threading.Lock()
        self.grabber_thread: Optional[threading.Thread] = None
        self.grabber_running = False
        
        # Ensure snapshot directory exists
        os.makedirs(snapshot_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to the RTSP stream"""
        logger.info(f"Connecting to RTSP stream...")
        
        # Configure OpenCV for better RTSP handling
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Set buffer size to minimum
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            logger.error("Failed to connect to RTSP stream")
            return False
        
        logger.info("Successfully connected to RTSP stream")
        
        # Start background frame grabber thread
        self._start_grabber()
        
        # Wait for buffer to have at least one frame (up to 5 seconds)
        for _ in range(50):
            with self.frame_lock:
                if self.frame_buffer:
                    logger.info("Frame buffer ready")
                    return True
            time.sleep(0.1)
        
        logger.warning("Timeout waiting for first frame")
        return False
    
    def _start_grabber(self) -> None:
        """Start the background frame grabber thread"""
        if self.grabber_thread is not None and self.grabber_thread.is_alive():
            return
        
        self.grabber_running = True
        self.grabber_thread = threading.Thread(target=self._grabber_loop, daemon=True)
        self.grabber_thread.start()
        logger.info("Started background frame grabber thread")
    
    def _stop_grabber(self) -> None:
        """Stop the background frame grabber thread"""
        self.grabber_running = False
        if self.grabber_thread is not None:
            self.grabber_thread.join(timeout=2.0)
            self.grabber_thread = None
    
    def _grabber_loop(self) -> None:
        """Continuously grab frames to keep buffer fresh with latest frames"""
        while self.grabber_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame_buffer.append(frame)
            else:
                # Brief pause on read failure to avoid busy loop
                time.sleep(0.01)
    
    def grab_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the buffer"""
        with self.frame_lock:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()
        return None
    
    def analyze_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Send frame to ML API for analysis"""
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            # Send to ML API
            response = requests.post(
                f"{self.ml_api_url}/p/",
                json={'image': img_base64},
                timeout=30
            )
            
            if response.ok:
                return response.json().get('detections', [])
            else:
                logger.error(f"ML API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return []
    
    def save_snapshot(self, frame: np.ndarray, detections: List[Dict], prefix: str = "detection"):
        """Save a snapshot with detection annotations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(self.snapshot_dir, filename)
        
        # Draw detections on frame
        annotated = frame.copy()
        for det in detections:
            if det['confidence'] >= self.detection_threshold:
                box = det['box']
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Draw label
                label = f"{det['label']}: {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imwrite(filepath, annotated)
        logger.info(f"Saved snapshot: {filepath}")
        
        # Cleanup old snapshots if over size limit
        self.cleanup_snapshots()
        
        return filepath
    
    def cleanup_snapshots(self) -> None:
        """Remove oldest snapshots to keep total size under max_snapshot_size_bytes"""
        try:
            # Get all jpg files with their stats
            snapshots = []
            for f in os.listdir(self.snapshot_dir):
                if f.endswith('.jpg'):
                    filepath = os.path.join(self.snapshot_dir, f)
                    stat = os.stat(filepath)
                    snapshots.append((filepath, stat.st_mtime, stat.st_size))
            
            # Sort by modification time (oldest first)
            snapshots.sort(key=lambda x: x[1])
            
            # Calculate total size
            total_size = sum(s[2] for s in snapshots)
            
            # Remove oldest files until under limit
            while total_size > self.max_snapshot_size_bytes and snapshots:
                oldest_file, _, file_size = snapshots.pop(0)
                try:
                    os.remove(oldest_file)
                    total_size -= file_size
                    logger.debug(f"Removed old snapshot: {oldest_file}")
                except OSError as e:
                    logger.warning(f"Failed to remove snapshot {oldest_file}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error cleaning up snapshots: {e}")
    
    def process_detections(self, frame: np.ndarray, detections: List[Dict]) -> bool:
        """
        Process detection results.
        
        Returns True if print should be stopped.
        """
        # Filter detections above threshold
        high_confidence = [d for d in detections if d['confidence'] >= self.detection_threshold]
        
        if high_confidence:
            self.consecutive_failures += 1
            max_confidence = max(d['confidence'] for d in high_confidence)
            
            logger.warning(
                f"🚨 Failure detected! Confidence: {max_confidence:.2f}, "
                f"Consecutive: {self.consecutive_failures}/{self.consecutive_failures_to_stop}"
            )
            
            # Save snapshot of the detection
            self.save_snapshot(frame, detections, "failure")
            
            if self.consecutive_failures >= self.consecutive_failures_to_stop:
                logger.critical("⛔ STOPPING PRINT - Too many consecutive failures detected!")
                return True
        else:
            if self.consecutive_failures > 0:
                logger.info("✅ No failure detected - resetting counter")
            self.consecutive_failures = 0
        
        return False
    
    def run(self):
        """Main analysis loop"""
        self.running = True
        
        while self.running:
            # Connect/reconnect if needed
            if not self.cap or not self.cap.isOpened():
                if not self.connect():
                    logger.warning("Retrying connection in 10 seconds...")
                    time.sleep(10)
                    continue
            
            # Grab and analyze frame
            frame = self.grab_frame()
            if frame is None:
                logger.warning("Failed to grab frame, reconnecting...")
                self._stop_grabber()
                if self.cap is not None:
                    self.cap.release()
                self.cap = None
                with self.frame_lock:
                    self.frame_buffer.clear()
                continue
            
            logger.info("Analyzing frame...")
            detections = self.analyze_frame(frame)
            
            # Save snapshot of every analyzed frame
            if detections:
                self.save_snapshot(frame, detections, "frame")
            else:
                # Save without detections
                self.save_snapshot(frame, [], "frame")
            
            # Log all detections
            if detections:
                for det in detections:
                    logger.info(f"  Detection: {det['label']} ({det['confidence']:.3f})")
            else:
                logger.info("  No detections")
            
            # Process and potentially stop print
            should_stop = self.process_detections(frame, detections)
            
            if should_stop and self.printer_controller:
                self.printer_controller.stop_print("Multiple consecutive failures detected")
                # Continue monitoring but reset counter
                self.consecutive_failures = 0
            
            # Wait for next analysis
            time.sleep(self.analysis_interval)
    
    def stop(self):
        """Stop the analyzer"""
        self.running = False
        self._stop_grabber()
        if self.cap:
            self.cap.release()


def main():
    """Main entry point"""
    # Load configuration from environment
    camera_user = os.getenv('CAMERA_USERNAME', 'admin')
    camera_pass = os.getenv('CAMERA_PASSWORD', '')
    camera_host = os.getenv('CAMERA_HOST', '192.168.1.100')
    camera_port = os.getenv('CAMERA_PORT', '554')
    camera_path = os.getenv('CAMERA_STREAM_PATH', 'stream1')
    
    ml_api_url = os.getenv('ML_API_URL', 'http://ml_api:3333')
    detection_threshold = float(os.getenv('DETECTION_THRESHOLD', '0.40'))
    analysis_interval = float(os.getenv('ANALYSIS_INTERVAL_SECONDS', '10'))
    consecutive_to_stop = int(os.getenv('CONSECUTIVE_FAILURES_TO_STOP', '3'))
    max_snapshot_size_mb = float(os.getenv('MAX_SNAPSHOT_SIZE_MB', '200'))
    
    printer_webhook = os.getenv('PRINTER_STOP_WEBHOOK', '')
    printer_api_key = os.getenv('PRINTER_API_KEY', '')
    
    # Build RTSP URL
    if camera_pass:
        rtsp_url = f"rtsp://{camera_user}:{camera_pass}@{camera_host}:{camera_port}/{camera_path}"
    else:
        rtsp_url = f"rtsp://{camera_host}:{camera_port}/{camera_path}"
    
    logger.info("=" * 60)
    logger.info("3D Print Failure Detection - Stream Analyzer")
    logger.info("=" * 60)
    logger.info(f"Camera: {camera_host}:{camera_port}/{camera_path}")
    logger.info(f"ML API: {ml_api_url}")
    logger.info(f"Detection Threshold: {detection_threshold}")
    logger.info(f"Analysis Interval: {analysis_interval}s")
    logger.info(f"Consecutive Failures to Stop: {consecutive_to_stop}")
    logger.info(f"Max Snapshot Storage: {max_snapshot_size_mb}MB")
    logger.info(f"Printer Webhook: {'configured' if printer_webhook else 'not configured'}")
    logger.info("=" * 60)
    
    # Setup printer controller if webhook is configured
    printer_controller = None
    if printer_webhook:
        printer_controller = PrinterController(printer_webhook, printer_api_key)
    
    # Create and run analyzer
    analyzer = StreamAnalyzer(
        rtsp_url=rtsp_url,
        ml_api_url=ml_api_url,
        detection_threshold=detection_threshold,
        analysis_interval=analysis_interval,
        consecutive_failures_to_stop=consecutive_to_stop,
        printer_controller=printer_controller,
        max_snapshot_size_mb=max_snapshot_size_mb
    )
    
    try:
        analyzer.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        analyzer.stop()


if __name__ == "__main__":
    main()
