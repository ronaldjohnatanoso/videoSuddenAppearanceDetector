import cv2
import numpy as np
from pathlib import Path
import datetime
import logging
import json
from typing import List, Tuple
import os
from dataclasses import dataclass
import cupy as cp
from numba import cuda
import torch

@dataclass
class DetectionEvent:
    timestamp: float
    frame_number: int
    confidence: float
    bbox: Tuple[int, int, int, int]

class GPUSuddenAppearanceDetector:
    def __init__(
        self,
        threshold_pixel_diff=25,
        min_area=400,
        min_confidence=0.6,
        batch_size=8,
        blur_kernel_size=3,
        output_dir="detected_events",
        use_gpu=True
    ):
        self.threshold_pixel_diff = threshold_pixel_diff
        self.min_area = min_area
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.blur_kernel_size = blur_kernel_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'detection_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        if self.use_gpu:
            self.device = torch.device('cuda')
            self.stream = cp.cuda.Stream()
        else:
            self.logger.warning("GPU not available, falling back to CPU processing")

    def preprocess_frame(self, frame):
        """CPU fallback for frame preprocessing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        return blurred

    def preprocess_frame_gpu(self, frame):
        """GPU-accelerated frame preprocessing"""
        try:
            # Convert BGR weights to GPU array
            bgr_weights = cp.array([0.299, 0.587, 0.114], dtype=cp.float32)
            
            # Transfer frame to GPU and ensure proper type
            gpu_frame = cp.asarray(frame, dtype=cp.float32)
            
            # Convert to grayscale on GPU
            gpu_gray = cp.dot(gpu_frame[..., :3], bgr_weights)
            
            # Ensure proper type
            gpu_gray = gpu_gray.astype(cp.uint8)
            
            # Apply Gaussian blur on GPU
            gpu_blurred = cp.empty_like(gpu_gray)
            kernel_size = self.blur_kernel_size
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
            
            # Create Gaussian kernel
            kernel = cp.asarray(cv2.getGaussianKernel(kernel_size, sigma))
            kernel = cp.outer(kernel, kernel)
            
            # Apply convolution on GPU
            gpu_blurred = cp.array(cv2.GaussianBlur(
                cp.asnumpy(gpu_gray),
                (kernel_size, kernel_size),
                sigma
            ))
            
            return gpu_blurred
            
        except Exception as e:
            self.logger.error(f"GPU preprocessing failed: {str(e)}")
            self.logger.info("Falling back to CPU preprocessing")
            return cp.asarray(self.preprocess_frame(frame))

    def detect_changes_gpu(self, prev_frame_gpu, curr_frame_gpu):
        """GPU-accelerated change detection"""
        try:
            # Compute frame difference
            frame_diff_gpu = cp.abs(curr_frame_gpu - prev_frame_gpu)
            
            # Threshold the difference
            thresh_gpu = (frame_diff_gpu > self.threshold_pixel_diff) * 255
            thresh_gpu = thresh_gpu.astype(cp.uint8)
            
            # Transfer to CPU for contour detection
            thresh_cpu = cp.asnumpy(thresh_gpu)
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh_cpu,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            events = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                region = frame_diff_gpu[y:y+h, x:x+w].get()
                avg_change = float(cp.mean(region))
                
                confidence = min(1.0, (
                    (area / (self.min_area * 2)) * 0.5 +
                    (avg_change / 255.0) * 0.5
                ))
                
                if confidence >= self.min_confidence:
                    events.append(
                        DetectionEvent(
                            timestamp=0.0,
                            frame_number=0,
                            confidence=float(confidence),
                            bbox=(x, y, w, h)
                        )
                    )
            
            return events
            
        except Exception as e:
            self.logger.error(f"GPU change detection failed: {str(e)}")
            return []

    def save_detection(self, frame, event, video_fps):
        """Save detected frame and metadata"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_filename = f"detection_{timestamp}.jpg"
        meta_filename = f"detection_{timestamp}.json"
        
        frame_path = self.output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)
        
        metadata = {
            "timestamp": event.timestamp,
            "frame_number": event.frame_number,
            "confidence": float(event.confidence),
            "bbox": event.bbox,
            "video_fps": video_fps
        }
        
        meta_path = self.output_dir / meta_filename
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(
            f"Saved detection: {frame_filename} (confidence: {event.confidence:.2f})"
        )

    def print_gpu_memory_usage(self):
        """Print current GPU memory usage"""
        if self.use_gpu:
            mem_free, mem_total = cp.cuda.Device().mem_info
            mem_used = mem_total - mem_free
            self.logger.info(f"GPU Memory: Used {mem_used / 1024**2:.1f}MB / Total {mem_total / 1024**2:.1f}MB")
            

    def process_video(self, video_path: str, display_output: bool = False, start_time: str = "03:15:00"):
        """Process video with GPU acceleration"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Error opening video file: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"FPS: {fps}, Total frames: {frame_count}")
        self.logger.info(f"GPU acceleration: {'enabled' if self.use_gpu else 'disabled'}")
        
        # Convert start_time to total seconds and then to frame number
        hours, minutes, seconds = map(int, start_time.split(':'))
        start_frame = int((hours * 3600 + minutes * 60 + seconds) * fps)
        
        # Skip frames until reaching the starting frame
        for _ in range(start_frame):
            ret = cap.grab()  # Grabs the next frame without decoding it
            if not ret:
                self.logger.warning("Reached end of video before start time.")
                return

        try:
            if self.use_gpu:
                with cp.cuda.Stream() as stream:
                    self.print_gpu_memory_usage()
                    frame_number = start_frame  # Start from the specified frame
                    
                    while True:
                        batch_frames = []
                        for _ in range(self.batch_size):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            batch_frames.append(frame)

                        if not batch_frames:
                            break

                        # Process batch frames on GPU
                        try:
                            gpu_frames = [self.preprocess_frame_gpu(f) for f in batch_frames]

                            for i in range(len(gpu_frames) - 1):
                                events = self.detect_changes_gpu(
                                    gpu_frames[i],
                                    gpu_frames[i + 1]
                                )

                                for event in events:
                                    event.frame_number = frame_number + i
                                    event.timestamp = event.frame_number / fps
                                    self.save_detection(batch_frames[i], event, fps)

                                    if display_output:
                                        frame = batch_frames[i].copy()
                                        x, y, w, h = event.bbox
                                        cv2.rectangle(
                                            frame,
                                            (x, y),
                                            (x + w, y + h),
                                            (0, 255, 0),
                                            2
                                        )
                                        cv2.imshow('Detection', frame)
                                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                            return

                            frame_number += len(batch_frames)

                            # Log progress every 100 frames
                                # print("frame number: ",frame_number)
                            if (frame_number - start_frame) % 100 == 0:
                                progress = (frame_number / frame_count) * 100
                                self.logger.info(f"Progress: {progress:.1f}%")

                                
                                # Calculate and log current video timestamp
                                current_time = frame_number / fps
                                self.logger.info(f"Current video time: {datetime.timedelta(seconds=current_time)}")
                                self.print_gpu_memory_usage()

                        except Exception as e:
                            self.logger.error(f"Error processing batch: {str(e)}")
                            continue
            
            else:
                # CPU fallback processing
                frame_number = start_frame  # Start from the specified frame
                prev_frame = None
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frame = self.preprocess_frame(frame)
                    
                    # Calculate and log the current time in the video
                    current_time = frame_number / fps
                    self.logger.info(f"Current video time: {datetime.timedelta(seconds=current_time)} (Frame {frame_number})")
                    
                    if prev_frame is not None:
                        # Compute difference
                        frame_diff = cv2.absdiff(processed_frame, prev_frame)
                        _, thresh = cv2.threshold(
                            frame_diff,
                            self.threshold_pixel_diff,
                            255,
                            cv2.THRESH_BINARY
                        )
                        
                        contours, _ = cv2.findContours(
                            thresh,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area < self.min_area:
                                continue
                            
                            x, y, w, h = cv2.boundingRect(contour)
                            region = frame_diff[y:y+h, x:x+w]
                            avg_change = np.mean(region)
                            
                            confidence = min(1.0, (
                                (area / (self.min_area * 2)) * 0.5 +
                                (avg_change / 255.0) * 0.5
                            ))
                            
                            if confidence >= self.min_confidence:
                                event = DetectionEvent(
                                    timestamp=frame_number / fps,
                                    frame_number=frame_number,
                                    confidence=float(confidence),
                                    bbox=(x, y, w, h)
                                )
                                self.save_detection(frame, event, fps)
                                
                                if display_output:
                                    display_frame = frame.copy()
                                    cv2.rectangle(
                                        display_frame,
                                        (x, y),
                                        (x + w, y + h),
                                        (0, 255, 0),
                                        2
                                    )
                                    cv2.imshow('Detection', display_frame)
                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        break
                    
                    prev_frame = processed_frame
                    frame_number += 1
                    
                    if frame_number % 100 == 0:
                        progress = (frame_number / frame_count) * 100
                        self.logger.info(f"Progress: {progress:.1f}%")
        
        except Exception as e:
            self.logger.error(f"Error during video processing: {str(e)}")
        
        finally:
            cap.release()
            if display_output:
                cv2.destroyAllWindows()
            
            if self.use_gpu:
                cp.get_default_memory_pool().free_all_blocks()
                self.print_gpu_memory_usage()
            
            self.logger.info("Video processing completed")


def main():
    # Create detector instance
    detector = GPUSuddenAppearanceDetector(
        threshold_pixel_diff=300,    # Adjust sensitivity
        min_area=100,              # Minimum size of changes to detect
        min_confidence=0.6,        # Minimum confidence threshold
        batch_size=1500,             # Reduced batch size for better stability
        use_gpu=True,             # Enable GPU acceleration
        output_dir="detected_events"  # Output directory for detections
    )
    
    # Process video
    video_path = "v1.mp4"  # Replace with your video path
    detector.process_video(
        video_path,
        start_time="03:15:00",
        display_output=True  # Set to False for headless operation
    )

if __name__ == "__main__":
    main()