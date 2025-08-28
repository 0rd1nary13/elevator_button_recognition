#!/usr/bin/env python3
"""
实时摄像头电梯按钮检测脚本
Real-time camera elevator button detection
"""

import os
import sys
import cv2
import time
import numpy as np
import tensorflow as tf
import threading
from queue import Queue

# Add utils to path
sys.path.append('src/button_recognition/scripts')
sys.path.append('src/button_recognition/scripts/utils')

try:
    import label_map_util
    import visualization_utils as vis_util
except ImportError as e:
    print(f"❌ Error: Cannot import utils modules: {e}")
    print("Make sure you're in the project root directory")
    sys.exit(1)

class RealtimeButtonDetector:
    """Real-time button detector using webcam"""
    
    def __init__(self):
        self.session = None
        self.img_key = None
        self.results = []
        self.category_index = {}
        self.frame_queue = Queue(maxsize=2)  # Limit queue size to avoid lag
        self.result_queue = Queue(maxsize=2)
        self.is_running = False
        self.detection_interval = 5  # Process every N frames to improve performance
        self.frame_count = 0
        
    def init_model(self):
        """Initialize the detection model"""
        graph_path = 'src/button_recognition/ocr_rcnn_model/frozen_inference_graph.pb'
        label_path = 'src/button_recognition/model_config/button_label_map.pbtxt'
        
        print(f"📁 Loading model from: {graph_path}")
        print(f"📁 Loading labels from: {label_path}")
        
        if not os.path.exists(graph_path):
            print(f"❌ Model file not found: {graph_path}")
            return False
            
        if not os.path.exists(label_path):
            print(f"❌ Label file not found: {label_path}")
            return False
            
        try:
            # Load frozen graph (TensorFlow 2.x compatible)
            tf.compat.v1.disable_eager_execution()
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.compat.v1.GraphDef()
                with open(graph_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                    
            self.session = tf.compat.v1.Session(graph=detection_graph)
            self.img_key = detection_graph.get_tensor_by_name('image_tensor:0')
            self.results.append(detection_graph.get_tensor_by_name('detection_boxes:0'))
            self.results.append(detection_graph.get_tensor_by_name('detection_scores:0'))
            self.results.append(detection_graph.get_tensor_by_name('detection_classes:0'))
            self.results.append(detection_graph.get_tensor_by_name('predicted_chars:0'))
            self.results.append(detection_graph.get_tensor_by_name('num_detections:0'))
            
            # Load label map
            label_map = label_map_util.load_labelmap(label_path)
            categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=1, use_display_name=True)
            self.category_index = label_map_util.create_category_index(categories)
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detection_worker(self):
        """Worker thread for running detection"""
        while self.is_running:
            try:
                # Get frame from queue
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Resize frame for faster processing
                    height, width = frame.shape[:2]
                    scale_factor = 0.5  # Reduce to 50% for faster processing
                    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
                    frame_resized = cv2.resize(frame, (new_width, new_height))
                    
                    start_time = time.time()
                    
                    # Prepare image for prediction
                    image_np_expanded = np.expand_dims(frame_resized, axis=0)
                    
                    # Run prediction
                    (boxes, scores, classes, chars, num) = self.session.run(
                        self.results, feed_dict={self.img_key: image_np_expanded})
                    
                    # Process results
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)
                    classes = np.squeeze(classes)
                    chars = np.squeeze(chars)
                    
                    # Scale boxes back to original frame size
                    boxes[:, 0] *= height  # y_min
                    boxes[:, 1] *= width   # x_min  
                    boxes[:, 2] *= height  # y_max
                    boxes[:, 3] *= width   # x_max
                    
                    # Filter detections
                    valid_detections = []
                    for i, score in enumerate(scores):
                        if score >= 0.5:
                            detection = {
                                'box': [int(boxes[i][1]), int(boxes[i][0]), 
                                       int(boxes[i][3]), int(boxes[i][2])],  # [x_min, y_min, x_max, y_max]
                                'score': float(score),
                                'class': int(classes[i]),
                                'chars': chars[i] if len(chars.shape) > 1 else [chars[i]]
                            }
                            valid_detections.append(detection)
                    
                    processing_time = time.time() - start_time
                    
                    # Put result in queue
                    if not self.result_queue.full():
                        self.result_queue.put({
                            'detections': valid_detections,
                            'processing_time': processing_time
                        })
                        
            except Exception as e:
                print(f"❌ Detection error: {e}")
                continue
    
    def start_camera_detection(self, camera_id=0):
        """Start real-time camera detection"""
        print(f"📷 Starting camera {camera_id}...")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return False
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start detection worker thread
        self.is_running = True
        detection_thread = threading.Thread(target=self.detection_worker)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Main display loop
        latest_detections = []
        fps_counter = 0
        fps_start_time = time.time()
        
        print("🎯 Real-time detection started!")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from camera!")
                break
            
            self.frame_count += 1
            
            # Add frame to detection queue every N frames
            if self.frame_count % self.detection_interval == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
            
            # Get latest detection results
            if not self.result_queue.empty():
                result = self.result_queue.get()
                latest_detections = result['detections']
                processing_time = result['processing_time']
            
            # Draw detections on frame
            display_frame = frame.copy()
            
            for detection in latest_detections:
                x_min, y_min, x_max, y_max = detection['box']
                score = detection['score']
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw score
                label = f"Button: {score:.2f}"
                cv2.putText(display_frame, label, (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate and display FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                fps = fps_counter / (current_time - fps_start_time)
                fps_counter = 0
                fps_start_time = current_time
            else:
                fps = fps_counter / (current_time - fps_start_time) if current_time - fps_start_time > 0 else 0
            
            # Draw status info
            status_text = f"FPS: {fps:.1f} | Buttons: {len(latest_detections)}"
            if latest_detections:
                status_text += f" | Last Processing: {processing_time:.1f}s"
            
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Real-time Button Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"button_detection_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"📸 Screenshot saved: {filename}")
        
        # Cleanup
        self.is_running = False
        cap.release()
        cv2.destroyAllWindows()
        print("🔒 Camera detection stopped")
        
    def close(self):
        """Clean up resources"""
        self.is_running = False
        if self.session is not None:
            self.session.close()

def main():
    """Main function"""
    print("📹 电梯按钮实时检测系统")
    print("=" * 50)
    
    detector = RealtimeButtonDetector()
    
    # Initialize model
    if not detector.init_model():
        print("❌ Model initialization failed!")
        return
    
    try:
        # Start camera detection
        detector.start_camera_detection(camera_id=0)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        detector.close()
    
    print("👋 程序结束")

if __name__ == '__main__':
    main()
