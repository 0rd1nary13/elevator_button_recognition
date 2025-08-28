#!/usr/bin/env python3
"""
Standalone Elevator Button Recognition Script
去除ROS依赖的独立测试版本
"""

import os
import sys
import cv2
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Add utils to path
sys.path.append('src/button_recognition/scripts')
sys.path.append('src/button_recognition/scripts/utils')

try:
    # Try different import methods
    try:
        from utils import label_map_util, visualization_utils as vis_util
    except ImportError:
        import label_map_util
        import visualization_utils as vis_util
except ImportError as e:
    print(f"❌ Error: Cannot import utils modules: {e}")
    print("Available modules test:")
    
    # Test imports individually
    try:
        import sys
        sys.path.append('src/button_recognition/scripts/utils')
        import label_map_util
        print("✅ label_map_util imported successfully")
        import visualization_utils as vis_util
        print("✅ visualization_utils imported successfully")
    except Exception as e:
        print(f"❌ Individual import failed: {e}")
        print("Make sure you're in the project root directory")
        sys.exit(1)

# Configuration
DISP_IMG_SIZE = (12, 8)
NUM_CLASSES = 1
VERBOSE = True

class StandaloneButtonRecognition:
    """Standalone Button Recognition without ROS dependencies"""
    
    def __init__(self):
        self.session = None
        self.img_key = None
        self.results = []
        self.ocr_rcnn = {}
        
    def init_model(self, graph_path=None, label_path=None):
        """Initialize the OCR-RCNN model"""
        # Default paths
        if graph_path is None:
            graph_path = 'src/button_recognition/ocr_rcnn_model/frozen_inference_graph.pb'
        if label_path is None:
            label_path = 'src/button_recognition/model_config/button_label_map.pbtxt'
            
        print(f"📁 Loading model from: {graph_path}")
        print(f"📁 Loading labels from: {label_path}")
        
        # Check if files exist
        if not os.path.exists(graph_path):
            print(f"❌ Model file not found: {graph_path}")
            print("💡 Please download the model from:")
            print("   https://drive.google.com/file/d/1SM3p5NW6k2R04Bn72T1veE8hJSNnbvzf/view?usp=sharing")
            print("   And place it in src/button_recognition/ocr_rcnn_model/")
            return False
            
        if not os.path.exists(label_path):
            print(f"❌ Label file not found: {label_path}")
            return False
            
        try:
            # Load frozen graph (TensorFlow 2.x compatible)
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.compat.v1.GraphDef()
                with open(graph_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                    
            # Disable eager execution for TensorFlow 1.x compatibility
            tf.compat.v1.disable_eager_execution()
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
                label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)
            
            self.ocr_rcnn['category_index'] = category_index
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_image(self, image_path):
        """Predict buttons in an image"""
        if self.session is None:
            print("❌ Model not initialized!")
            return None
            
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
            
        # Load image
        print(f"🖼️  Processing image: {image_path}")
        image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_np is None:
            print("❌ Failed to load image!")
            return None
            
        # Convert BGR to RGB for display
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        
        # Prepare image for prediction
        img_height, img_width = image_np.shape[:2]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        # Run prediction
        (boxes, scores, classes, chars, num) = self.session.run(
            self.results, feed_dict={self.img_key: image_np_expanded})
        
        # Process results
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        chars = np.squeeze(chars)
        num = int(np.squeeze(num))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Filter by score threshold
        valid_detections = []
        for i, score in enumerate(scores):
            if score >= 0.5:
                detection = {
                    'box': boxes[i],
                    'score': float(score),
                    'class': int(classes[i]),
                    'chars': chars[i] if len(chars.shape) > 1 else [chars[i]],
                    'pixel_box': {
                        'y_min': int(boxes[i][0] * img_height),
                        'x_min': int(boxes[i][1] * img_width),
                        'y_max': int(boxes[i][2] * img_height),
                        'x_max': int(boxes[i][3] * img_width)
                    }
                }
                valid_detections.append(detection)
        
        print(f"🎯 Found {len(valid_detections)} buttons (processing time: {processing_time:.2f}s)")
        
        # Visualize results
        if VERBOSE and len(valid_detections) > 0:
            self.visualize_results(image_np.copy(), boxes, classes, scores, chars)
        
        return {
            'image': image_np,
            'detections': valid_detections,
            'processing_time': processing_time
        }
    
    def visualize_results(self, image_np, boxes, classes, scores, chars):
        """Visualize detection results"""
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            classes.astype(np.int32),
            scores,
            self.ocr_rcnn['category_index'],
            max_boxes_to_draw=100,
            use_normalized_coordinates=True,
            line_thickness=5,
            predict_chars=chars
        )
        
        plt.figure(figsize=DISP_IMG_SIZE)
        plt.imshow(image_np)
        plt.axis('off')
        plt.title('Elevator Button Recognition Results')
        plt.show()
    
    def close(self):
        """Clean up resources"""
        if self.session is not None:
            self.session.close()
            print("🔒 Session closed")

def test_with_demo_images():
    """Test with demo images and all images in test_samples folder"""
    recognizer = StandaloneButtonRecognition()
    
    # Initialize model
    if not recognizer.init_model():
        return
    
    # Test images from demos folder
    demo_images = []
    demos_dir = 'demos/'
    if os.path.exists(demos_dir):
        for file in os.listdir(demos_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                demo_images.append(os.path.join(demos_dir, file))
        demo_images.sort()  # Sort for consistent ordering
    
    # Scan test_samples folder for all image files
    test_images = []
    test_samples_dir = 'src/button_tracker/test_samples/'
    if os.path.exists(test_samples_dir):
        print(f"📁 Scanning {test_samples_dir} for images...")
        for file in os.listdir(test_samples_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                full_path = os.path.join(test_samples_dir, file)
                if os.path.isfile(full_path):  # Make sure it's a file, not a directory
                    test_images.append(full_path)
        test_images.sort()  # Sort for consistent ordering
        print(f"📸 Found {len(test_images)} image files in test_samples folder")
    
    all_images = demo_images + test_images
    available_images = [img for img in all_images if os.path.exists(img)]
    
    if not available_images:
        print("❌ No test images found!")
        print("Available directories:")
        for d in ['demos/', 'src/button_tracker/test_samples/']:
            if os.path.exists(d):
                files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
                print(f"  📁 {d}: {len(files)} image files - {files}")
        return
    
    print(f"🖼️  Found {len(available_images)} total images to process")
    print(f"   📁 Demo images: {len(demo_images)}")
    print(f"   📁 Test samples: {len(test_images)}")
    
    # Process each image
    total_buttons_detected = 0
    total_processing_time = 0
    successful_detections = 0
    detection_summary = []
    
    for idx, image_path in enumerate(available_images):
        print(f"\n{'='*60}")
        print(f"Processing image {idx+1}/{len(available_images)}")
        
        # Show which folder the image is from
        if image_path.startswith('demos/'):
            print(f"📁 Source: Demo folder - {os.path.basename(image_path)}")
        elif image_path.startswith('src/button_tracker/test_samples/'):
            print(f"📁 Source: Test samples folder - {os.path.basename(image_path)}")
        else:
            print(f"📁 Source: {image_path}")
            
        result = recognizer.predict_image(image_path)
        
        if result:
            print(f"✅ Detection Results:")
            num_buttons = len(result['detections'])
            total_buttons_detected += num_buttons
            total_processing_time += result['processing_time']
            successful_detections += 1
            
            # Store summary info
            detection_summary.append({
                'image': os.path.basename(image_path),
                'buttons': num_buttons,
                'processing_time': result['processing_time']
            })
            
            if result['detections']:
                for i, detection in enumerate(result['detections']):
                    print(f"  🎯 Button {i+1}: Score={detection['score']:.3f}, "
                          f"Box=({detection['pixel_box']['x_min']}, {detection['pixel_box']['y_min']}, "
                          f"{detection['pixel_box']['x_max']}, {detection['pixel_box']['y_max']})")
            else:
                print("  ❌ No buttons detected in this image")
        else:
            print("❌ Failed to process image")
            detection_summary.append({
                'image': os.path.basename(image_path),
                'buttons': 0,
                'processing_time': 0
            })
        
        # Wait for user input to continue (except for the last image)
        if idx < len(available_images) - 1:
            user_input = input("\nPress Enter to continue to next image, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful_detections}/{len(available_images)} images")
    print(f"🎯 Total buttons detected: {total_buttons_detected}")
    print(f"⏱️  Average processing time: {total_processing_time/max(successful_detections, 1):.2f}s per image")
    print(f"📈 Average buttons per image: {total_buttons_detected/max(successful_detections, 1):.1f}")
    
    print("\n📋 Detailed Results:")
    for summary in detection_summary:
        print(f"  📸 {summary['image']}: {summary['buttons']} buttons ({summary['processing_time']:.2f}s)")
    print(f"{'='*60}")
    
    recognizer.close()

if __name__ == '__main__':
    print("🚀 Standalone Elevator Button Recognition")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('src/button_recognition'):
        print("❌ Error: Please run this script from the project root directory")
        print("   cd /Users/james/Projects/elevator_button_recognition")
        sys.exit(1)
    
    test_with_demo_images()
