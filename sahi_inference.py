from ultralytics import YOLO
import os
import json
import numpy as np
import time
from tqdm import tqdm
import argparse
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import list_files
from sahi.utils.cv import read_image

def get_image_id(img_name):
    """
    Extract image ID based on the provided function.
    Format expected: camera{camera_idx}_{scene}_{frame}.png
    Where scene is one of ['M', 'A', 'E', 'N']
    """
    img_name = img_name.split('.png')[0]
    scene_list = ['M', 'A', 'E', 'N']
    
    try:
        camera_idx = int(img_name.split('_')[0].split('camera')[1])
        scene_idx = scene_list.index(img_name.split('_')[1])
        frame_idx = int(img_name.split('_')[2])
        
        # Combine all parts as per the function logic
        image_id = int(str(camera_idx) + str(scene_idx) + str(frame_idx))
        return image_id
    except (IndexError, ValueError) as e:
        print(f"Error parsing image name '{img_name}': {e}")
        return None

def process_detections_with_sahi(model_path, test_folder, output_file, conf_thresholds, 
                                slice_size=512, overlap_ratio=0.2, device=""):
    """
    Process each image with SAHI, slicing it into smaller patches for improved detection
    
    Args:
        model_path: Path to YOLOv8 model
        test_folder: Folder containing test images
        output_file: Path to save JSON results
        conf_thresholds: List of confidence thresholds, one per class
        slice_size: Size of the slices (patches)
        overlap_ratio: Overlap between slices
        device: Device to run inference on
    """
    # Get list of image files
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {test_folder}")
        return
    
    # Create SAHI detection model wrapper around YOLOv8
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=min(conf_thresholds),  # We'll filter by class-specific thresholds later
        device=device if device else None,
    )
    
    results = []
    total_time = 0
    total_images = 0
    
    # Process each image individually for real-time application
    for img_file in tqdm(image_files, desc="Processing images"):
        start_time = time.time()
        
        img_path = os.path.join(test_folder, img_file)
        
        # Get image_id from filename
        image_id = get_image_id(img_file)
        if image_id is None:
            print(f"Skipping {img_file} due to filename format mismatch")
            continue
        
        # Get sliced predictions
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            perform_standard_pred=True,  # Also perform full-image prediction
            postprocess_type="NMM",  # Non-Maximum Merging for better results
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        
        # Process each detection
        for object_prediction in result.object_prediction_list:
            bbox = object_prediction.bbox
            category_id = object_prediction.category.id
            score = object_prediction.score.value
            
            # Apply class-specific thresholds
            if category_id < len(conf_thresholds):
                threshold = conf_thresholds[category_id]
            else:
                threshold = conf_thresholds[0]
                
            if score < threshold:
                continue
            
            # Convert to JSON format
            detection_json = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(bbox.minx), float(bbox.miny), 
                         float(bbox.maxx - bbox.minx), float(bbox.maxy - bbox.miny)],
                "score": float(score)
            }
            
            results.append(detection_json)
        
        # Calculate FPS
        process_time = time.time() - start_time
        total_time += process_time
        total_images += 1
        
        # Clear memory
        if hasattr(detection_model.model, "cuda"):
            if hasattr(detection_model.model, "clear_cuda_cache") and device != "cpu":
                detection_model.model.clear_cuda_cache()
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Count detections per class
    class_counts = {}
    for detection in results:
        class_id = detection["category_id"]
        if class_id not in class_counts:
            class_counts[class_id] = 0
        class_counts[class_id] += 1
    
    # Print statistics
    avg_fps = total_images / total_time if total_time > 0 else 0
    print(f"\nAverage FPS: {avg_fps:.2f}")
    print(f"Saved {len(results)} detections to {output_file}")
    print("Detections by class:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  Class {class_id}: {count} detections (threshold: {conf_thresholds[class_id] if class_id < len(conf_thresholds) else conf_thresholds[0]})")

def benchmark_configs(model_path, test_folder, device, conf_thresholds, num_test_images=10):
    """
    Benchmark different SAHI configurations to find the optimal settings for FPS
    """
    print("Benchmarking SAHI configurations for optimal FPS...")
    
    # Get a subset of images for benchmarking
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) > num_test_images:
        image_files = image_files[:num_test_images]
    
    # Test configurations
    slice_sizes = [256, 512, 640]
    overlap_ratios = [0.1, 0.2, 0.3]
    
    best_fps = 0
    best_config = None
    
    # Create detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=min(conf_thresholds),
        device=device if device else None,
    )
    
    print(f"{'Slice Size':<10} | {'Overlap':<10} | {'FPS':<10}")
    print("-" * 34)
    
    for slice_size in slice_sizes:
        for overlap_ratio in overlap_ratios:
            total_time = 0
            
            for img_file in image_files:
                img_path = os.path.join(test_folder, img_file)
                
                start_time = time.time()
                
                # Test current configuration
                result = get_sliced_prediction(
                    img_path,
                    detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap_ratio,
                    overlap_width_ratio=overlap_ratio,
                    perform_standard_pred=True,
                    postprocess_type="NMM",
                    postprocess_match_metric="IOU",
                    postprocess_match_threshold=0.5,
                    verbose=0,
                )
                
                process_time = time.time() - start_time
                total_time += process_time
            
            avg_fps = len(image_files) / total_time if total_time > 0 else 0
            print(f"{slice_size:<10} | {overlap_ratio:<10.1f} | {avg_fps:<10.2f}")
            
            if avg_fps > best_fps:
                best_fps = avg_fps
                best_config = (slice_size, overlap_ratio)
    
    print(f"\nBest configuration: Slice Size={best_config[0]}, Overlap Ratio={best_config[1]}")
    print(f"Best FPS: {best_fps:.2f}")
    
    return best_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO inference with SAHI and JSON export")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to the model checkpoint")
    parser.add_argument("--test_folder", type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument("--output", type=str, default="detections.json", help="Output JSON file path")
    parser.add_argument("--conf", type=str, default="0.25,0.25,0.25,0.25,0.25", 
                        help="Confidence thresholds for each class, comma-separated (e.g., '0.1,0.2,0.3,0.4,0.5')")
    parser.add_argument("--device", type=str, default="", 
                        help="Device to run inference on (e.g., '0' for GPU 0, or 'cpu')")
    parser.add_argument("--slice_size", type=int, default=512, 
                        help="Size of image slices for SAHI")
    parser.add_argument("--overlap", type=float, default=0.2, 
                        help="Overlap ratio between slices")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmarks to find optimal configurations")
    
    args = parser.parse_args()
    
    # Parse confidence thresholds
    try:
        conf_thresholds = [float(t.strip()) for t in args.conf.split(',')]
        print(f"Using class-specific confidence thresholds: {conf_thresholds}")
    except ValueError:
        print(f"Error parsing confidence thresholds. Using default 0.25 for all classes.")
        conf_thresholds = [0.25] * 5
    
    print(f"Loading model from {args.model}...")
    
    # Set device
    if args.device:
        print(f"Using device: {args.device}")
    
    if args.benchmark:
        # Run benchmarks to find optimal configuration
        best_config = benchmark_configs(
            args.model, 
            args.test_folder, 
            args.device, 
            conf_thresholds
        )
        slice_size, overlap_ratio = best_config
    else:
        slice_size = args.slice_size
        overlap_ratio = args.overlap
    
    # Process detections with SAHI
    process_detections_with_sahi(
        args.model,
        args.test_folder,
        args.output,
        conf_thresholds,
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
        device=args.device
    )