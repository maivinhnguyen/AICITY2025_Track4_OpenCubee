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
from sahi.utils.file import list_files # Not used directly, os.listdir is used
# from sahi.utils.cv import read_image # read_image is done by SAHI internally

def get_image_id(img_name):
    """
    Extract image ID based on the provided function.
    Format expected: camera{camera_idx}_{scene}_{frame}.png
    Where scene is one of ['M', 'A', 'E', 'N']
    """
    img_name_stem = os.path.splitext(img_name)[0] # More robust way to remove extension
    scene_list = ['M', 'A', 'E', 'N']
    
    parts = img_name_stem.split('_')
    if len(parts) != 3:
        print(f"Error parsing image name '{img_name}': Incorrect number of parts.")
        return None
        
    try:
        camera_idx_str = parts[0].split('camera')
        if len(camera_idx_str) < 2 or not camera_idx_str[1].isdigit():
            raise ValueError("Camera index format error")
        camera_idx = int(camera_idx_str[1])
        
        scene_char = parts[1]
        if scene_char not in scene_list:
            raise ValueError(f"Scene '{scene_char}' not in {scene_list}")
        scene_idx = scene_list.index(scene_char)
        
        if not parts[2].isdigit():
            raise ValueError("Frame index format error")
        frame_idx = int(parts[2])
        
        # Combine all parts as per the function logic
        image_id = int(f"{camera_idx}{scene_idx}{frame_idx:04d}") # Added padding for frame_idx for consistency
        return image_id
    except (IndexError, ValueError) as e:
        print(f"Error parsing image name '{img_name_stem}.png': {e}")
        return None

def process_detections_with_sahi(detection_model, test_folder, output_file, conf_thresholds, 
                                slice_size=640, overlap_ratio=0.2, 
                                perform_standard_pred=True, postprocess_type="NMM"): # Added more SAHI params
    """
    Process each image with SAHI.
    
    Args:
        detection_model: Initialized SAHI AutoDetectionModel
        test_folder: Folder containing test images
        output_file: Path to save JSON results
        conf_thresholds: List of confidence thresholds, one per class
        slice_size: Size of the slices (patches)
        overlap_ratio: Overlap between slices
        perform_standard_pred: Whether to perform prediction on the full image as well
        postprocess_type: "NMM" or "GREEDYNMM" or "NMS"
    """
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {test_folder}")
        return
    
    results = []
    total_time = 0
    total_images = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        start_time = time.time()
        
        img_path = os.path.join(test_folder, img_file)
        
        image_id = get_image_id(img_file)
        if image_id is None:
            print(f"Skipping {img_file} due to filename format mismatch or parsing error")
            continue
        
        # Update confidence threshold for SAHI if needed (using min for SAHI's initial pass)
        # This is already done when detection_model is created, but good to be aware.
        # detection_model.confidence_threshold = min(conf_thresholds)

        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            perform_standard_pred=perform_standard_pred,
            postprocess_type=postprocess_type,
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5, # Standard for IOU matching in NMM/NMS
            verbose=0, # 0 for no print, 1 for progress bar, 2 for detailed prints
        )
        
        for object_prediction in result.object_prediction_list:
            bbox = object_prediction.bbox
            category_id = object_prediction.category.id 
            score = object_prediction.score.value
            
            # Ensure category_id is within bounds for conf_thresholds
            # If not, use the first threshold or a default (e.g., min_conf_thresholds)
            current_threshold = conf_thresholds[category_id] if category_id < len(conf_thresholds) else min(conf_thresholds)
                
            if score < current_threshold:
                continue
            
            detection_json = {
                "image_id": image_id,
                "category_id": category_id, # Ultralytics/SAHI usually gives 0-indexed
                "bbox": [float(bbox.minx), float(bbox.miny), 
                         float(bbox.maxx - bbox.minx), float(bbox.maxy - bbox.miny)],
                "score": float(score)
            }
            results.append(detection_json)
        
        process_time = time.time() - start_time
        total_time += process_time
        total_images += 1
        
        # DO NOT DO THIS: detection_model.model.clear_cuda_cache() - it slows things down
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    class_counts = {}
    for detection in results:
        class_id = detection["category_id"]
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    avg_fps = total_images / total_time if total_time > 0 else 0
    print(f"\nAverage FPS: {avg_fps:.2f}")
    print(f"Saved {len(results)} detections from {total_images} images to {output_file}")
    print("Detections by class:")
    for class_id, count in sorted(class_counts.items()):
        threshold_info = conf_thresholds[class_id] if class_id < len(conf_thresholds) else f"default ({min(conf_thresholds)})"
        print(f"  Class {class_id}: {count} detections (threshold: {threshold_info})")

def benchmark_configs(model_path, test_folder, device, conf_thresholds, use_fp16, num_test_images=10):
    """
    Benchmark different SAHI configurations.
    Model is loaded ONCE before benchmarking loop.
    """
    print(f"\nBenchmarking SAHI configurations (FP16: {use_fp16})...")
    
    # Load the base YOLO model
    yolo_model = YOLO(model_path)
    if use_fp16 and device != "cpu":
        print("Using FP16 for YOLO model.")
        yolo_model.half() 
    yolo_model.to(device if device else "cuda" if torch.cuda.is_available() else "cpu") # Ensure model is on device

    # Create SAHI detection model from the loaded YOLO model object
    # Use a general low confidence for SAHI's own filtering, actual filtering is done later per class
    # This is important because SAHI's `confidence_threshold` is a single value.
    sahi_confidence_threshold = min(conf_thresholds) if conf_thresholds else 0.01 
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", # Important to specify if passing a model object
        model=yolo_model,    # Pass the loaded Ultralytics model object
        confidence_threshold=sahi_confidence_threshold, 
        device=device if device else None, # SAHI will use the device the yolo_model is on
    )

    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found for benchmarking.")
        return None
        
    benchmark_images = image_files[:num_test_images]
    if not benchmark_images:
        print(f"Not enough images for benchmarking (need {num_test_images}, found {len(image_files)})")
        return None

    # slice_sizes = [256, 512, 640, 768] # Test larger sizes
    # overlap_ratios = [0.1, 0.2] # Smaller overlap is faster
    # perform_standard_preds = [True, False]
    # postprocess_types = ["NMM", "GREEDYNMM"]

    # Simplified for quicker feedback, expand as needed
    slice_sizes = [512, 640, 960] 
    overlap_ratios = [0.15, 0.25] 
    perform_standard_preds = [True, False] 
    postprocess_types = ["NMM"]


    best_fps = 0
    best_config = {}
    
    print(f"{'Slice':<7} | {'Overlap':<7} | {'FullImg':<7} | {'Post':<10} | {'FPS':<7}")
    print("-" * 45)
    
    for slice_size in slice_sizes:
        for overlap_ratio in overlap_ratios:
            for perform_standard in perform_standard_preds:
                for post_type in postprocess_types:
                    total_time = 0
                    processed_count = 0
                    
                    for img_file in benchmark_images:
                        img_path = os.path.join(test_folder, img_file)
                        start_time = time.time()
                        
                        _ = get_sliced_prediction( # Store result in _ as we only need time
                            img_path,
                            detection_model,
                            slice_height=slice_size,
                            slice_width=slice_size,
                            overlap_height_ratio=overlap_ratio,
                            overlap_width_ratio=overlap_ratio,
                            perform_standard_pred=perform_standard,
                            postprocess_type=post_type,
                            postprocess_match_metric="IOU",
                            postprocess_match_threshold=0.5,
                            verbose=0,
                        )
                        
                        total_time += (time.time() - start_time)
                        processed_count +=1
                    
                    if processed_count == 0: continue

                    current_fps = processed_count / total_time if total_time > 0 else 0
                    print(f"{slice_size:<7} | {overlap_ratio:<7.2f} | {str(perform_standard):<7} | {post_type:<10} | {current_fps:<7.2f}")
                    
                    if current_fps > best_fps:
                        best_fps = current_fps
                        best_config = {
                            "slice_size": slice_size, 
                            "overlap_ratio": overlap_ratio,
                            "perform_standard_pred": perform_standard,
                            "postprocess_type": post_type
                        }
    
    if best_config:
        print(f"\nBest benchmarked config: {best_config}")
        print(f"Best benchmarked FPS: {best_fps:.2f}")
    else:
        print("\nBenchmarking failed to find a best configuration.")
    
    return best_config, detection_model # Return the loaded model too

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO inference with SAHI and JSON export")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLOv8 model checkpoint (e.g., yolov8n.pt)")
    parser.add_argument("--test_folder", type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument("--output", type=str, default="detections.json", help="Output JSON file path")
    parser.add_argument("--conf", type=str, default="0.25,0.25,0.25,0.25,0.25", 
                        help="Confidence thresholds for each class, comma-separated. Number of values should match number of classes.")
    parser.add_argument("--device", type=str, default="", 
                        help="Device to run inference on (e.g., '0' for GPU 0, 'cpu'). If empty, auto-detects CUDA.")
    parser.add_argument("--slice_size", type=int, default=640, help="Default slice size if not benchmarking.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Default overlap ratio if not benchmarking.")
    parser.add_argument("--perform_full_pred", type=lambda x: (str(x).lower() == 'true'), default=True, help="Default for perform_standard_pred (True/False) if not benchmarking.")
    parser.add_argument("--postprocess_type", type=str, default="NMM", choices=["NMM", "GREEDYNMM", "NMS"], help="Default postprocess type if not benchmarking.")

    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks to find optimal SAHI configurations.")
    parser.add_argument("--num_benchmark_images", type=int, default=10, help="Number of images for benchmarking.")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 (half-precision) for inference (GPU only).")
    
    args = parser.parse_args()
    
    # PyTorch import for device check and FP16
    import torch # Import here to make it conditional if not strictly needed at top level
    
    # Parse confidence thresholds
    try:
        conf_thresholds = [float(t.strip()) for t in args.conf.split(',')]
    except ValueError:
        print(f"Error parsing confidence thresholds from '{args.conf}'. Using default 0.25 for all classes.")
        # Assuming 5 classes if not specified, adjust if your model has different number
        conf_thresholds = [0.25] * 5 # Or set to a single value list: [0.25]
    print(f"Using class-specific confidence thresholds: {conf_thresholds}")
    if not conf_thresholds: # Fallback if parsing results in empty list
        print("Confidence thresholds list is empty. Using 0.25 for all.")
        conf_thresholds = [0.25] 
    
    # Set device
    if args.device:
        device_to_use = args.device
    elif torch.cuda.is_available():
        device_to_use = "cuda" # Or "cuda:0"
    else:
        device_to_use = "cpu"
    print(f"Using device: {device_to_use}")
    if args.fp16 and device_to_use == "cpu":
        print("Warning: FP16 is requested but device is CPU. FP16 will not be used.")
        use_fp16_runtime = False
    else:
        use_fp16_runtime = args.fp16


    # Initialize these variables
    slice_size_to_use = args.slice_size
    overlap_ratio_to_use = args.overlap
    perform_standard_pred_to_use = args.perform_full_pred
    postprocess_type_to_use = args.postprocess_type
    sahi_detection_model = None

    if args.benchmark:
        best_config, loaded_model_for_benchmark = benchmark_configs(
            args.model, 
            args.test_folder, 
            device_to_use, 
            conf_thresholds,
            use_fp16_runtime, # Pass FP16 preference
            num_test_images=args.num_benchmark_images
        )
        if best_config:
            slice_size_to_use = best_config["slice_size"]
            overlap_ratio_to_use = best_config["overlap_ratio"]
            perform_standard_pred_to_use = best_config["perform_standard_pred"]
            postprocess_type_to_use = best_config["postprocess_type"]
            sahi_detection_model = loaded_model_for_benchmark # Reuse the model
            print(f"Proceeding with benchmarked config: Slice={slice_size_to_use}, Overlap={overlap_ratio_to_use:.2f}, FullPred={perform_standard_pred_to_use}, PostProcess={postprocess_type_to_use}")
        else:
            print("Benchmarking did not yield a best configuration. Using default parameters.")
            # Fall through to load model if benchmark didn't provide one
    
    # Load model if not already loaded by benchmark or if benchmark failed
    if sahi_detection_model is None:
        print(f"Loading model from {args.model} for main processing (FP16: {use_fp16_runtime})...")
        yolo_model_main = YOLO(args.model)
        if use_fp16_runtime and device_to_use != "cpu":
            yolo_model_main.half()
        yolo_model_main.to(device_to_use)

        sahi_confidence_threshold = min(conf_thresholds) if conf_thresholds else 0.01
        sahi_detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model=yolo_model_main, # Pass the loaded Ultralytics model object
            confidence_threshold=sahi_confidence_threshold, 
            device=device_to_use, # SAHI uses the device yolo_model_main is on
        )
    
    # Process detections with SAHI using chosen/benchmarked parameters
    process_detections_with_sahi(
        sahi_detection_model,
        args.test_folder,
        args.output,
        conf_thresholds,
        slice_size=slice_size_to_use,
        overlap_ratio=overlap_ratio_to_use,
        perform_standard_pred=perform_standard_pred_to_use,
        postprocess_type=postprocess_type_to_use
    )