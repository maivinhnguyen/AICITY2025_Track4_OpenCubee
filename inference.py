from ultralytics import YOLO
import os
import json
import numpy as np
from tqdm import tqdm
import argparse

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

def process_detections(model, test_folder, output_file, conf_thresholds):
    """
    Process all images in the test folder and save detections in the specified JSON format
    
    Args:
        model: Loaded YOLO model
        test_folder: Folder containing test images
        output_file: Path to save JSON results
        conf_thresholds: List of confidence thresholds, one per class
    """
    # Get list of image files
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {test_folder}")
        return
    
    results = []
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(test_folder, img_file)
        
        # Get image_id from filename
        image_id = get_image_id(img_file)
        if image_id is None:
            print(f"Skipping {img_file} due to filename format mismatch")
            continue
        
        # Run inference
        predictions = model(img_path)[0]
        
        # Extract detections
        for detection in predictions.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, category_id = detection
            
            # Get class-specific confidence threshold
            class_id = int(category_id)
            if class_id < len(conf_thresholds):
                threshold = conf_thresholds[class_id]
            else:
                # Use the first threshold as default if class_id is out of range
                threshold = conf_thresholds[0]
                
            # Filter by confidence threshold
            if score < threshold:
                continue
                
            # Convert to JSON format
            detection_json = {
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(score)
            }
            
            results.append(detection_json)
    
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
    
    print(f"Saved {len(results)} detections to {output_file}")
    print("Detections by class:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  Class {class_id}: {count} detections (threshold: {conf_thresholds[class_id] if class_id < len(conf_thresholds) else conf_thresholds[0]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO inference and JSON export")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to the model checkpoint")
    parser.add_argument("--test_folder", type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument("--output", type=str, default="detections.json", help="Output JSON file path")
    parser.add_argument("--conf", type=str, default="0.25,0.25,0.25,0.25,0.25", 
                        help="Confidence thresholds for each class, comma-separated (e.g., '0.1,0.2,0.3,0.4,0.5')")
    
    args = parser.parse_args()
    
    # Parse confidence thresholds
    try:
        conf_thresholds = [float(t.strip()) for t in args.conf.split(',')]
        print(f"Using class-specific confidence thresholds: {conf_thresholds}")
    except ValueError:
        print(f"Error parsing confidence thresholds. Using default 0.25 for all classes.")
        conf_thresholds = [0.25] * 5
    
    # Load the YOLOv8 model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Process detections
    process_detections(model, args.test_folder, args.output, conf_thresholds)