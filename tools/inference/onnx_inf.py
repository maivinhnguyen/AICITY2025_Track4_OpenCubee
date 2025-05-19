"""
Modified ONNX Inference Script for AI CITY Challenge Track 4
Follows competition guidelines: sequential processing, no batching, includes all timing
"""

import os
import time
import json
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def get_image_id(img_name):
    """Generate image ID according to the specified format."""
    img_name = img_name.split('.png')[0]
    scene_list = ['M', 'A', 'E', 'N']
    camera_idx = int(img_name.split('_')[0].split('camera')[1])
    scene_idx = scene_list.index(img_name.split('_')[1])
    frame_idx = int(img_name.split('_')[2])
    image_id = int(str(camera_idx) + str(scene_idx) + str(frame_idx))
    return image_id


def process_detections(image_name, boxes, labels, scores, ratio, padding, conf_threshold=0.4):
    """Convert detections to the required JSON format."""
    pad_w, pad_h = padding
    
    image_id = get_image_id(image_name)
    detections = []
    
    for i, score in enumerate(scores):
        if score <= conf_threshold:
            continue
            
        box = boxes[i]
        label = int(labels[i])  # Ensure it's an integer
        
        # Adjust bounding boxes according to the resizing and padding
        x1 = (box[0] - pad_w) / ratio
        y1 = (box[1] - pad_h) / ratio
        x2 = (box[2] - pad_w) / ratio
        y2 = (box[3] - pad_h) / ratio
        
        # Convert from [x1, y1, x2, y2] to [x1, y1, width, height]
        width = x2 - x1
        height = y2 - y1
        
        detection = {
            "image_id": image_id,
            "category_id": label,
            "bbox": [float(x1), float(y1), float(width), float(height)],
            "score": float(score)
        }
        detections.append(detection)
    
    return detections


def init_onnx_session(onnx_model_path):
    """Initialize ONNX Runtime session with CUDA as provider."""
    print(f"Loading ONNX model from {onnx_model_path}")
    
    # Set up ONNX Runtime session with optimization options
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = True
    
    # Force CUDA:0 device
    cuda_provider_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }
    
    providers = [('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
    
    try:
        sess = ort.InferenceSession(onnx_model_path, session_options, providers=providers)
        # Check if CUDA is being used
        device = ort.get_device()
        if 'GPU' in device or 'CUDA' in device:
            print(f"Using CUDA device: {device}")
        else:
            print("WARNING: Not using CUDA for inference. Using CPU instead.")
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        print("Falling back to CPU")
        sess = ort.InferenceSession(onnx_model_path, session_options)
    
    return sess


def process_image_folder(image_folder, sess, output_json_path, target_size=960):
    """Process all images in a folder sequentially according to competition rules."""
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort to ensure consistent processing order
    print(f"Found {len(image_files)} images to process")
    
    all_detections = []
    
    # Start timer - after model is loaded but before processing any images
    timer_start = time.time()
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # LOAD image
        image_path = os.path.join(image_folder, img_file)
        im_pil = Image.open(image_path).convert("RGB")
        
        # PREPROCESS image
        resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, target_size)
        orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])
        
        transforms = T.Compose([T.ToTensor()])
        im_data = transforms(resized_im_pil).unsqueeze(0)
        
        # PERFORM_INFERENCE on image
        output = sess.run(
            output_names=None,
            input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()}
        )
        labels, boxes, scores = output
        
        # POSTPROCESS inference_result
        image_detections = process_detections(
            img_file,
            boxes[0],  # Single image, so we use [0]
            labels[0],
            scores[0],
            ratio,
            (pad_w, pad_h)
        )
        
        # SAVE result (append to our collection)
        all_detections.extend(image_detections)
    
    # End timer after all images are processed
    timer_end = time.time()
    elapsed_time = timer_end - timer_start
    
    # Calculate FPS according to competition formula
    num_images = len(image_files)
    fps = num_images / elapsed_time
    
    print(f"Processed {num_images} images in {elapsed_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    
    # Save final results to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"Results saved to {output_json_path}")
    return fps


def main(args):
    """Main function."""
    # Initialize the model (this part is not timed according to rules)
    sess = init_onnx_session(args.onnx)
    
    # Process the folder of images (this part is timed)
    fps = process_image_folder(
        args.image_folder,
        sess,
        args.output_json,
        target_size=args.image_size
    )
    
    print(f"Final FPS: {fps:.2f}")
    print("To calculate the final metric, you'll need to compute the F1-score and take the harmonic mean with FPS")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images.")
    parser.add_argument("--output_json", type=str, default="detections.json", help="Path to save output JSON.")
    parser.add_argument("--image_size", type=int, default=960, help="Target size for image preprocessing.")
    args = parser.parse_args()
    main(args)