# --- START OF FILE torch_inf.py ---

"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.utils.data

import numpy as np
from PIL import Image, ImageDraw
import json
import datetime
import os
import glob
import sys
import cv2

# Ensure the path is correct based on your project structure
# If torch_inf.py is in experiments/xxx/, this should work.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Use the original import from torch_inf.py
from engine.core import YAMLConfig


def get_image_Id(img_name):
  img_name = img_name.split('.png')[0]
  sceneList = ['M', 'A', 'E', 'N']
  cameraIndx = int(img_name.split('_')[0].split('camera')[1])
  sceneIndx = sceneList.index(img_name.split('_')[1])
  frameIndx = int(img_name.split('_')[2])
  imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
  return imageId


class CocoImageDataset(torch.utils.data.Dataset):
    """Dataset class for loading images for COCO format output."""
    def __init__(self, image_files_list, transforms):
        self.image_files = image_files_list
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        img_filename_with_ext = os.path.basename(image_path)
        img_filename_stem, _ = os.path.splitext(img_filename_with_ext)

        try:
            # Attempt to generate ID first using the provided function
            try:
                coco_image_id = get_image_Id(img_filename_stem)
            except ValueError as e_id: # Catch specific errors from get_image_Id
                # Re-raise to be caught by the outer exception handler, adding context
                raise ValueError(f"ID generation failed for '{img_filename_with_ext}' (stem: '{img_filename_stem}'): {e_id}") from e_id

            im_pil = Image.open(image_path).convert('RGB')
            original_width, original_height = im_pil.size
            im_data = self.transforms(im_pil) # Apply transforms (Resize, ToTensor)

            return {
                "im_data": im_data,
                # Store original size as W, H for the model's postprocessor
                "original_size_for_model": torch.tensor([original_width, original_height], dtype=torch.float32),
                "coco_image_id": coco_image_id,
                "file_name": img_filename_with_ext, # Keep filename for reference if needed
                "status": "ok"
            }
        except Exception as e:
            print(f"Error processing image {image_path} (or its ID): {e}", file=sys.stderr)
            # Return a dummy item with status 'error' to be filtered by collate_fn
            return {
                "im_data": torch.zeros((3, 960, 960)), # Dummy data matching expected tensor shape
                "original_size_for_model": torch.tensor([0,0], dtype=torch.float32),
                "coco_image_id": -1, # Invalid ID indicates error
                "file_name": img_filename_with_ext,
                "status": "error"
            }


def coco_collate_fn_revised(batch):
    """Collate function to filter errors and stack batch data."""
    # Filter out items that had errors during loading/ID generation
    batch = [item for item in batch if item["status"] == "ok"]
    if not batch: # If all items in the batch failed
        return {
            "im_data_batch": torch.empty(0, 3, 960, 960), # Match dims but size 0
            "original_sizes_for_model": torch.empty(0, 2),
            "coco_image_ids": [],
            "file_names": [],
            "empty_batch": True # Flag to indicate the batch is unusable
        }

    # Stack tensors for model input
    im_data_batch = torch.stack([item['im_data'] for item in batch])
    original_sizes_for_model = torch.stack([item['original_size_for_model'] for item in batch])

    # Collect metadata
    coco_image_ids = [item['coco_image_id'] for item in batch]
    file_names = [item['file_name'] for item in batch]

    return {
        "im_data_batch": im_data_batch,
        "original_sizes_for_model": original_sizes_for_model,
        "coco_image_ids": coco_image_ids,
        "file_names": file_names,
        "empty_batch": False
    }


def draw_detections(image_pil, labels, boxes, scores, threshold_map, default_threshold):
    """Draws filtered detections on a PIL image."""
    draw_obj = ImageDraw.Draw(image_pil) # Modifies image_pil in place

    for i in range(len(labels)):
        label = labels[i].item() # Get integer class label
        score = scores[i].item()
        box = boxes[i].tolist() # Convert tensor box to list [x1, y1, x2, y2]

        # Use class-specific threshold from map, fallback to default_threshold
        # Ensure label used as key is an integer
        threshold = threshold_map.get(int(label), default_threshold)

        if score >= threshold:
            draw_obj.rectangle(box, outline='red', width=2)

            # Prepare text and position
            text = f"{int(label)} {round(score, 2)}" # Display integer label
            text_x = box[0]
            text_y = box[1] - 10 if box[1] >= 10 else box[1] # Adjust text position up

            # Optional: Add background to text for better visibility
            # text_bbox = draw_obj.textbbox((text_x, text_y), text)
            # draw_obj.rectangle(text_bbox, fill='white')
            draw_obj.text((text_x, text_y), text, fill='blue') # Changed fill color

    return image_pil # Return the modified image


def process_directory_to_coco(model, device, input_dir, output_json, threshold_map, default_threshold, batch_size, num_workers):
    """
    Processes all images in a directory, applies per-class thresholds,
    and outputs results in COCO JSON format.
    """
    all_detections_list = [] # Initialize an empty list for COCO format detections

    # Find all supported image files
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + \
                         glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(input_dir, "*.png")) + \
                         glob.glob(os.path.join(input_dir, "*.bmp")))

    if not image_files:
        print(f"Warning: No images found in directory: {input_dir}")
        # Save an empty JSON list if no images are found
        # Ensure output directory exists for empty json file as well
        output_dir = os.path.dirname(output_json)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory for empty results: {output_dir}")
            except OSError as e:
                print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
                # Potentially exit or handle differently if directory creation is critical
        with open(output_json, 'w') as f:
            json.dump(all_detections_list, f, indent=2)
        print(f"Empty results list saved to {output_json}")
        return

    # Define transformations (Resize to model input size, Convert to Tensor)
    transforms_val = T.Compose([
        T.Resize((960, 960)), # Assuming model input size is 640x640
        T.ToTensor(),
    ])

    dataset = CocoImageDataset(image_files, transforms_val)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Important for consistent order if needed, usually false for inference
        num_workers=num_workers,
        collate_fn=coco_collate_fn_revised, # Use the revised collate function
        pin_memory=True if 'cuda' in device.type else False # Use pin_memory for faster GPU transfer
    )

    print(f"Found {len(image_files)} images. Starting processing for COCO JSON output...")
    start_time = datetime.datetime.now()

    processed_image_count = 0
    total_detections_count = 0
    failed_image_count = 0

    with torch.no_grad(): # Disable gradient calculations for inference
        for batch_data in dataloader:
            # Skip batch if collate_fn flagged it as empty/failed
            if batch_data.get("empty_batch", False):
                num_potential_failed = batch_size # Estimate failed count
                failed_image_count += num_potential_failed
                print(f"Skipping an empty or fully failed batch (approx. {num_potential_failed} images).", file=sys.stderr)
                continue

            im_data_batch = batch_data['im_data_batch'].to(device)
            # Ensure original sizes are on the correct device
            orig_sizes_batch = batch_data['original_sizes_for_model'].to(device)

            # Model inference
            outputs = model(im_data_batch, orig_sizes_batch)
            # Output format: labels_batch, boxes_batch, scores_batch (lists of tensors)
            labels_batch, boxes_batch, scores_batch = outputs

            # Process results for each image in the batch
            for i in range(len(batch_data['coco_image_ids'])):
                processed_image_count += 1
                current_image_id = batch_data['coco_image_ids'][i]

                # Get predictions for the current image
                labels = labels_batch[i].detach().cpu() # Move to CPU for processing
                boxes = boxes_batch[i].detach().cpu()   # Shape: [N, 4] (x1, y1, x2, y2)
                scores = scores_batch[i].detach().cpu() # Shape: [N]

                # Apply per-class threshold filtering
                for k in range(len(labels)):
                    category_id = int(labels[k].item()) # Get class label (integer)
                    score = scores[k].item()       # Get confidence score
                    # Get threshold: Use map if category_id is a key, else use default
                    threshold = threshold_map.get(category_id, default_threshold)

                    if score >= threshold:
                        # Box format for COCO is [x, y, width, height]
                        x1, y1, x2, y2 = boxes[k].tolist()
                        # Basic check for valid box dimensions
                        if x2 > x1 and y2 > y1:
                            width = x2 - x1
                            height = y2 - y1
                            # Format to 2 decimal places for bbox, 4 for score as per some conventions
                            bbox_coco = [float(f"{val:.2f}") for val in [x1, y1, width, height]] 

                            detection_entry = {
                                "image_id": current_image_id,
                                "category_id": category_id,
                                "bbox": bbox_coco,
                                "score": float(f"{score:.4f}") # Format score
                            }
                            all_detections_list.append(detection_entry)
                            total_detections_count += 1
                        # else: # Optional: Log invalid boxes
                        #     print(f"Warning: Invalid box coordinates [{x1}, {y1}, {x2}, {y2}] for image {current_image_id}, skipping.")


                if processed_image_count % 100 == 0: # Log progress periodically
                     print(f"Processed {processed_image_count}/{len(image_files)} images...")


    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    # Save the results list to the specified JSON file
    # Ensure output directory exists
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty string
        try:
            os.makedirs(output_dir, exist_ok=True) # exist_ok=True avoids error if dir already exists
            print(f"Ensured output directory exists: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
            # Decide if this is fatal or if we can proceed (e.g., by saving to current dir)

    with open(output_json, 'w') as f:
        json.dump(all_detections_list, f, indent=2)

    print("-" * 30)
    print(f"Processing complete. Results saved to: {output_json}")
    print(f"Successfully processed images: {processed_image_count}")
    if failed_image_count > 0 or processed_image_count < len(image_files):
         actual_failed = len(image_files) - processed_image_count
         print(f"Warning: Failed to process {actual_failed} images (due to loading errors or invalid format).")
    print(f"Total detections meeting thresholds: {total_detections_count}")
    if elapsed_time > 0 and processed_image_count > 0:
        fps = processed_image_count / elapsed_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {fps:.2f} FPS")
    else:
        print(f"Time elapsed: {elapsed_time:.2f} seconds.")
    print("-" * 30)


def process_single_image(model, device, file_path, threshold_map, default_threshold):
    """Processes a single image and saves the result with drawn boxes."""
    try:
        im_pil = Image.open(file_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image file {file_path}: {e}", file=sys.stderr)
        return

    w, h = im_pil.size
    # Model expects original size as a batch of tensors [[W, H]]
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(device)

    # Define transformations for a single image
    transforms = T.Compose([
        T.Resize((960, 960)),
        T.ToTensor(),
    ])
    # Add batch dimension (unsqueeze(0)) and send to device
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(im_data, orig_size)
        # Outputs are lists of tensors, even for a single image batch
        labels, boxes, scores = outputs

    # We processed one image, so take the first element from the output lists
    labels_img = labels[0].cpu()
    boxes_img = boxes[0].cpu()
    scores_img = scores[0].cpu()

    # Draw detections on the original PIL image
    drawn_image = draw_detections(im_pil.copy(), labels_img, boxes_img, scores_img, threshold_map, default_threshold)

    output_path = 'torch_results.jpg'
    drawn_image.save(output_path)
    print(f"Image processing complete. Result saved as '{output_path}'.")


def process_video_batched(model, device, file_path, threshold_map, default_threshold, batch_size):
    """Processes a video using batching and saves the result with drawn boxes."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {file_path}", file=sys.stderr)
        return

    # Get video properties
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output video writer
    output_video_path = 'torch_results.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps_video, (orig_w, orig_h))

    # Define transformations for video frames
    transforms = T.Compose([
        T.Resize((960, 960)),
        T.ToTensor(),
    ])

    frames_buffer_cv2 = [] # Stores raw OpenCV frames for the current batch
    processed_frame_count = 0

    print("Processing video frames...")
    start_time = datetime.datetime.now()

    with torch.no_grad():
        while True:
            ret, frame_cv2 = cap.read()

            # Add frame to buffer if read successfully
            if ret:
                frames_buffer_cv2.append(frame_cv2)

            # Process the batch if it's full OR if it's the end of the video and buffer isn't empty
            if len(frames_buffer_cv2) == batch_size or (not ret and len(frames_buffer_cv2) > 0):
                if not frames_buffer_cv2: # Should not happen if logic is correct, but safeguard
                     break

                # Convert CV2 frames (BGR) to PIL images (RGB)
                pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_buffer_cv2]

                # Prepare batch of original sizes (W, H) for the model
                original_sizes_for_model = torch.tensor(
                    [[p.size[0], p.size[1]] for p in pil_images], dtype=torch.float32
                ).to(device)

                # Apply transforms and stack into a batch tensor
                im_data_batch = torch.stack([transforms(p) for p in pil_images]).to(device)

                # Model inference
                outputs = model(im_data_batch, original_sizes_for_model)
                labels_batch, boxes_batch, scores_batch = outputs # Lists of tensors

                # Draw detections on each frame in the batch
                for i in range(len(pil_images)):
                    labels_img = labels_batch[i].cpu()
                    boxes_img = boxes_batch[i].cpu()
                    scores_img = scores_batch[i].cpu()

                    # Draw on the corresponding PIL image
                    drawn_pil_image = draw_detections(pil_images[i], labels_img, boxes_img, scores_img, threshold_map, default_threshold)

                    # Convert drawn PIL image back to CV2 format (BGR) for writing
                    processed_cv2_frame = cv2.cvtColor(np.array(drawn_pil_image), cv2.COLOR_RGB2BGR)
                    out_writer.write(processed_cv2_frame)
                    processed_frame_count += 1

                    if processed_frame_count % 30 == 0: # Log progress
                        print(f"Processed {processed_frame_count} frames...")

                # Clear the buffer for the next batch
                frames_buffer_cv2.clear()

            # If cap.read() returned False, it's the end of the video
            if not ret:
                break

    # Release resources
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows() # Close any OpenCV windows if they were opened

    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    print("-" * 30)
    print(f"Video processing complete. Result saved as '{output_video_path}'.")
    print(f"Processed {processed_frame_count} frames.")
    if elapsed_time > 0 and processed_frame_count > 0:
        fps_proc = processed_frame_count / elapsed_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {fps_proc:.2f} FPS")
    else:
         print(f"Time elapsed: {elapsed_time:.2f} seconds.")
    print("-" * 30)


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Disable pretrained backbone loading if specified in config (common practice)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    # Add similar checks for other backbone types if necessary

    # Load model checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        # Prioritize EMA weights if available
        if "ema" in checkpoint and checkpoint["ema"]:
             print("Loading EMA weights...")
             state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
             print("Loading model weights...")
             state = checkpoint["model"]
        else:
             # Handle checkpoints that might directly contain the state_dict
             print("Loading weights directly from checkpoint root...")
             state = checkpoint
             # Basic check if it looks like a state_dict
             if not isinstance(state, dict) or not any(k.endswith('.weight') or k.endswith('.bias') for k in state):
                 raise ValueError("Checkpoint format not recognized. Expected 'ema', 'model', or a raw state_dict.")

    else:
        # Resume argument is mandatory now based on the original script logic
        raise AttributeError("A checkpoint path must be provided via -r or --resume.")

    # --- Per-Class Threshold Handling ---
    threshold_map = {}
    if args.thresholds:
        try:
            # Split the comma-separated string and convert to floats
            threshold_values = [float(t.strip()) for t in args.thresholds.split(',')]
            # Create map: Class ID 0 maps to threshold_values[0], Class ID 1 to threshold_values[1], etc.
            # This assumes the model outputs class IDs starting from 0.
            threshold_map = {i: threshold for i, threshold in enumerate(threshold_values)}
            print(f"Using custom per-class thresholds (Class ID: Threshold): {threshold_map}")
            print(f"(Assuming list '{args.thresholds}' corresponds to class IDs 0, 1, 2, ...)")
        except ValueError:
            print(f"Error: Invalid format for --thresholds argument. Expected comma-separated floats (e.g., '0.1,0.2,0.3'). Got: {args.thresholds}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
             print(f"Error processing --thresholds argument '{args.thresholds}': {e}", file=sys.stderr)
             sys.exit(1)
    else:
        print(f"No specific per-class thresholds provided via --thresholds.")

    default_threshold = args.threshold # Use the general threshold as default
    print(f"Using default confidence threshold: {default_threshold} (for classes not specified in --thresholds or if --thresholds is not used)")


    # Load model structure from config and apply loaded weights
    cfg.model.load_state_dict(state)

    # Define the deployable model class (includes postprocessing)
    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy() # Get the deploy-ready model part
            self.postprocessor = cfg.postprocessor.deploy() # Get the deploy-ready postprocessor

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images) # Get raw model output
            processed_outputs = self.postprocessor(outputs, orig_target_sizes)
            return processed_outputs

    # Instantiate the deployable model and move to the specified device
    device = torch.device(args.device)
    model = DeployModel().to(device)
    model.eval() # Set model to evaluation mode

    # --- Input Processing ---
    input_path = args.input
    if os.path.isdir(input_path):
        print(f"Input is a directory: {input_path}. Processing for COCO JSON output.")
        process_directory_to_coco(
            model=model,
            device=device,
            input_dir=input_path,
            output_json=args.output,
            threshold_map=threshold_map,
            default_threshold=default_threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    elif os.path.isfile(input_path):
        print(f"Input is a file: {input_path}.")
        ext = os.path.splitext(input_path)[-1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            print("Processing as single image...")
            process_single_image(
                model=model,
                device=device,
                file_path=input_path,
                threshold_map=threshold_map,
                default_threshold=default_threshold
            )
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
             print("Processing as video...")
             process_video_batched(
                 model=model,
                 device=device,
                 file_path=input_path,
                 threshold_map=threshold_map,
                 default_threshold=default_threshold,
                 batch_size=args.batch_size
             )
        else:
             print(f"Error: Unsupported file type: {ext}. Provide an image (jpg, png, bmp), video (mp4, avi, mov), or a directory.", file=sys.stderr)
             sys.exit(1)
    else:
        print(f"Error: Input path not found or invalid: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="D-FINE Object Detection Inference")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to the model config file (.yaml)")
    parser.add_argument("-r", "--resume", type=str, required=True,
                        help="Path to the model checkpoint file (.pth)")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input image file, video file, or directory of images")
    parser.add_argument("-o", "--output", type=str, default="coco_results.json",
                        help="Path to the output COCO format JSON file (used only when --input is a directory)")
    parser.add_argument("-d", "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (e.g., 'cpu', 'cuda:0', 'cuda:1')")
    parser.add_argument("-t", "--threshold", type=float, default=0.4,
                        help="Default confidence threshold for detections (used if a class is not specified in --thresholds)")
    parser.add_argument("--thresholds", type=str, default=None,
                        help='Per-class confidence thresholds as a comma-separated string. '
                             'The list order corresponds to class IDs starting from 0. '
                             'Example for 5 classes (IDs 0-4): "0.5,0.45,0.6,0.5,0.55"')
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing (used for directory and video input)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for data loading (used for directory input)")

    args = parser.parse_args()

    # Basic validation
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.resume):
        print(f"Error: Checkpoint file not found at {args.resume}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.input):
         print(f"Error: Input path not found at {args.input}", file=sys.stderr)
         sys.exit(1)
    if args.device != "cpu" and not torch.cuda.is_available():
        print(f"Warning: Device set to '{args.device}' but CUDA is not available. Using CPU instead.", file=sys.stderr)
        args.device = "cpu"
    if args.threshold < 0 or args.threshold > 1:
        print(f"Warning: Default threshold {args.threshold} is outside the [0, 1] range.", file=sys.stderr)

    main(args)
# --- END OF FILE torch_inf.py ---