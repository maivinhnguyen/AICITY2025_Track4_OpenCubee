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
import datetime # Retain for overall start/end
import time     # Add for more precise interval timing
import os
import glob
import sys
import cv2

# Ensure the path is correct based on your project structure
# If torch_inf.py is in experiments/xxx/, this should work.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Use the original import from torch_inf.py
from src.core import YAMLConfig


def get_image_Id(img_name):
  # Ensure we handle filenames with or without extensions correctly for the stem
  img_name_stem = os.path.splitext(img_name)[0]
  sceneList = ['M', 'A', 'E', 'N']
  try:
    cameraIndx = int(img_name_stem.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name_stem.split('_')[1])
    frameIndx = int(img_name_stem.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
  except (IndexError, ValueError) as e:
      raise ValueError(f"Filename '{img_name}' (stem: '{img_name_stem}') does not match expected format for ID generation (cameraX_SCENE_FRAME): {e}")
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
            try:
                coco_image_id = get_image_Id(img_filename_stem)
            except ValueError as e_id:
                raise ValueError(f"ID generation failed for '{img_filename_with_ext}' (stem: '{img_filename_stem}'): {e_id}") from e_id

            im_pil = Image.open(image_path).convert('RGB')
            original_width, original_height = im_pil.size
            im_data = self.transforms(im_pil)

            return {
                "im_data": im_data,
                "original_size_for_model": torch.tensor([original_width, original_height], dtype=torch.float32),
                "coco_image_id": coco_image_id,
                "file_name": img_filename_with_ext,
                "status": "ok"
            }
        except Exception as e:
            print(f"Error processing image {image_path} (or its ID): {e}", file=sys.stderr)
            return {
                "im_data": torch.zeros((3, 960, 960)),
                "original_size_for_model": torch.tensor([0,0], dtype=torch.float32),
                "coco_image_id": -1,
                "file_name": img_filename_with_ext,
                "status": "error"
            }


def coco_collate_fn_revised(batch):
    """Collate function to filter errors and stack batch data."""
    batch = [item for item in batch if item["status"] == "ok"]
    if not batch:
        return {
            "im_data_batch": torch.empty(0, 3, 960, 960),
            "original_sizes_for_model": torch.empty(0, 2),
            "coco_image_ids": [],
            "file_names": [],
            "empty_batch": True
        }

    im_data_batch = torch.stack([item['im_data'] for item in batch])
    original_sizes_for_model = torch.stack([item['original_size_for_model'] for item in batch])
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
    draw_obj = ImageDraw.Draw(image_pil)
    for i in range(len(labels)):
        label = labels[i].item()
        score = scores[i].item()
        box = boxes[i].tolist()
        threshold = threshold_map.get(int(label), default_threshold)
        if score >= threshold:
            draw_obj.rectangle(box, outline='red', width=2)
            text = f"{int(label)} {round(score, 2)}"
            text_x = box[0]
            text_y = box[1] - 10 if box[1] >= 10 else box[1]
            draw_obj.text((text_x, text_y), text, fill='blue')
    return image_pil


def process_directory_to_coco(model, device, input_dir, output_json, threshold_map, default_threshold, batch_size, num_workers):
    all_detections_list = []
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + \
                         glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(input_dir, "*.png")) + \
                         glob.glob(os.path.join(input_dir, "*.bmp")))

    if not image_files:
        print(f"Warning: No images found in directory: {input_dir}")
        output_dir = os.path.dirname(output_json)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
        with open(output_json, 'w') as f:
            json.dump(all_detections_list, f, indent=2)
        print(f"Empty results list saved to {output_json}")
        return

    transforms_val = T.Compose([
        T.Resize((960, 960)),
        T.ToTensor(),
    ])

    # --- TIMER: Data Loading and Dataset/Dataloader setup ---
    dataloader_setup_start_time = time.time()
    dataset = CocoImageDataset(image_files, transforms_val)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=coco_collate_fn_revised, pin_memory=True if 'cuda' in device.type else False
    )
    dataloader_setup_time = time.time() - dataloader_setup_start_time
    print(f"Dataloader setup time: {dataloader_setup_time:.4f} seconds")
    # ---

    print(f"Found {len(image_files)} images. Starting processing for COCO JSON output...")
    # Use datetime for overall as in original, time.time() for finer measures
    overall_loop_start_dt = datetime.datetime.now()


    processed_image_count = 0
    total_detections_count = 0
    failed_image_count = 0

    # --- MODIFICATION: Initialize timers ---
    total_batch_prep_time = 0.0  # Time to get batch ready for model (mostly .to(device))
    total_inference_time = 0.0
    total_batch_postprocess_time = 0.0 # Time for detaching, moving to CPU, and COCO formatting per batch

    with torch.no_grad():
        for batch_data in dataloader:
            # --- TIMER START: Batch Preparation (CPU -> GPU, etc.) ---
            batch_prep_start_time = time.time()

            if batch_data.get("empty_batch", False):
                failed_image_count += batch_size
                print(f"Skipping an empty or fully failed batch (approx. {batch_size} images).", file=sys.stderr)
                total_batch_prep_time += (time.time() - batch_prep_start_time) # Add time for skipped batch prep
                continue

            im_data_batch = batch_data['im_data_batch'].to(device)
            orig_sizes_batch = batch_data['original_sizes_for_model'].to(device)

            batch_prep_end_time = time.time()
            total_batch_prep_time += (batch_prep_end_time - batch_prep_start_time)
            # --- TIMER END: Batch Preparation ---

            # --- TIMER START: Inference for this batch ---
            batch_inference_start_time = time.time()
            if device.type == 'cuda':
                torch.cuda.synchronize()

            outputs = model(im_data_batch, orig_sizes_batch)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_inference_end_time = time.time()
            total_inference_time += (batch_inference_end_time - batch_inference_start_time)
            # --- TIMER END: Inference for this batch ---

            # --- TIMER START: Batch Postprocessing (results handling) ---
            batch_post_start_time = time.time()
            labels_batch, boxes_batch, scores_batch = outputs
            for i in range(len(batch_data['coco_image_ids'])):
                processed_image_count += 1
                current_image_id = batch_data['coco_image_ids'][i]
                labels = labels_batch[i].detach().cpu()
                boxes = boxes_batch[i].detach().cpu()
                scores = scores_batch[i].detach().cpu()

                for k in range(len(labels)):
                    category_id = int(labels[k].item())
                    score = scores[k].item()
                    threshold = threshold_map.get(category_id, default_threshold)
                    if score >= threshold:
                        x1, y1, x2, y2 = boxes[k].tolist()
                        if x2 > x1 and y2 > y1:
                            width = x2 - x1
                            height = y2 - y1
                            bbox_coco = [float(f"{val:.2f}") for val in [x1, y1, width, height]]
                            detection_entry = {
                                "image_id": current_image_id,
                                "category_id": category_id,
                                "bbox": bbox_coco,
                                "score": float(f"{score:.4f}")
                            }
                            all_detections_list.append(detection_entry)
                            total_detections_count += 1
                if processed_image_count > 0 and processed_image_count % 100 == 0:
                     print(f"Processed {processed_image_count}/{len(image_files)} images...")
            batch_post_end_time = time.time()
            total_batch_postprocess_time += (batch_post_end_time - batch_post_start_time)
            # --- TIMER END: Batch Postprocessing ---

    overall_loop_end_dt = datetime.datetime.now()
    overall_loop_elapsed_time = (overall_loop_end_dt - overall_loop_start_dt).total_seconds()

    # --- TIMER: JSON saving time ---
    json_save_start_time = time.time()
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
    with open(output_json, 'w') as f:
        json.dump(all_detections_list, f, indent=2)
    json_save_time = time.time() - json_save_start_time
    # ---

    print("-" * 30)
    print(f"Processing complete. Results saved to: {output_json}")
    print(f"Successfully processed images: {processed_image_count}")
    if failed_image_count > 0 or processed_image_count < len(image_files):
         actual_failed = len(image_files) - processed_image_count
         print(f"Warning: Failed to process {actual_failed} images.")
    print(f"Total detections meeting thresholds: {total_detections_count}")

    # --- MODIFICATION: Print detailed timings ---
    print(f"Total elapsed time (directory processing loop): {overall_loop_elapsed_time:.2f} seconds")
    print(f"  Time for DataLoader setup: {dataloader_setup_time:.4f} seconds")
    print(f"  Accumulated batch preparation time (e.g. .to(device)): {total_batch_prep_time:.2f} seconds")
    print(f"  Accumulated model inference time: {total_inference_time:.2f} seconds")
    print(f"  Accumulated batch post-processing time (CPU ops, list appends): {total_batch_postprocess_time:.2f} seconds")
    print(f"  Time for saving JSON results: {json_save_time:.4f} seconds")

    # Note: The sum of these accumulated times might not perfectly match overall_loop_elapsed_time
    # due to Python overheads not captured, and dataloader_setup_time being outside the main loop timer.
    # The `num_workers > 0` in DataLoader means actual image file reading and transforms
    # are pipelined and happen in background processes, so their time is not directly part of
    # `total_batch_prep_time` if the GPU is busy.

    if overall_loop_elapsed_time > 0 and processed_image_count > 0:
        fps_overall = processed_image_count / overall_loop_elapsed_time
        print(f"Average processing speed (overall loop): {fps_overall:.2f} FPS")
        if total_inference_time > 0 :
            fps_inference_only = processed_image_count / total_inference_time
            print(f"Average model inference speed (based on inference time only): {fps_inference_only:.2f} FPS")
    print("-" * 30)


def process_single_image(model, device, file_path, threshold_map, default_threshold):
    # --- TIMER: Preprocessing ---
    prep_start_time = time.time()
    try:
        im_pil = Image.open(file_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image file {file_path}: {e}", file=sys.stderr)
        return

    w, h = im_pil.size
    orig_size_cpu = torch.tensor([[w, h]], dtype=torch.float32) # CPU

    transforms = T.Compose([
        T.Resize((960, 960)),
        T.ToTensor(),
    ])
    im_data_cpu = transforms(im_pil).unsqueeze(0) # CPU

    # Move to device just before inference
    im_data_gpu = im_data_cpu.to(device)
    orig_size_gpu = orig_size_cpu.to(device)
    prep_time = time.time() - prep_start_time
    # ---

    # --- TIMER: Inference ---
    infer_start_time = time.time()
    if device.type == 'cuda': torch.cuda.synchronize()
    with torch.no_grad():
        outputs = model(im_data_gpu, orig_size_gpu)
    if device.type == 'cuda': torch.cuda.synchronize()
    infer_time = time.time() - infer_start_time
    # ---

    # --- TIMER: Postprocessing & Save ---
    post_save_start_time = time.time()
    labels, boxes, scores = outputs
    labels_img = labels[0].cpu()
    boxes_img = boxes[0].cpu()
    scores_img = scores[0].cpu()
    drawn_image = draw_detections(im_pil.copy(), labels_img, boxes_img, scores_img, threshold_map, default_threshold)
    output_path = 'torch_results.jpg'
    drawn_image.save(output_path)
    post_save_time = time.time() - post_save_start_time
    # ---

    print(f"Image processing complete. Result saved as '{output_path}'.")
    print(f"  Preprocessing time: {prep_time:.4f} s")
    print(f"  Inference time: {infer_time:.4f} s")
    print(f"  Postprocessing & Save time: {post_save_time:.4f} s")


def process_video_batched(model, device, file_path, threshold_map, default_threshold, batch_size):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {file_path}", file=sys.stderr)
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = 'torch_results.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps_video, (orig_w, orig_h))

    transforms_vid = T.Compose([
        T.Resize((960, 960)),
        T.ToTensor(),
    ])

    frames_buffer_cv2 = []
    processed_frame_count = 0
    print(f"Processing video: {file_path}")
    overall_vid_start_dt = datetime.datetime.now()

    # --- MODIFICATION: Initialize timers for video ---
    total_frame_read_time = 0.0
    total_frame_transform_time = 0.0 # Includes PIL conversion, tensor, to(device)
    total_vid_inference_time = 0.0
    total_vid_postprocess_draw_write_time = 0.0

    with torch.no_grad():
        while True:
            # --- TIMER: Frame Reading ---
            frame_read_start_time = time.time()
            ret, frame_cv2 = cap.read()
            total_frame_read_time += (time.time() - frame_read_start_time)
            # ---

            if ret:
                frames_buffer_cv2.append(frame_cv2)

            if len(frames_buffer_cv2) == batch_size or (not ret and len(frames_buffer_cv2) > 0):
                if not frames_buffer_cv2:
                     break

                # --- TIMER: Batch Frame Transformation & to(device) ---
                frame_transform_start_time = time.time()
                pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_buffer_cv2]
                original_sizes_for_model_cpu = torch.tensor(
                    [[p.size[0], p.size[1]] for p in pil_images], dtype=torch.float32
                )
                im_data_batch_cpu = torch.stack([transforms_vid(p) for p in pil_images])
                
                im_data_batch_gpu = im_data_batch_cpu.to(device)
                original_sizes_for_model_gpu = original_sizes_for_model_cpu.to(device)
                total_frame_transform_time += (time.time() - frame_transform_start_time)
                # ---

                # --- TIMER: Video Batch Inference ---
                vid_infer_start_time = time.time()
                if device.type == 'cuda': torch.cuda.synchronize()
                outputs = model(im_data_batch_gpu, original_sizes_for_model_gpu)
                if device.type == 'cuda': torch.cuda.synchronize()
                total_vid_inference_time += (time.time() - vid_infer_start_time)
                # ---

                # --- TIMER: Video Batch Postprocessing (Draw, Write) ---
                vid_post_draw_write_start_time = time.time()
                labels_batch, boxes_batch, scores_batch = outputs
                for i in range(len(pil_images)):
                    labels_img = labels_batch[i].cpu()
                    boxes_img = boxes_batch[i].cpu()
                    scores_img = scores_batch[i].cpu()
                    drawn_pil_image = draw_detections(pil_images[i], labels_img, boxes_img, scores_img, threshold_map, default_threshold)
                    processed_cv2_frame = cv2.cvtColor(np.array(drawn_pil_image), cv2.COLOR_RGB2BGR)
                    out_writer.write(processed_cv2_frame)
                    processed_frame_count += 1
                    if processed_frame_count > 0 and processed_frame_count % 30 == 0:
                        print(f"Processed {processed_frame_count} frames...")
                total_vid_postprocess_draw_write_time += (time.time() - vid_post_draw_write_start_time)
                # ---

                frames_buffer_cv2.clear()
            if not ret:
                break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    overall_vid_end_dt = datetime.datetime.now()
    overall_vid_elapsed_time = (overall_vid_end_dt - overall_vid_start_dt).total_seconds()

    print("-" * 30)
    print(f"Video processing complete. Result saved as '{output_video_path}'.")
    print(f"Processed {processed_frame_count} frames.")
    print(f"Total elapsed time for video: {overall_vid_elapsed_time:.2f} seconds")
    print(f"  Accumulated frame read time: {total_frame_read_time:.2f} s")
    print(f"  Accumulated frame transform & to(device) time: {total_frame_transform_time:.2f} s")
    print(f"  Accumulated video inference time: {total_vid_inference_time:.2f} s")
    print(f"  Accumulated video postprocess (draw/write) time: {total_vid_postprocess_draw_write_time:.2f} s")
    
    other_vid_time = overall_vid_elapsed_time - (total_frame_read_time + total_frame_transform_time + total_vid_inference_time + total_vid_postprocess_draw_write_time)
    print(f"  Other video processing time (loop overhead, buffer management, etc.): {other_vid_time:.2f} s")

    if overall_vid_elapsed_time > 0 and processed_frame_count > 0:
        fps_vid_overall = processed_frame_count / overall_vid_elapsed_time
        print(f"Average processing speed (overall video): {fps_vid_overall:.2f} FPS")
        if total_vid_inference_time > 0:
            fps_vid_inference_only = processed_frame_count / total_vid_inference_time
            print(f"Average video inference speed (based on inference time only): {fps_vid_inference_only:.2f} FPS")
    print("-" * 30)


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint and checkpoint["ema"]:
             print("Loading EMA weights...")
             state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
             print("Loading model weights...")
             state = checkpoint["model"]
        else:
             print("Loading weights directly from checkpoint root...")
             state = checkpoint
             if not isinstance(state, dict) or not any(k.endswith(('.weight', '.bias')) for k in state):
                 raise ValueError("Checkpoint format not recognized. Expected 'ema', 'model', or a raw state_dict.")
    else:
        raise AttributeError("A checkpoint path must be provided via -r or --resume.")

    threshold_map = {}
    if args.thresholds:
        try:
            threshold_values = [float(t.strip()) for t in args.thresholds.split(',')]
            threshold_map = {i: threshold for i, threshold in enumerate(threshold_values)}
            print(f"Using custom per-class thresholds: {threshold_map}")
        except Exception as e:
             print(f"Error processing --thresholds: {e}", file=sys.stderr)
             sys.exit(1)
    default_threshold = args.threshold
    print(f"Using default confidence threshold: {default_threshold}")

    # --- TIMER: Model loading ---
    model_load_start_time = time.time()
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self): # cfg is accessible from main's scope
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            processed_outputs = self.postprocessor(outputs, orig_target_sizes)
            return processed_outputs

    device = torch.device(args.device)
    model = DeployModel().to(device)
    model.eval()
    model_load_time = time.time() - model_load_start_time
    print(f"Model loading and setup time: {model_load_time:.4f} seconds")
    # ---

    input_path = args.input
    if os.path.isdir(input_path):
        print(f"Input is a directory: {input_path}. Processing for COCO JSON output.")
        process_directory_to_coco(
            model=model, device=device, input_dir=input_path, output_json=args.output,
            threshold_map=threshold_map, default_threshold=default_threshold,
            batch_size=args.batch_size, num_workers=args.num_workers
        )
    elif os.path.isfile(input_path):
        print(f"Input is a file: {input_path}.")
        ext = os.path.splitext(input_path)[-1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            print("Processing as single image...")
            process_single_image(
                model=model, device=device, file_path=input_path,
                threshold_map=threshold_map, default_threshold=default_threshold
            )
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
             print("Processing as video...")
             process_video_batched(
                 model=model, device=device, file_path=input_path,
                 threshold_map=threshold_map, default_threshold=default_threshold,
                 batch_size=args.batch_size
             )
        else:
             print(f"Error: Unsupported file type: {ext}.", file=sys.stderr)
             sys.exit(1)
    else:
        print(f"Error: Input path not found or invalid: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="D-FINE Object Detection Inference")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to model config (.yaml)")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image, video, or directory")
    parser.add_argument("-o", "--output", type=str, default="coco_results.json", help="Output COCO JSON (for directory input)")
    parser.add_argument("-d", "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device ('cpu', 'cuda:0')")
    parser.add_argument("-t", "--threshold", type=float, default=0.4, help="Default confidence threshold")
    parser.add_argument("--thresholds", type=str, default=None, help='Per-class thresholds "0.5,0.45,..."')
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (directory/video)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (directory)")

    args = parser.parse_args()
    if not os.path.exists(args.config): sys.exit(f"Error: Config not found: {args.config}")
    if not os.path.exists(args.resume): sys.exit(f"Error: Checkpoint not found: {args.resume}")
    if not os.path.exists(args.input): sys.exit(f"Error: Input not found: {args.input}")
    if args.device != "cpu" and not torch.cuda.is_available():
        print(f"Warning: Device '{args.device}' unavailable. Using CPU.", file=sys.stderr)
        args.device = "cpu"
    if not (0 <= args.threshold <= 1): print(f"Warning: Threshold {args.threshold} out of [0,1] range.", file=sys.stderr)

    main(args)
# --- END OF FILE torch_inf.py ---