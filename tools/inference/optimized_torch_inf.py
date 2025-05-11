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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def get_image_Id(img_name_stem):
    """
    Generates an image ID based on the filename stem.
    Filename stem is expected to be in the format 'cameraX_SCENE_FRAME'.
    Example: 'camera0_M_123'
    """
    sceneList = ['M', 'A', 'E', 'N']
    try:
        parts = img_name_stem.split('_')
        if len(parts) < 3:
            raise ValueError(f"Filename stem '{img_name_stem}' does not have at least 3 parts separated by '_' (e.g., cameraX_SCENE_FRAME).")

        camera_part = parts[0]
        if not camera_part.startswith("camera") or len(camera_part) <= len("camera"):
            raise ValueError(f"Camera part '{camera_part}' in '{img_name_stem}' is not in 'cameraX' format.")
        cameraIndx = int(camera_part[len("camera"):])

        scene_part = parts[1]
        sceneIndx = sceneList.index(scene_part) # Raises ValueError if scene_part not in sceneList

        frame_part = parts[2]
        frameIndx = int(frame_part) # Raises ValueError if frame_part is not a valid integer

        imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    except ValueError as e:
        raise ValueError(f"Error parsing ID from filename stem '{img_name_stem}': {e}") from e
    except IndexError as e: # Should be caught by len(parts) check, but good for safety
        raise ValueError(f"Error parsing ID from filename stem '{img_name_stem}' due to unexpected format (IndexError): {e}") from e
    return imageId


class CocoImageDataset(torch.utils.data.Dataset):
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
            im_data = self.transforms(im_pil)

            return {
                "im_data": im_data,
                "original_size_for_model": torch.tensor([original_width, original_height], dtype=torch.float32),  # W, H
                "coco_image_id": coco_image_id, # Use new ID
                "original_width": original_width,
                "original_height": original_height,
                "file_name": img_filename_with_ext,
                "status": "ok"
            }
        except Exception as e:
            print(f"Error processing image {image_path} (or its ID): {e}", file=sys.stderr)
            # Return a dummy item or skip, here returning status to filter later
            return {
                "im_data": torch.zeros((3, 640, 640)), # Dummy data
                "original_size_for_model": torch.tensor([0,0], dtype=torch.float32),
                "coco_image_id": idx + 1, # Fallback ID for error item (won't be used if filtered)
                "original_width": 0,
                "original_height": 0,
                "file_name": img_filename_with_ext,
                "status": "error"
            }


def coco_collate_fn_revised(batch):
    # Filter out items that had errors during loading
    batch = [item for item in batch if item["status"] == "ok"]
    if not batch: # If all items in batch had errors
        return {
            "im_data_batch": torch.empty(0), # Or handle as needed
            "original_sizes_for_model": torch.empty(0),
            "coco_image_ids": [],
            "original_widths": [],
            "original_heights": [],
            "file_names": [],
            "empty_batch": True
        }

    return {
        "im_data_batch": torch.stack([item['im_data'] for item in batch]),
        "original_sizes_for_model": torch.stack([item['original_size_for_model'] for item in batch]),
        "coco_image_ids": [item['coco_image_id'] for item in batch],
        "original_widths": [item['original_width'] for item in batch],
        "original_heights": [item['original_height'] for item in batch],
        "file_names": [item['file_name'] for item in batch],
        "empty_batch": False
    }


def process_directory_to_coco(model, device, input_dir, output_json, conf_threshold=0.4, batch_size=8, num_workers=4):
    """
    Process all images in a directory and output results in the specified JSON list format.
    """
    all_detections_list = [] # Initialize an empty list for the output

    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + \
                         glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(input_dir, "*.png")) + \
                         glob.glob(os.path.join(input_dir, "*.bmp")))
    
    if not image_files:
        print("No images found in the directory.")
        # Save empty JSON list
        with open(output_json, 'w') as f:
            json.dump(all_detections_list, f, indent=2) # Save empty list
        print(f"Empty results list saved to {output_json}")
        return

    transforms_val = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    dataset = CocoImageDataset(image_files, transforms_val)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=coco_collate_fn_revised,
        pin_memory=True if device != 'cpu' else False
    )
    
    print(f"Found {len(image_files)} images to process for detection output.")
    start_time = datetime.datetime.now()
    
    processed_image_count = 0
    total_detections_count = 0

    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data.get("empty_batch", False) or batch_data["im_data_batch"].numel() == 0:
                num_failed_in_batch = batch_size
                print(f"Skipping an empty or failed batch of approx {num_failed_in_batch} images.", file=sys.stderr)
                continue

            im_data_batch = batch_data['im_data_batch'].to(device)
            orig_sizes_batch = batch_data['original_sizes_for_model'].to(device)
            
            outputs = model(im_data_batch, orig_sizes_batch)
            labels_batch, boxes_batch, scores_batch = outputs
            
            for i in range(len(batch_data['coco_image_ids'])):
                processed_image_count += 1
                current_image_id = batch_data['coco_image_ids'][i]
                # original_width = batch_data['original_widths'][i] # Not needed for this output format directly
                # original_height = batch_data['original_heights'][i] # Not needed for this output format directly
                # file_name = batch_data['file_names'][i] # Not needed for this output format directly
                
                labels = labels_batch[i].detach()
                boxes = boxes_batch[i].detach()
                scores = scores_batch[i].detach()
                
                mask = scores > conf_threshold
                filtered_labels = labels[mask].cpu().numpy()
                filtered_boxes = boxes[mask].cpu().numpy()
                filtered_scores = scores[mask].cpu().numpy()
                
                for k_detection in range(len(filtered_labels)):
                    category_id = int(filtered_labels[k_detection])
                    
                    x1, y1, x2, y2 = filtered_boxes[k_detection]
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection_entry = {
                        "image_id": current_image_id,
                        "category_id": category_id,
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "score": float(filtered_scores[k_detection])
                    }
                    all_detections_list.append(detection_entry)
                    total_detections_count +=1
    
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    # The new format is a flat list, so no complex structure to build or sort here.
    
    with open(output_json, 'w') as f:
        json.dump(all_detections_list, f, indent=2)
    
    print(f"Detection results list saved to {output_json}")
    print(f"Successfully processed {processed_image_count} images, yielding {total_detections_count} total detections.")
    if processed_image_count < len(image_files):
        print(f"Warning: {len(image_files) - processed_image_count} images could not be processed.")

    if elapsed_time > 0 and processed_image_count > 0 :
        fps = processed_image_count / elapsed_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {fps:.2f} FPS")
    else:
        print(f"Time elapsed: {elapsed_time:.2f} seconds. No images processed or time too short for FPS.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = torch.device(args.device) # Use torch.device object
    model = Model().to(device)
    model.eval()

    if os.path.isdir(args.input):
        process_directory_to_coco(model, device, args.input, args.output, 
                                  args.threshold, args.batch_size, args.num_workers)
    elif os.path.isfile(args.input):
        file_path = args.input
        if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            process_image(model, device, file_path, args.threshold)
            print("Image processing complete. Result saved as 'torch_results.jpg'.")
        else:
            process_video_batched(model, device, file_path, args.threshold, args.batch_size)
    else:
        print(f"Input path {args.input} does not exist")


def process_image(model, device, file_path, conf_threshold=0.4):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(device) # W, H

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(im_data, orig_size)
    
    labels, boxes, scores = outputs # labels, boxes, scores are lists of tensors

    # Pass the single tensor from the list to draw
    draw([im_pil], labels=[labels[0]], boxes=[boxes[0]], scores=[scores[0]], 
         thrh=conf_threshold, output_path='torch_results.jpg')


def draw(images_pil, labels_batch, boxes_batch, scores_batch, thrh=0.4, output_path=None):
    # images_pil is a list of PIL images
    # labels_batch, boxes_batch, scores_batch are lists of tensors (one tensor per image)
    drawn_images = []
    for i, im_pil in enumerate(images_pil): # im_pil is a PIL image
        draw_obj = ImageDraw.Draw(im_pil) # Modifies im_pil in place

        scores_i = scores_batch[i].detach() # Tensor of scores for this image
        labels_i = labels_batch[i].detach() # Tensor of labels for this image
        boxes_i = boxes_batch[i].detach()   # Tensor of boxes for this image

        mask = scores_i > thrh
        filtered_labels = labels_i[mask]
        filtered_boxes = boxes_i[mask]
        filtered_scores = scores_i[mask]

        for j, b_tensor in enumerate(filtered_boxes):
            # Ensure box coordinates are on CPU for numpy/list conversion
            b_list = b_tensor.cpu().tolist() 
            draw_obj.rectangle(b_list, outline='red', width=2)
            
            # Ensure text coordinates are Python floats/ints
            text_x = b_list[0]
            text_y = b_list[1] - 10 if b_list[1] - 10 > 0 else b_list[1] # Adjust text position

            label_val = filtered_labels[j].item()
            score_val = round(filtered_scores[j].item(), 2)
            draw_obj.text((text_x, text_y), text=f"{label_val} {score_val}", fill='blue')
        
        if output_path:
            if len(images_pil) == 1:
                im_pil.save(output_path)
            else:
                base, ext = os.path.splitext(output_path)
                im_pil.save(f"{base}_{i}{ext}")
        drawn_images.append(im_pil)
    return drawn_images


def process_video_batched(model, device, file_path, conf_threshold=0.4, batch_size=4):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {file_path}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = 'torch_results.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps_video, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frames_buffer_cv2 = []
    processed_frame_count = 0
    
    print("Processing video frames...")
    start_time = datetime.datetime.now()

    with torch.no_grad():
        while True:
            ret, frame_cv2 = cap.read()
            if ret:
                frames_buffer_cv2.append(frame_cv2)
            
            if len(frames_buffer_cv2) == batch_size or (not ret and len(frames_buffer_cv2) > 0):
                pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_buffer_cv2]
                
                original_sizes_for_model = torch.tensor(
                    [[p.size[0], p.size[1]] for p in pil_images], dtype=torch.float32 # W, H
                ).to(device)
                
                im_data_batch = torch.stack([transforms(p) for p in pil_images]).to(device)

                outputs = model(im_data_batch, original_sizes_for_model)
                labels_batch, boxes_batch, scores_batch = outputs # lists of tensors

                # Draw detections on each frame in the batch
                drawn_pil_images = draw(pil_images, labels_batch, boxes_batch, scores_batch, 
                                        thrh=conf_threshold, output_path=None)

                for drawn_pil_image in drawn_pil_images:
                    processed_cv2_frame = cv2.cvtColor(np.array(drawn_pil_image), cv2.COLOR_RGB2BGR)
                    out_writer.write(processed_cv2_frame)
                    processed_frame_count += 1
                    if processed_frame_count % 30 == 0: # Log every 30 frames
                        print(f"Processed {processed_frame_count} frames...")
                
                frames_buffer_cv2.clear()

            if not ret: # End of video
                break
    
    cap.release()
    out_writer.release()
    
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()

    print(f"Video processing complete. Result saved as '{output_video_path}'.")
    print(f"Processed {processed_frame_count} frames.")
    if elapsed_time > 0 and processed_frame_count > 0:
        fps_proc = processed_frame_count / elapsed_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average processing speed: {fps_proc:.2f} FPS")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True, 
                      help='Input image file, video file, or directory of images')
    parser.add_argument('-o', '--output', type=str, default='test_pred.json',
                      help='Output JSON file in COCO format (for directory processing)')
    parser.add_argument('-d', '--device', type=str, default='cpu', help="Device to use ('cpu', 'cuda:0', etc.)")
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Confidence threshold for detections')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for directory/video processing')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading (directory processing)')
    args = parser.parse_args()
    main(args)