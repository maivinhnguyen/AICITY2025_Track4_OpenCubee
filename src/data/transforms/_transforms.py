"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional, Tuple

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import json
import random
import math
import os

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
RandomAffine = register()(T.RandomAffine)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)
RandomRotation = register()(T.RandomRotation)
RandomPerspective = register()(T.RandomPerspective)

@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt

@register()
class CopyPaste(T.Transform):
    """
    CopyPaste augmentation optimized for FishEye8K dataset and DFINE training.
    
    Key features:
    - FishEye distortion awareness for realistic placement
    - Enhanced small object handling for traffic scenarios
    - DFINE-compatible label format preservation
    - Improved overlap detection with IoU thresholding
    - Balanced category sampling for traffic scenes
    """
    
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    
    def __init__(
        self,
        p: float = 0.6,
        target_categories: List[int] = [0, 1, 2, 3, 5, 7],  # person, bicycle, car, motorcycle, bus, truck
        max_paste_objects: int = 4,
        min_object_area: int = 400,
        max_object_area: int = 50000,
        iou_threshold: float = 0.15,
        edge_buffer: int = 20,
        fisheye_aware: bool = True,
        preserve_aspect: bool = True,
        scale_range: Tuple[float, float] = (0.7, 1.3),
    ) -> None:
        super().__init__()
        self.p = p
        self.target_categories = target_categories
        self.max_paste_objects = max_paste_objects
        self.min_object_area = min_object_area
        self.max_object_area = max_object_area
        self.iou_threshold = iou_threshold
        self.edge_buffer = edge_buffer
        self.fisheye_aware = fisheye_aware
        self.preserve_aspect = preserve_aspect
        self.scale_range = scale_range
        self._dataset_cache = None
        
        # FishEye8K specific: weight center region less for realistic placement
        self.center_weight = 0.3
    
    def forward(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Handle different input formats
        if len(inputs) == 2:
            image, target = inputs
            return self._apply_copy_paste(image, target)
        elif len(inputs) == 3:
            image, target, dataset = inputs
            self._dataset_cache = dataset
            result = self._apply_copy_paste(image, target)
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], result[1], dataset
            return result
        else:
            return inputs if len(inputs) > 1 else inputs[0]
    
    def _apply_copy_paste(self, image, target):
        """Apply copy-paste augmentation to image and target."""
        if self._dataset_cache is None:
            return image, target
        
        dataset = self._dataset_cache
        
        # Get multiple source images for better diversity
        paste_samples = []
        for _ in range(min(3, len(dataset))):
            try:
                paste_idx = torch.randint(0, len(dataset), (1,)).item()
                paste_image, paste_target = dataset.load_item(paste_idx)
                paste_samples.append((paste_image, paste_target))
            except:
                continue
        
        if not paste_samples:
            return image, target
        
        # Create working copies
        new_image = image.copy() if hasattr(image, 'copy') else F.to_pil_image(image)
        new_boxes = target.get('boxes', torch.empty(0, 4)).clone()
        new_labels = target.get('labels', torch.empty(0)).clone()
        new_areas = target.get('area', torch.empty(0)).clone()
        new_iscrowd = target.get('iscrowd', torch.empty(0, dtype=torch.int64)).clone()
        
        # Track pasted objects for category balancing
        pasted_count = 0
        category_counts = {cat: 0 for cat in self.target_categories}
        
        # Process each source image
        for paste_image, paste_target in paste_samples:
            if pasted_count >= self.max_paste_objects:
                break
                
            valid_objects = self._get_valid_objects(paste_target)
            if not valid_objects:
                continue
            
            # Balance categories - prefer underrepresented ones
            balanced_objects = self._balance_category_selection(valid_objects, category_counts)
            
            for obj_idx, (box, label, area) in enumerate(balanced_objects):
                if pasted_count >= self.max_paste_objects:
                    break
                
                paste_img = paste_image.copy() if hasattr(paste_image, 'copy') else F.to_pil_image(paste_image)
                
                success = self._paste_object(
                    paste_img, new_image, box, label, area,
                    new_boxes, new_labels, new_areas, new_iscrowd
                )
                
                if success:
                    pasted_count += 1
                    category_counts[label.item()] += 1
        
        # Convert back to appropriate format
        if isinstance(image, Image):
            new_image = Image(F.pil_to_tensor(new_image))
        
        # Update target with DFINE-compatible format
        target = self._create_updated_target(target, new_image, new_boxes, new_labels, new_areas, new_iscrowd)
        
        return new_image, target
    
    def _get_valid_objects(self, paste_target) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Extract valid objects for pasting based on category and size filters."""
        paste_labels = paste_target.get('labels', torch.empty(0))
        paste_boxes = paste_target.get('boxes', torch.empty(0, 4))
        paste_areas = paste_target.get('area', torch.empty(0))
        
        if len(paste_labels) == 0 or len(paste_boxes) == 0:
            return []
        
        valid_objects = []
        for i, (label, box, area) in enumerate(zip(paste_labels, paste_boxes, paste_areas)):
            if (label.item() in self.target_categories and 
                self.min_object_area <= area.item() <= self.max_object_area):
                valid_objects.append((box, label, area))
        
        return valid_objects
    
    def _balance_category_selection(self, valid_objects, category_counts) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Select objects with category balancing to avoid over-representation."""
        if not valid_objects:
            return []
        
        # Sort by category count (ascending) to prefer underrepresented categories
        valid_objects.sort(key=lambda x: category_counts.get(x[1].item(), 0))
        
        # Return up to remaining slots
        remaining_slots = self.max_paste_objects - sum(category_counts.values())
        return valid_objects[:remaining_slots]
    
    def _paste_object(self, paste_image, target_image, box, label, area, 
                     new_boxes, new_labels, new_areas, new_iscrowd) -> bool:
        """Paste a single object with fisheye-aware positioning."""
        
        # Extract object region
        object_crop, crop_coords = self._extract_object_crop(paste_image, box)
        if object_crop is None:
            return False
        
        # Find optimal placement position
        placement_pos = self._find_placement_position(
            target_image, object_crop, new_boxes, crop_coords
        )
        if placement_pos is None:
            return False
        
        new_x1, new_y1, scale_factor = placement_pos
        
        # Scale object if needed
        if scale_factor != 1.0 and self.preserve_aspect:
            new_w = int(object_crop.size[0] * scale_factor)
            new_h = int(object_crop.size[1] * scale_factor)
            object_crop = object_crop.resize((new_w, new_h), PIL.Image.LANCZOS)
        
        # Paste object
        target_image.paste(object_crop, (new_x1, new_y1))
        
        # Update annotations
        target_w, target_h = target_image.size
        new_x2 = new_x1 + object_crop.size[0]
        new_y2 = new_y1 + object_crop.size[1]
        
        # Create normalized bounding box
        new_box_norm = torch.tensor([
            new_x1 / target_w, new_y1 / target_h,
            new_x2 / target_w, new_y2 / target_h
        ])
        
        # Add to collections (in-place modification)
        new_boxes.data = torch.cat([new_boxes, new_box_norm.unsqueeze(0)], dim=0)
        new_labels.data = torch.cat([new_labels, label.unsqueeze(0)], dim=0)
        
        # Calculate new area
        new_area = object_crop.size[0] * object_crop.size[1]
        new_areas.data = torch.cat([new_areas, torch.tensor([new_area], dtype=new_areas.dtype)], dim=0)
        new_iscrowd.data = torch.cat([new_iscrowd, torch.tensor([0], dtype=torch.int64)], dim=0)
        
        return True
    
    def _extract_object_crop(self, paste_image, box) -> Tuple[Optional[PIL.Image.Image], Optional[Tuple[int, int, int, int]]]:
        """Extract object crop from source image."""
        img_w, img_h = paste_image.size
        
        # Handle different box formats and normalization
        if box.numel() != 4:
            return None, None
        
        x1, y1, x2, y2 = box.tolist()
        
        # Denormalize if needed
        if x2 <= 1.0 and y2 <= 1.0:
            x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
        
        # Ensure valid coordinates
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(img_w, x2)), int(min(img_h, y2))
        
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        crop = paste_image.crop((x1, y1, x2, y2))
        return crop, (x1, y1, x2, y2)
    
    def _find_placement_position(self, target_image, object_crop, existing_boxes, 
                               original_coords) -> Optional[Tuple[int, int, float]]:
        """Find optimal placement position with fisheye awareness."""
        target_w, target_h = target_image.size
        crop_w, crop_h = object_crop.size
        
        max_attempts = 30
        
        for attempt in range(max_attempts):
            # Generate position with fisheye awareness
            if self.fisheye_aware:
                new_x1, new_y1 = self._generate_fisheye_position(target_w, target_h, crop_w, crop_h)
            else:
                new_x1 = random.randint(self.edge_buffer, max(self.edge_buffer, target_w - crop_w - self.edge_buffer))
                new_y1 = random.randint(self.edge_buffer, max(self.edge_buffer, target_h - crop_h - self.edge_buffer))
            
            # Apply random scaling
            scale_factor = random.uniform(*self.scale_range) if attempt > 15 else 1.0
            scaled_w = int(crop_w * scale_factor)
            scaled_h = int(crop_h * scale_factor)
            
            # Ensure scaled object fits
            if (new_x1 + scaled_w >= target_w - self.edge_buffer or 
                new_y1 + scaled_h >= target_h - self.edge_buffer):
                continue
            
            new_x2 = new_x1 + scaled_w
            new_y2 = new_y1 + scaled_h
            
            # Create test box for overlap checking
            test_box = torch.tensor([new_x1/target_w, new_y1/target_h, new_x2/target_w, new_y2/target_h])
            
            # Check overlap with existing boxes
            if not self._has_significant_overlap(test_box, existing_boxes):
                return new_x1, new_y1, scale_factor
        
        return None
    
    def _generate_fisheye_position(self, target_w: int, target_h: int, crop_w: int, crop_h: int) -> Tuple[int, int]:
        """Generate position considering fisheye distortion characteristics."""
        center_x, center_y = target_w // 2, target_h // 2
        
        if random.random() < self.center_weight:
            # Place near center (less preferred for fisheye)
            x_range = (center_x - crop_w//2, center_x + crop_w//2)
            y_range = (center_y - crop_h//2, center_y + crop_h//2)
        else:
            # Place in outer regions (more realistic for fisheye)
            if random.random() < 0.5:
                # Horizontal edges
                x_range = (self.edge_buffer, target_w - crop_w - self.edge_buffer)
                y_range = (self.edge_buffer, center_y - crop_h//2) if random.random() < 0.5 else (center_y + crop_h//2, target_h - crop_h - self.edge_buffer)
            else:
                # Vertical edges  
                x_range = (self.edge_buffer, center_x - crop_w//2) if random.random() < 0.5 else (center_x + crop_w//2, target_w - crop_w - self.edge_buffer)
                y_range = (self.edge_buffer, target_h - crop_h - self.edge_buffer)
        
        x_range = (max(self.edge_buffer, x_range[0]), min(target_w - crop_w - self.edge_buffer, x_range[1]))
        y_range = (max(self.edge_buffer, y_range[0]), min(target_h - crop_h - self.edge_buffer, y_range[1]))
        
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            # Fallback to simple random placement
            x_range = (self.edge_buffer, max(self.edge_buffer, target_w - crop_w - self.edge_buffer))
            y_range = (self.edge_buffer, max(self.edge_buffer, target_h - crop_h - self.edge_buffer))
        
        new_x = random.randint(int(x_range[0]), int(x_range[1])) if x_range[1] > x_range[0] else int(x_range[0])
        new_y = random.randint(int(y_range[0]), int(y_range[1])) if y_range[1] > y_range[0] else int(y_range[0])
        
        return new_x, new_y
    
    def _has_significant_overlap(self, test_box: torch.Tensor, existing_boxes: torch.Tensor) -> bool:
        """Check if test box has significant overlap with existing boxes."""
        if len(existing_boxes) == 0:
            return False
        
        for existing_box in existing_boxes:
            iou = self._calculate_iou(test_box, existing_box)
            if iou > self.iou_threshold:
                return True
        
        return False
    
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Calculate Intersection over Union of two boxes."""
        x1_max = torch.max(box1[0], box2[0])
        y1_max = torch.max(box1[1], box2[1])
        x2_min = torch.min(box1[2], box2[2])
        y2_min = torch.min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return float(intersection / union) if union > 0 else 0.0
    
    def _create_updated_target(self, original_target: Dict[str, Any], new_image, 
                             new_boxes: torch.Tensor, new_labels: torch.Tensor, 
                             new_areas: torch.Tensor, new_iscrowd: torch.Tensor) -> Dict[str, Any]:
        """Create updated target dictionary with DFINE-compatible format."""
        target = original_target.copy() if hasattr(original_target, 'copy') else dict(original_target)
        
        if len(new_boxes) > 0:
            # Get image size
            if hasattr(new_image, 'size'):
                img_size = new_image.size  # PIL Image (width, height)
                canvas_size = (img_size[1], img_size[0])  # (height, width)
            else:
                # Tensor image - shape is (C, H, W)
                canvas_size = new_image.shape[-2:]  # (height, width)
            
            # Ensure boxes are in DFINE-expected format
            target['boxes'] = BoundingBoxes(
                new_boxes, 
                format="XYXY", 
                canvas_size=canvas_size
            )
        
        target['labels'] = new_labels
        target['area'] = new_areas
        target['iscrowd'] = new_iscrowd
        
        # Preserve any additional DFINE-specific fields
        if 'image_id' in original_target:
            target['image_id'] = original_target['image_id']
        if 'orig_size' in original_target:
            if hasattr(new_image, 'size'):
                target['orig_size'] = torch.tensor([new_image.size[1], new_image.size[0]])
            else:
                target['orig_size'] = torch.tensor(new_image.shape[-2:])
        
        return target
