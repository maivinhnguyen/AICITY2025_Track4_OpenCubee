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
import torch.nn.functional as NF
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
class CopyPaste(nn.Module):
    """
    Efficient Copy-Paste augmentation for DFINE model.
    Supports small object detection and category-specific copying.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        blend: bool = True,
        sigma: float = 1.0,
        min_area: float = 0.01,
        category_ids: Optional[List[int]] = None,
        small_object_threshold: float = 0.05,
        max_paste_objects: float = 0.05,  # Fraction of image area or number of objects
        paste_all_matching: bool = True,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        max_attempts: int = 50,
        iou_threshold: float = 0.3,
    ):
        super().__init__()
        self.p = p
        self.blend = blend
        self.sigma = sigma
        self.min_area = min_area
        self.category_ids = category_ids or []
        self.small_object_threshold = small_object_threshold
        self.max_paste_objects = max_paste_objects
        self.paste_all_matching = paste_all_matching
        self.scale_range = scale_range
        self.max_attempts = max_attempts
        self.iou_threshold = iou_threshold
        
        # Object cache for copy-paste between different images
        self.object_cache = []
        self.cache_size = 100
    
    def _calculate_bbox_area_ratio(self, bbox: torch.Tensor, img_size: Tuple[int, int]) -> float:
        """Calculate normalized area of bounding box."""
        if bbox.numel() == 0:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        img_area = img_size[0] * img_size[1]
        return float(bbox_area / img_area)
    
    def _is_small_object(self, bbox: torch.Tensor, img_size: Tuple[int, int]) -> bool:
        """Check if object is considered small based on area threshold."""
        area_ratio = self._calculate_bbox_area_ratio(bbox, img_size)
        return area_ratio < self.small_object_threshold
    
    def _should_copy_object(self, bbox: torch.Tensor, label: int, img_size: Tuple[int, int]) -> bool:
        """Determine if object should be copied based on criteria."""
        # Check category filter
        if self.category_ids and int(label) not in self.category_ids:
            return False
        
        # Check minimum area
        area_ratio = self._calculate_bbox_area_ratio(bbox, img_size)
        if area_ratio < self.min_area:
            return False
        
        # Priority for small objects
        if self._is_small_object(bbox, img_size):
            return True
        
        # Copy other matching categories with lower probability
        return random.random() < 0.7 if self.paste_all_matching else random.random() < 0.3
    
    def _extract_object_region(
        self, 
        image: torch.Tensor, 
        bbox: torch.Tensor,
        margin_ratio: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract object region with margin and create blend mask."""
        h, w = image.shape[-2:]
        x1, y1, x2, y2 = bbox.clamp(min=0)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Add margin
        bbox_w, bbox_h = x2 - x1, y2 - y1
        margin_x = max(1, int(bbox_w * margin_ratio))
        margin_y = max(1, int(bbox_h * margin_ratio))
        
        # Expand with margin but stay within image bounds
        x1_exp = max(0, x1 - margin_x)
        y1_exp = max(0, y1 - margin_y)
        x2_exp = min(w, x2 + margin_x)
        y2_exp = min(h, y2 + margin_y)
        
        # Extract region
        obj_region = image[..., y1_exp:y2_exp, x1_exp:x2_exp].clone()
        
        # Create blend mask
        region_h, region_w = obj_region.shape[-2:]
        mask = torch.ones((region_h, region_w), dtype=torch.float32, device=image.device)
        
        if self.blend and self.sigma > 0:
            # Create Gaussian blend mask
            center_x, center_y = region_w // 2, region_h // 2
            y_grid, x_grid = torch.meshgrid(
                torch.arange(region_h, dtype=torch.float32, device=image.device),
                torch.arange(region_w, dtype=torch.float32, device=image.device),
                indexing='ij'
            )
            
            # Gaussian falloff from center
            sigma_x = region_w * self.sigma * 0.1
            sigma_y = region_h * self.sigma * 0.1
            
            if sigma_x > 0 and sigma_y > 0:
                gaussian = torch.exp(-((x_grid - center_x) ** 2 / (2 * sigma_x ** 2) +
                                     (y_grid - center_y) ** 2 / (2 * sigma_y ** 2)))
                mask = gaussian
        
        # Relative bbox within extracted region
        rel_bbox = torch.tensor([
            x1 - x1_exp, y1 - y1_exp, x2 - x1_exp, y2 - y1_exp
        ], dtype=bbox.dtype, device=bbox.device)
        
        return obj_region, mask, rel_bbox
    
    def _augment_object(self, obj_region: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations to object region."""
        # Scale augmentation
        if random.random() < 0.6:
            scale = random.uniform(*self.scale_range)
            if scale != 1.0:
                new_h = max(1, int(obj_region.shape[-2] * scale))
                new_w = max(1, int(obj_region.shape[-1] * scale))
                
                obj_region = NF.interpolate(
                    obj_region.unsqueeze(0),
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                mask = NF.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
        
        # Color jittering for small objects
        if random.random() < 0.4:
            # Brightness
            brightness = random.uniform(0.85, 1.15)
            obj_region = torch.clamp(obj_region * brightness, 0, 1)
            
            # Contrast
            if random.random() < 0.5:
                contrast = random.uniform(0.85, 1.15)
                mean_val = obj_region.mean(dim=[-2, -1], keepdim=True)
                obj_region = torch.clamp((obj_region - mean_val) * contrast + mean_val, 0, 1)
        
        return obj_region, mask
    
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Calculate IoU between two bounding boxes."""
        # Intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Check if there's intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return float(intersection / union) if union > 0 else 0.0
    
    def _find_paste_location(
        self,
        target_img: torch.Tensor,
        obj_size: Tuple[int, int],
        existing_boxes: torch.Tensor
    ) -> Optional[Tuple[int, int]]:
        """Find valid location to paste object avoiding overlaps."""
        img_h, img_w = target_img.shape[-2:]
        obj_h, obj_w = obj_size
        
        # Ensure object fits in image
        if obj_h >= img_h or obj_w >= img_w:
            return None
        
        max_x = img_w - obj_w
        max_y = img_h - obj_h
        
        for _ in range(self.max_attempts):
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)
            
            # Create candidate box
            candidate_box = torch.tensor([paste_x, paste_y, paste_x + obj_w, paste_y + obj_h])
            
            # Check overlaps with existing boxes
            valid_location = True
            if existing_boxes.numel() > 0:
                for existing_box in existing_boxes:
                    iou = self._calculate_iou(candidate_box, existing_box)
                    if iou > self.iou_threshold:
                        valid_location = False
                        break
            
            if valid_location:
                return (paste_x, paste_y)
        
        return None
    
    def _paste_object(
        self,
        target_img: torch.Tensor,
        obj_region: torch.Tensor,
        mask: torch.Tensor,
        paste_location: Tuple[int, int]
    ) -> torch.Tensor:
        """Paste object region into target image with blending."""
        paste_x, paste_y = paste_location
        obj_h, obj_w = obj_region.shape[-2:]
        img_h, img_w = target_img.shape[-2:]
        
        # Ensure paste region fits
        end_x = min(paste_x + obj_w, img_w)
        end_y = min(paste_y + obj_h, img_h)
        actual_w = end_x - paste_x
        actual_h = end_y - paste_y
        
        if actual_w <= 0 or actual_h <= 0:
            return target_img
        
        # Crop object and mask if needed
        obj_crop = obj_region[..., :actual_h, :actual_w]
        mask_crop = mask[:actual_h, :actual_w]
        
        # Expand mask to match image channels
        if target_img.dim() == 3:
            mask_crop = mask_crop.unsqueeze(0).expand(target_img.shape[0], -1, -1)
        
        # Blend object into target
        result_img = target_img.clone()
        target_region = result_img[..., paste_y:end_y, paste_x:end_x]
        
        if self.blend:
            blended_region = obj_crop * mask_crop + target_region * (1 - mask_crop)
        else:
            blended_region = obj_crop
        
        result_img[..., paste_y:end_y, paste_x:end_x] = blended_region
        
        return result_img
    
    def _update_object_cache(self, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor):
        """Update object cache with current image objects."""
        if boxes.numel() == 0:
            return
        
        img_size = (image.shape[-1], image.shape[-2])  # (w, h)
        
        for i, (bbox, label) in enumerate(zip(boxes, labels)):
            if self._should_copy_object(bbox, label, img_size):
                obj_region, mask, rel_bbox = self._extract_object_region(image, bbox)
                
                cache_entry = {
                    'region': obj_region,
                    'mask': mask,
                    'label': label,
                    'original_bbox': rel_bbox
                }
                
                self.object_cache.append(cache_entry)
        
        # Limit cache size
        if len(self.object_cache) > self.cache_size:
            self.object_cache = self.object_cache[-self.cache_size:]
    
    def _apply_copy_paste(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply copy-paste augmentation."""
        if not self.object_cache:
            return image, boxes, labels
        
        img_size = (image.shape[-1], image.shape[-2])  # (w, h)
        img_area = img_size[0] * img_size[1]
        
        # Determine max objects to paste
        if self.max_paste_objects < 1.0:
            # Treat as fraction of image area or existing objects
            max_objects = max(1, int(len(boxes) * self.max_paste_objects))
        else:
            max_objects = int(self.max_paste_objects)
        
        # Filter cache for matching categories and small objects
        valid_objects = []
        for obj in self.object_cache:
            if not self.category_ids or int(obj['label']) in self.category_ids:
                # Prioritize small objects
                obj_area = obj['region'].shape[-2] * obj['region'].shape[-1]
                if obj_area < img_area * self.small_object_threshold * 2:  # 2x threshold for cache
                    valid_objects.append(obj)
        
        if not valid_objects:
            return image, boxes, labels
        
        # Shuffle and limit objects
        random.shuffle(valid_objects)
        objects_to_paste = valid_objects[:max_objects]
        
        # Apply pasting
        result_img = image.clone()
        new_boxes = []
        new_labels = []
        all_boxes = boxes.clone() if boxes.numel() > 0 else torch.empty((0, 4))
        
        for obj in objects_to_paste:
            # Augment object
            aug_region, aug_mask = self._augment_object(obj['region'], obj['mask'])
            
            # Find paste location
            paste_location = self._find_paste_location(
                result_img,
                (aug_region.shape[-2], aug_region.shape[-1]),
                all_boxes
            )
            
            if paste_location is None:
                continue
            
            # Paste object
            result_img = self._paste_object(result_img, aug_region, aug_mask, paste_location)
            
            # Create new bounding box
            paste_x, paste_y = paste_location
            new_box = torch.tensor([
                paste_x,
                paste_y,
                paste_x + aug_region.shape[-1],
                paste_y + aug_region.shape[-2]
            ], dtype=boxes.dtype, device=boxes.device)
            
            new_boxes.append(new_box)
            new_labels.append(obj['label'])
            
            # Add to existing boxes for overlap checking
            all_boxes = torch.cat([all_boxes, new_box.unsqueeze(0)], dim=0)
        
        # Combine original and new annotations
        if new_boxes:
            final_boxes = torch.cat([boxes, torch.stack(new_boxes)], dim=0)
            final_labels = torch.cat([labels, torch.stack(new_labels)], dim=0)
        else:
            final_boxes = boxes
            final_labels = labels
        
        return result_img, final_boxes, final_labels
    
    def forward(self, *inputs: Any) -> Any:
        """Apply copy-paste augmentation."""
        if random.random() >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Parse inputs
        if len(inputs) >= 2:
            image, targets = inputs[0], inputs[1]
        else:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Extract boxes and labels
        if isinstance(targets, dict):
            boxes = targets.get('boxes', torch.empty((0, 4)))
            labels = targets.get('labels', torch.empty((0,), dtype=torch.long))
        else:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Update cache with current objects
        self._update_object_cache(image, boxes, labels)
        
        # Apply copy-paste augmentation
        aug_image, aug_boxes, aug_labels = self._apply_copy_paste(image, boxes, labels)
        
        # Update targets
        new_targets = targets.copy() if isinstance(targets, dict) else {}
        new_targets['boxes'] = aug_boxes
        new_targets['labels'] = aug_labels
        
        result = [aug_image, new_targets] + list(inputs[2:])
        return result if len(result) > 1 else result[0]

    def __call__(self, *inputs: Any) -> Any:
        """Make the transform callable."""
        return self.forward(*inputs)