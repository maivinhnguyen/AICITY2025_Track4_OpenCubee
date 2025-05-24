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
    _transformed_types = (  
        PIL.Image.Image,  
        Image,  
        Video,  
        Mask,  
        BoundingBoxes,  
    )  
      
    def __init__(self, p=0.5, target_categories=[0, 1, 3, 4], max_paste_objects=3):  
        super().__init__()  
        self.p = p  
        self.target_categories = target_categories  
        self.max_paste_objects = max_paste_objects  
          
    def forward(self, *inputs):  
        if torch.rand(1) >= self.p:  
            return inputs if len(inputs) > 1 else inputs[0]  
              
        image, target, dataset = inputs  
          
        # Get another random sample from dataset  
        paste_idx = torch.randint(0, len(dataset), (1,)).item()  
        paste_image, paste_target = dataset.load_item(paste_idx)  
          
        # Filter objects by target categories  
        paste_labels = paste_target.get('labels', [])  
        paste_boxes = paste_target.get('boxes', [])  
          
        if len(paste_labels) == 0 or len(paste_boxes) == 0:  
            return inputs if len(inputs) > 1 else inputs[0]  
              
        # Find objects with target category IDs  
        valid_indices = []  
        for i, label in enumerate(paste_labels):  
            if label.item() in self.target_categories:  
                valid_indices.append(i)  
                  
        if not valid_indices:  
            return inputs if len(inputs) > 1 else inputs[0]  
              
        # Randomly select objects to paste (up to max_paste_objects)  
        num_paste = min(len(valid_indices), self.max_paste_objects)  
        selected_indices = torch.tensor(valid_indices)[torch.randperm(len(valid_indices))[:num_paste]]  
          
        # Extract regions and paste them  
        new_image = image.copy()  
        new_boxes = target.get('boxes', []).clone() if len(target.get('boxes', [])) > 0 else torch.empty((0, 4))  
        new_labels = target.get('labels', []).clone() if len(target.get('labels', [])) > 0 else torch.empty((0,))  
        new_areas = target.get('area', []).clone() if len(target.get('area', [])) > 0 else torch.empty((0,))  
        new_iscrowd = target.get('iscrowd', []).clone() if len(target.get('iscrowd', [])) > 0 else torch.empty((0,))  
          
        for idx in selected_indices:  
            paste_box = paste_boxes[idx]  
            paste_label = paste_labels[idx]  
              
            # Convert normalized coordinates to pixel coordinates for cropping  
            img_w, img_h = paste_image.size  
            x1, y1, x2, y2 = paste_box * torch.tensor([img_w, img_h, img_w, img_h])  
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  
              
            # Crop object from paste image  
            object_crop = paste_image.crop((x1, y1, x2, y2))  
            crop_w, crop_h = object_crop.size  
              
            # Find non-overlapping position in target image  
            target_w, target_h = new_image.size  
            max_attempts = 50  
            placed = False  
              
            for _ in range(max_attempts):  
                # Random position ensuring object fits  
                new_x1 = torch.randint(0, max(1, target_w - crop_w), (1,)).item()  
                new_y1 = torch.randint(0, max(1, target_h - crop_h), (1,)).item()  
                new_x2 = new_x1 + crop_w  
                new_y2 = new_y1 + crop_h  
                  
                # Check for overlap with existing boxes  
                new_box_norm = torch.tensor([new_x1/target_w, new_y1/target_h, new_x2/target_w, new_y2/target_h])  
                  
                overlap = False  
                if len(new_boxes) > 0:  
                    # Calculate IoU with existing boxes  
                    for existing_box in new_boxes:  
                        iou = self._calculate_iou(new_box_norm, existing_box)  
                        if iou > 0.1:  # Threshold for overlap  
                            overlap = True  
                            break  
                  
                if not overlap:  
                    # Paste the object  
                    new_image.paste(object_crop, (new_x1, new_y1))  
                      
                    # Add new annotations  
                    new_boxes = torch.cat([new_boxes, new_box_norm.unsqueeze(0)], dim=0)  
                    new_labels = torch.cat([new_labels, paste_label.unsqueeze(0)], dim=0)  
                      
                    # Calculate area  
                    area = crop_w * crop_h  
                    new_areas = torch.cat([new_areas, torch.tensor([area])], dim=0)  
                    new_iscrowd = torch.cat([new_iscrowd, torch.tensor([0])], dim=0)  
                      
                    placed = True  
                    break  
          
        # Update target with new annotations  
        if len(new_boxes) > 0:  
            target['boxes'] = convert_to_tv_tensor(new_boxes, key="boxes", spatial_size=new_image.size[::-1])  
        target['labels'] = new_labels  
        target['area'] = new_areas  
        target['iscrowd'] = new_iscrowd  
          
        return new_image, target, dataset  
      
    def _calculate_iou(self, box1, box2):  
        """Calculate Intersection over Union of two boxes"""  
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
          
        return intersection / union if union > 0 else 0.0