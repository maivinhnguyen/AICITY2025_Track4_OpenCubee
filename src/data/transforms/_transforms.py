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
    Simple CopyPaste augmentation.
    Copy objects from one image and paste them onto another with their bounding boxes.
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
        p: float = 0.3,
        target_categories: List[int] = [0, 1, 3, 4],  # person, bicycle, motorcycle, car
        max_paste_objects: int = 2,
    ) -> None:
        super().__init__()
        self.p = p
        self.target_categories = target_categories
        self.max_paste_objects = max_paste_objects
        self._dataset_cache = None
    
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
        """Apply simple copy-paste augmentation."""
        if self._dataset_cache is None:
            return image, target
        
        dataset = self._dataset_cache
        
        # Get a random source image
        try:
            paste_idx = torch.randint(0, len(dataset), (1,)).item()
            paste_image, paste_target = dataset.load_item(paste_idx)
        except:
            return image, target
        
        # Convert to PIL if needed
        if isinstance(image, Image):
            target_image = F.to_pil_image(image)
        else:
            target_image = image.copy()
            
        if isinstance(paste_image, Image):
            source_image = F.to_pil_image(paste_image)
        else:
            source_image = paste_image.copy()
        
        # Get valid objects from source
        valid_objects = self._get_valid_objects(paste_target)
        if not valid_objects:
            return image, target
        
        # Randomly select objects to paste
        num_paste = min(len(valid_objects), self.max_paste_objects)
        selected_objects = random.sample(valid_objects, num_paste)
        
        # Copy target annotations
        new_boxes = target.get('boxes', torch.empty(0, 4)).clone()
        new_labels = target.get('labels', torch.empty(0)).clone()
        new_areas = target.get('area', torch.empty(0)).clone()
        new_iscrowd = target.get('iscrowd', torch.empty(0, dtype=torch.int64)).clone()
        
        # Paste each selected object
        for box, label, area in selected_objects:
            success = self._paste_single_object(
                source_image, target_image, box, label, area,
                new_boxes, new_labels, new_areas, new_iscrowd
            )
        
        # Convert back to original format
        if isinstance(image, Image):
            target_image = Image(F.pil_to_tensor(target_image))
        
        # Create updated target
        target = self._update_target(target, target_image, new_boxes, new_labels, new_areas, new_iscrowd)
        
        return target_image, target
    
    def _get_valid_objects(self, paste_target) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get objects with target categories."""
        paste_labels = paste_target.get('labels', torch.empty(0))
        paste_boxes = paste_target.get('boxes', torch.empty(0, 4))
        paste_areas = paste_target.get('area', torch.empty(0))
        
        if len(paste_labels) == 0 or len(paste_boxes) == 0:
            return []
        
        valid_objects = []
        for i, (label, box, area) in enumerate(zip(paste_labels, paste_boxes, paste_areas)):
            if label.item() in self.target_categories:
                valid_objects.append((box, label, area))
        
        return valid_objects
    
    def _paste_single_object(self, source_image, target_image, box, label, area,
                            new_boxes, new_labels, new_areas, new_iscrowd) -> bool:
        """Paste a single object onto target image."""
        
        # Extract object from source image
        object_crop = self._extract_object_crop(source_image, box)
        if object_crop is None:
            return False
        
        # Find a random position in target image
        target_w, target_h = target_image.size
        crop_w, crop_h = object_crop.size
        
        # Simple random placement with basic bounds checking
        max_x = max(0, target_w - crop_w)
        max_y = max(0, target_h - crop_h)
        
        if max_x <= 0 or max_y <= 0:
            return False
        
        new_x = random.randint(0, max_x)
        new_y = random.randint(0, max_y)
        
        # Paste the object
        target_image.paste(object_crop, (new_x, new_y))
        
        # Create new bounding box
        new_x2 = new_x + crop_w
        new_y2 = new_y + crop_h
        
        # Normalize coordinates
        norm_box = torch.tensor([
            new_x / target_w,
            new_y / target_h,
            new_x2 / target_w,
            new_y2 / target_h
        ])
        
        # Add annotations
        new_boxes.data = torch.cat([new_boxes, norm_box.unsqueeze(0)], dim=0)
        new_labels.data = torch.cat([new_labels, label.unsqueeze(0)], dim=0)
        
        # Calculate new area
        new_area = crop_w * crop_h
        new_areas.data = torch.cat([new_areas, torch.tensor([new_area], dtype=new_areas.dtype)], dim=0)
        new_iscrowd.data = torch.cat([new_iscrowd, torch.tensor([0], dtype=torch.int64)], dim=0)
        
        return True
    
    def _extract_object_crop(self, source_image, box) -> Optional[PIL.Image.Image]:
        """Extract object crop from source image."""
        if box.numel() != 4:
            return None
        
        img_w, img_h = source_image.size
        x1, y1, x2, y2 = box.tolist()
        
        # Denormalize if coordinates are normalized
        if x2 <= 1.0 and y2 <= 1.0:
            x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
        
        # Ensure valid coordinates
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img_w, int(x2))
        y2 = min(img_h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return source_image.crop((x1, y1, x2, y2))
    
    def _update_target(self, original_target, new_image, new_boxes, new_labels, new_areas, new_iscrowd):
        """Update target dictionary with new annotations."""
        target = original_target.copy() if hasattr(original_target, 'copy') else dict(original_target)
        
        if len(new_boxes) > 0:
            # Get image dimensions
            if hasattr(new_image, 'size'):
                canvas_size = (new_image.size[1], new_image.size[0])  # (height, width)
            else:
                canvas_size = new_image.shape[-2:]  # (height, width)
            
            target['boxes'] = BoundingBoxes(
                new_boxes,
                format="XYXY",
                canvas_size=canvas_size
            )
        
        target['labels'] = new_labels
        target['area'] = new_areas
        target['iscrowd'] = new_iscrowd
        
        return target
