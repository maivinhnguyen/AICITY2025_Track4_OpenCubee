"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
import math
import numpy as np

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
class RandomRotate90(T.Transform):
    """Rotate the image by 90 degrees with the specified probability.
    Useful for fisheye datasets where orientation can vary.
    """
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        # 0: no rotation, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees
        k = random.randint(0, 3) if torch.rand(1) < self.p else 0
        return {"k": k}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        k = params["k"]
        if k == 0:
            return inpt
        
        if isinstance(inpt, BoundingBoxes):
            # For bounding boxes, we need to handle the rotation specially
            boxes = inpt.as_tensor()
            format = inpt.format.value.lower()
            spatial_size = getattr(inpt, _boxes_keys[1])
            h, w = spatial_size
            
            # Convert to xyxy format for easier rotation
            if format != "xyxy":
                boxes = torchvision.ops.box_convert(boxes, in_fmt=format, out_fmt="xyxy")
            
            # Rotate the boxes
            for _ in range(k):
                # For 90 degree rotation: (x, y) -> (y, w-x)
                boxes_new = boxes.clone()
                boxes_new[:, 0] = h - boxes[:, 3]  # new_x1 = h - y2
                boxes_new[:, 1] = boxes[:, 0]      # new_y1 = x1
                boxes_new[:, 2] = h - boxes[:, 1]  # new_x2 = h - y1
                boxes_new[:, 3] = boxes[:, 2]      # new_y2 = x2
                boxes = boxes_new
                # Swap height and width
                h, w = w, h
            
            # Convert back to original format if needed
            if format != "xyxy":
                boxes = torchvision.ops.box_convert(boxes, in_fmt="xyxy", out_fmt=format)
            
            # Create new BoundingBoxes object with rotated boxes
            return convert_to_tv_tensor(
                boxes, key="boxes", box_format=inpt.format, spatial_size=(h, w)
            )
        
        # For images, masks, etc.
        return F.rotate(inpt, 90 * k)


@register()
class RandomPatchGaussian(T.Transform):
    """Add random Gaussian noise patches to enhance detection of small objects.
    This helps the model learn to identify objects in noisy areas.
    """
    _transformed_types = (PIL.Image.Image, Image)

    def __init__(
        self,
        p: float = 0.5,
        num_patches: int = 5,
        patch_size_range: List[float] = [0.01, 0.05],
        mean: float = 0.0,
        std: float = 0.2
    ) -> None:
        super().__init__()
        self.p = p
        self.num_patches = num_patches
        self.patch_size_range = patch_size_range
        self.mean = mean
        self.std = std

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        apply = torch.rand(1) < self.p
        return {"apply": apply}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params["apply"]:
            return inpt
        
        # Convert PIL image to tensor if needed
        is_pil = isinstance(inpt, PIL.Image.Image)
        if is_pil:
            img_tensor = F.pil_to_tensor(inpt).float()
            if img_tensor.max() <= 1.0:
                img_tensor = img_tensor * 255.0
        else:
            img_tensor = inpt.as_tensor().clone()
            if img_tensor.max() <= 1.0:
                img_tensor = img_tensor * 255.0
        
        c, h, w = img_tensor.shape
        
        # Add random Gaussian patches
        for _ in range(self.num_patches):
            # Random patch size
            rel_size = random.uniform(self.patch_size_range[0], self.patch_size_range[1])
            patch_h = max(1, int(h * rel_size))
            patch_w = max(1, int(w * rel_size))
            
            # Random position
            x = random.randint(0, w - patch_w)
            y = random.randint(0, h - patch_h)
            
            # Create Gaussian noise
            noise = torch.randn(c, patch_h, patch_w) * (self.std * 255.0) + (self.mean * 255.0)
            
            # Apply the noise patch
            img_tensor[:, y:y+patch_h, x:x+patch_w] += noise
        
        # Clip values to valid range
        img_tensor = torch.clamp(img_tensor, 0.0, 255.0)
        
        # Convert back to original format
        if is_pil:
            img_tensor = img_tensor.to(torch.uint8)
            return F.to_pil_image(img_tensor)
        else:
            if inpt.as_tensor().max() <= 1.0:
                img_tensor = img_tensor / 255.0
            return Image(img_tensor)


@register()
class FilterSmallInstances(T.Transform):
    """Filter out small instances that may be hard to detect or irrelevant.
    This helps in focusing model training on meaningful objects.
    """
    _transformed_types = (BoundingBoxes,)

    def __init__(
        self,
        min_pixels: int = 9,
        min_visibility: float = 0.2,
    ) -> None:
        super().__init__()
        self.min_pixels = min_pixels
        self.min_visibility = min_visibility

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        boxes = inpt.as_tensor()
        if boxes.shape[0] == 0:
            return inpt
        
        format = inpt.format.value.lower()
        spatial_size = getattr(inpt, _boxes_keys[1])
            
        # Convert to xyxy format for area calculation
        if format != "xyxy":
            boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt=format, out_fmt="xyxy")
        else:
            boxes_xyxy = boxes.clone()
        
        # Calculate areas
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        areas = widths * heights
        
        # Filter out small instances
        keep = areas >= self.min_pixels
        
        # Check for visibility (boxes not too close to edges)
        h, w = spatial_size
        visible_area = torch.minimum(boxes_xyxy[:, 2], torch.tensor(w)) - torch.maximum(boxes_xyxy[:, 0], torch.tensor(0))
        visible_area *= torch.minimum(boxes_xyxy[:, 3], torch.tensor(h)) - torch.maximum(boxes_xyxy[:, 1], torch.tensor(0))
        visibility = visible_area / (areas + 1e-8)
        keep = keep & (visibility >= self.min_visibility)
        
        # Filter the boxes
        filtered_boxes = boxes[keep]
        
        # If there are no boxes left, keep at least one (the largest)
        if filtered_boxes.shape[0] == 0 and boxes.shape[0] > 0:
            max_idx = torch.argmax(areas)
            filtered_boxes = boxes[max_idx:max_idx+1]
        
        # Create new BoundingBoxes object with filtered boxes
        return convert_to_tv_tensor(
            filtered_boxes, key="boxes", box_format=inpt.format, spatial_size=spatial_size
        )

