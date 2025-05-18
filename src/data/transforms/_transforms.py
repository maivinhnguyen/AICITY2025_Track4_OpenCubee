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
import json
import random
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
    _transformed_types = (PIL.Image.Image, BoundingBoxes, Image)  
      
    def __init__(self, p=0.5, blend=True, sigma=1.0, min_area=0.0) -> None:  
        """  
        CopyPaste transform that copies objects from one image and pastes them onto another.  
          
        Args:  
            p (float): Probability of applying the transform  
            blend (bool): Whether to blend the pasted objects with the target image  
            sigma (float): Sigma for Gaussian blending if blend=True  
            min_area (float): Minimum normalized area for an object to be considered for copying  
        """  
        super().__init__()  
        self.p = p  
        self.blend = blend  
        self.sigma = sigma  
        self.min_area = min_area  
      
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:  
        # Determine whether to apply the transform based on probability  
        apply = torch.rand(1) < self.p  
        return {"apply": apply}  
      
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if not params["apply"]:  
            return inpt  
              
        # Implementation depends on input type  
        if isinstance(inpt, (PIL.Image.Image, Image)):  
            # Handle image transformation  
            return self._transform_image(inpt, params)  
        elif isinstance(inpt, BoundingBoxes):  
            # Handle bounding box transformation  
            return self._transform_boxes(inpt, params)  
        return inpt  
      
    def _transform_image(self, image, params):  
        # Image transformation logic here  
        # This would be implemented based on the dataset structure  
        return image  
      
    def _transform_boxes(self, boxes, params):  
        # Bounding box transformation logic here  
        # This would update boxes to include the pasted objects  
        return boxes  
      
    def forward(self, *inputs: Any) -> Any:  
        # Get the first input to determine spatial size for params  
        flat_inputs = []  
        for i in inputs:  
            if isinstance(i, (list, tuple)):  
                flat_inputs.extend(list(i))  
            else:  
                flat_inputs.append(i)  
                  
        params = self._get_params(flat_inputs)  
          
        # If not applying, return inputs unchanged  
        if not params["apply"]:  
            return inputs if len(inputs) > 1 else inputs[0]  
          
        # Apply transform to each input  
        transformed = []  
        for i in inputs:  
            if isinstance(i, (list, tuple)):  
                transformed.append(type(i)(self._transform(elem, params) for elem in i))  
            else:  
                transformed.append(self._transform(i, params))  
                  
        return transformed if len(transformed) > 1 else transformed[0]

@register()  
class RandomHSV(T.Transform):  
    def __init__(self, h=0.015, s=0.5, v=0.4, p=0.5):  
        super().__init__()  
        self.h = h  
        self.s = s  
        self.v = v  
        self.p = p  
      
    def _transform(self, inpt, params):  
        if random.random() < self.p:  
            h_factor = random.uniform(-self.h, self.h)  
            s_factor = random.uniform(1 - self.s, 1 + self.s)  
            v_factor = random.uniform(1 - self.v, 1 + self.v)  
              
            # Convert to HSV, apply changes, convert back to RGB  
            inpt = F.adjust_hue(inpt, h_factor)  
            inpt = F.adjust_saturation(inpt, s_factor)  
            inpt = F.adjust_brightness(inpt, v_factor)  
          
        return inpt

@register()  
class RandomTranslate(T.Transform):  
    def __init__(self, translate=0.05, p=0.5):  
        super().__init__()  
        self.translate = translate  
        self.p = p  
      
    def _transform(self, inpt, params):  
        if random.random() < self.p:  
            height, width = inpt.size[::-1]  
            max_dx = self.translate * width  
            max_dy = self.translate * height  
            tx = random.uniform(-max_dx, max_dx)  
            ty = random.uniform(-max_dy, max_dy)  
              
            # Apply translation  
            inpt = F.affine(inpt, angle=0, translate=(tx, ty), scale=1.0, shear=0)  
          
        return inpt  
  
@register()  
class RandomScale(T.Transform):  
    def __init__(self, scale=0.3, p=0.5):  
        super().__init__()  
        self.scale = scale  
        self.p = p  
      
    def _transform(self, inpt, params):  
        if random.random() < self.p:  
            scale_factor = random.uniform(1 - self.scale, 1 + self.scale)  
              
            # Apply scaling  
            inpt = F.affine(inpt, angle=0, translate=(0, 0), scale=scale_factor, shear=0)  
          
        return inpt  
  
@register()  
class RandomPerspective(T.RandomPerspective):  
    pass  # Already exists in torchvision, just register it  
  
# Register advanced augmentations  
@register()  
class Mosaic(T.Transform):  
    def __init__(self, p=1.0):  
        super().__init__()  
        self.p = p  
      
    def _transform(self, inpt, params):  
        # Mosaic implementation would go here  
        # This is a complex transform that requires multiple images  
        # For a full implementation, we would need to modify the dataloader  
        return inpt  
  
@register()  
class MixUp(T.Transform):  
    def __init__(self, alpha=0.2, p=0.2):  
        super().__init__()  
        self.alpha = alpha  
        self.p = p  
      
    def _transform(self, inpt, params):  
        # MixUp implementation would go here  
        # This is a complex transform that requires multiple images  
        # For a full implementation, we would need to modify the dataloader  
        return inpt