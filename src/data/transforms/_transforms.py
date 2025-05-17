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
class FilterSmallInstance(T.Transform):  
    _transformed_types = (BoundingBoxes,)  
      
    def __init__(self, min_size=1, min_area=0.0, normalized=False) -> None:  
        """  
        Filter out small instances from bounding boxes.  
          
        Args:  
            min_size (int): Minimum size (width or height) in pixels  
            min_area (float): Minimum area as a fraction of image area (if normalized=True) or in pixels  
            normalized (bool): Whether min_area is normalized to image size  
        """  
        super().__init__()  
        self.min_size = min_size  
        self.min_area = min_area  
        self.normalized = normalized  
      
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if not isinstance(inpt, BoundingBoxes):  
            return inpt  
              
        boxes = inpt.as_tensor()  
        spatial_size = getattr(inpt, _boxes_keys[1])  
          
        # Calculate width and height  
        if inpt.format.value.lower() == 'xyxy':  
            widths = boxes[:, 2] - boxes[:, 0]  
            heights = boxes[:, 3] - boxes[:, 1]  
        elif inpt.format.value.lower() == 'xywh':  
            widths = boxes[:, 2]  
            heights = boxes[:, 3]  
        elif inpt.format.value.lower() == 'cxcywh':  
            widths = boxes[:, 2]  
            heights = boxes[:, 3]  
          
        # Calculate areas  
        areas = widths * heights  
          
        # Apply size filter  
        if self.normalized:  
            # If boxes are normalized, convert min_size to normalized units  
            norm_min_size_w = self.min_size / spatial_size[1]  # width  
            norm_min_size_h = self.min_size / spatial_size[0]  # height  
            mask = (widths >= norm_min_size_w) & (heights >= norm_min_size_h)  
        else:  
            mask = (widths >= self.min_size) & (heights >= self.min_size)  
          
        # Apply area filter  
        if self.min_area > 0:  
            if self.normalized:  
                area_mask = areas >= self.min_area  
            else:  
                # Convert absolute area to normalized  
                norm_min_area = self.min_area / (spatial_size[0] * spatial_size[1])  
                area_mask = areas >= norm_min_area  
            mask = mask & area_mask  
          
        # Filter boxes and labels  
        filtered_boxes = boxes[mask]  
          
        # Create new BoundingBoxes object with filtered data  
        result = convert_to_tv_tensor(  
            filtered_boxes,   
            key="boxes",   
            box_format=inpt.format.value,   
            spatial_size=spatial_size  
        )  
          
        # If there are additional fields in the input (like labels), filter them too  
        if hasattr(inpt, "extra_fields"):  
            for k, v in inpt.extra_fields.items():  
                if isinstance(v, torch.Tensor) and len(v) == len(boxes):  
                    result.extra_fields[k] = v[mask]  
          
        return result

@register()  
class RandomPatchGaussian(T.Transform):  
    _transformed_types = (PIL.Image.Image, Image)  
      
    def __init__(self, p=0.5, num_patches=3, patch_size=(0.1, 0.3), sigma=(0.1, 0.2)) -> None:  
        """  
        Apply random Gaussian noise patches to images.  
          
        Args:  
            p (float): Probability of applying the transform  
            num_patches (int or tuple): Number of patches to apply (if tuple, range of values)  
            patch_size (tuple): Range of patch sizes as fraction of image size (min, max)  
            sigma (tuple): Range of standard deviations for Gaussian noise (min, max)  
        """  
        super().__init__()  
        self.p = p  
        self.num_patches = num_patches if isinstance(num_patches, tuple) else (num_patches, num_patches)  
        self.patch_size = patch_size  
        self.sigma = sigma  
      
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:  
        # Determine whether to apply the transform based on probability  
        apply = torch.rand(1) < self.p  
        if not apply:  
            return {"apply": False}  
          
        # Get image size  
        img = flat_inputs[0]  
        if isinstance(img, PIL.Image.Image):  
            width, height = img.size  
        else:  # Image tensor  
            _, height, width = img.shape  
          
        # Determine number of patches  
        num_patches = torch.randint(self.num_patches[0], self.num_patches[1] + 1, (1,)).item()  
          
        # Generate patch parameters  
        patches = []  
        for _ in range(num_patches):  
            # Patch size as fraction of image size  
            patch_w_frac = torch.FloatTensor(1).uniform_(*self.patch_size).item()  
            patch_h_frac = torch.FloatTensor(1).uniform_(*self.patch_size).item()  
              
            # Convert to pixel dimensions  
            patch_w = int(width * patch_w_frac)  
            patch_h = int(height * patch_h_frac)  
              
            # Patch position  
            x = torch.randint(0, width - patch_w + 1, (1,)).item()  
            y = torch.randint(0, height - patch_h + 1, (1,)).item()  
              
            # Noise sigma  
            sigma = torch.FloatTensor(1).uniform_(*self.sigma).item()  
              
            patches.append((x, y, patch_w, patch_h, sigma))  
          
        return {"apply": True, "patches": patches}  
      
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if not params["apply"]:  
            return inpt  
          
        if isinstance(inpt, PIL.Image.Image):  
            # Convert PIL image to tensor for processing  
            tensor = F.pil_to_tensor(inpt).float() / 255.0  
            tensor = self._apply_patches(tensor, params["patches"])  
            # Convert back to PIL  
            return F.to_pil_image((tensor * 255.0).byte())  
        elif isinstance(inpt, Image):  
            # Process tensor directly  
            tensor = inpt.as_tensor()  
            tensor = self._apply_patches(tensor, params["patches"])  
            return Image(tensor)  
          
        return inpt  
      
    def _apply_patches(self, tensor, patches):  
        # Apply Gaussian noise patches to tensor  
        for x, y, w, h, sigma in patches:  
            noise = torch.randn(tensor.shape[0], h, w) * sigma  
            tensor[:, y:y+h, x:x+w] += noise  
              
            # Clamp values to valid range [0, 1]  
            tensor = torch.clamp(tensor, 0, 1)  
          
        return tensor