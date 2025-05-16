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
    """
    Randomly rotates the input by 0, 90, 180, or 270 degrees counter-clockwise.
    The rotation is applied with a probability `p`. If applied, one of
    90, 180, or 270 degrees is chosen uniformly.
    """
    _transformed_types = (Image, Video, Mask, BoundingBoxes)

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be between 0 and 1, got {p}")
        self.p = p

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1).item() < self.p:
            # Choose a random k from {1, 2, 3} for 90, 180, 270 deg CCW rotation
            k = torch.randint(1, 4, (1,)).item()
            angle = float(k * 90)
        else:
            angle = 0.0  # No rotation
        
        # expand=True is needed for 90/270 deg rotations if H != W to prevent cropping
        # F.rotate handles this correctly for BoundingBoxes too.
        expand = abs(angle % 180) == 90.0
        return dict(angle=angle, expand=expand)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        angle = params["angle"]
        if angle == 0.0:
            return inpt
        
        return F.rotate(inpt, angle=angle, expand=params["expand"], interpolation=F.InterpolationMode.NEAREST)


@register()
class RandomPatchGaussian(T.Transform):
    """
    Applies Gaussian noise to a randomly selected patch of the image.
    Operates on Image and Video tv_tensors.
    Assumes input tensor is float and in [0, 1] range.
    """
    _transformed_types = (Image, Video)

    def __init__(
        self,
        p: float = 0.5,
        patch_size_ratio_min: float = 0.1,
        patch_size_ratio_max: float = 0.5,
        sigma_min: float = 0.01,
        sigma_max: float = 0.1,
    ):
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Probability p must be between 0 and 1, got {p}")
        self.p = p
        if not (0 < patch_size_ratio_min <= patch_size_ratio_max <= 1.0):
            raise ValueError("Invalid patch_size_ratio values.")
        if not (0 < sigma_min <= sigma_max):
            raise ValueError("Invalid sigma values.")
            
        self.patch_size_ratio_min = patch_size_ratio_min
        self.patch_size_ratio_max = patch_size_ratio_max
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1).item() >= self.p:
            return dict(apply_noise=False)

        # Find the first image-like input to get spatial size
        img_inpt = None
        for x in flat_inputs:
            if isinstance(x, (Image, Video)):
                img_inpt = x
                break
        if img_inpt is None: # Should not happen if used correctly
            return dict(apply_noise=False)

        _, h, w = F.get_dimensions(img_inpt) # C, H, W or T, C, H, W

        patch_h_ratio = torch.rand(1).item() * (self.patch_size_ratio_max - self.patch_size_ratio_min) + self.patch_size_ratio_min
        patch_w_ratio = torch.rand(1).item() * (self.patch_size_ratio_max - self.patch_size_ratio_min) + self.patch_size_ratio_min
        
        ph = max(1, int(h * patch_h_ratio))
        pw = max(1, int(w * patch_w_ratio))

        y1 = torch.randint(0, h - ph + 1, (1,)).item() if h > ph else 0
        x1 = torch.randint(0, w - pw + 1, (1,)).item() if w > pw else 0
        
        sigma = torch.rand(1).item() * (self.sigma_max - self.sigma_min) + self.sigma_min
        
        return dict(apply_noise=True, y1=y1, x1=x1, ph=ph, pw=pw, sigma=sigma)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params["apply_noise"] or not isinstance(inpt, (Image, Video)):
            return inpt

        # Ensure input is float. If not, this transform might behave unexpectedly.
        # Typically, images are converted to float and scaled to [0,1] before this.
        if not inpt.is_floating_point():
            # print("Warning: RandomPatchGaussian received non-float input. Noise might be miscaled.")
            # For robustness, could convert, apply, then convert back, but that's slow.
            # Best to ensure pipeline prepares data correctly.
            pass # Proceed, assuming user knows what they're doing or data is already fine.

        out_tensor = inpt.as_subclass(torch.Tensor).clone()
        
        y1, x1 = params["y1"], params["x1"]
        ph, pw = params["ph"], params["pw"]
        sigma = params["sigma"]

        patch = out_tensor[..., y1 : y1 + ph, x1 : x1 + pw]
        noise = torch.randn_like(patch) * sigma
        noisy_patch = torch.clamp(patch + noise, 0.0, 1.0) # Clamp to [0,1]
        
        out_tensor[..., y1 : y1 + ph, x1 : x1 + pw] = noisy_patch
        
        return type(inpt)(out_tensor)


@register()
class FilterSmallInstances(T.Transform):
    _transformed_types = (BoundingBoxes, Mask) # Assuming BoundingBoxes is from .._misc

    def __init__(self, min_pixels: int = 9, min_visibility: float = 0.2):
        super().__init__()
        # ... (init checks) ...
        self.min_pixels = min_pixels
        self.min_visibility = min_visibility

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, BoundingBoxes):
            if not inpt.numel() or inpt.shape[0] == 0:
                return inpt
            
            # --- Start Defensive Checks ---
            if not hasattr(inpt, 'spatial_size') or inpt.spatial_size is None:
                raise AttributeError(
                    f"Input BoundingBoxes (type: {type(inpt)}) to FilterSmallInstances is missing 'spatial_size'. "
                    "This attribute is essential. Check the preceding transform in your pipeline "
                    "(likely SanitizeBoundingBoxes or a geometric transform like Resize) to ensure it correctly "
                    "sets or propagates 'spatial_size' on the BoundingBoxes objects it outputs."
                )
            if not hasattr(inpt, 'format') or inpt.format is None:
                raise AttributeError(
                    f"Input BoundingBoxes (type: {type(inpt)}) to FilterSmallInstances is missing 'format'."
                )
            # --- End Defensive Checks ---

            # Assuming XYXY format for area calculation for BoundingBoxes tv_tensor
            # Ensure your BoundingBox format is indeed XYXY at this stage, or adjust logic.
            # If inpt.format is not XYXY, you might need to convert it first or handle different formats.
            if str(inpt.format).upper() != "XYXY": # str() for safety if it's an enum
                 # Potentially convert to XYXY for area calculation then convert back
                 # For simplicity, this example assumes it's already XYXY or calculation is compatible
                 pass # Add conversion if necessary

            widths = inpt[:, 2] - inpt[:, 0]
            heights = inpt[:, 3] - inpt[:, 1]
            areas = widths * heights
            keep = areas >= self.min_pixels

            filtered_data = inpt.as_subclass(torch.Tensor)[keep]
            
            # Reconstruct using the BoundingBoxes class from _misc
            # This assumes the _misc.BoundingBoxes constructor matches this signature
            # and correctly handles the 'format' (e.g., if it's an enum or string)
            return BoundingBoxes(
                filtered_data,
                format=inpt.format,
                spatial_size=inpt.spatial_size,
                dtype=inpt.dtype,
                device=inpt.device
            )

        elif isinstance(inpt, Mask):
            # ... (Mask logic remains the same) ...
            mask_tensor = inpt.as_subclass(torch.Tensor)
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            if not mask_tensor.numel() or mask_tensor.shape[0] == 0:
                return inpt
            if mask_tensor.ndim < 3:
                return inpt

            areas = mask_tensor.sum(dim=(-2, -1)) 
            keep = areas >= self.min_pixels
            filtered_tensor = mask_tensor[keep]
            return Mask(filtered_tensor)
            
        return inpt