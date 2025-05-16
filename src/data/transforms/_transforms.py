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
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.k = None

    def _get_params(self, flat_inputs):
        if torch.rand(1) < self.p:
            self.k = torch.randint(1, 4, (1,)).item()  # 90, 180, or 270 degrees
        else:
            self.k = 0
        return {"k": self.k}

    def _transform(self, inpt, params):
        k = params["k"]
        if k == 0:
            return inpt

        return F.rotate(inpt, angle=90 * k, expand=False)


@register()
class RandomPatchGaussian(T.Transform):
    def __init__(self, p=0.3, patch_size=64, sigma=0.2):
        super().__init__()
        self.p = p
        self.patch_size = patch_size
        self.sigma = sigma

    def _transform(self, inpt, params):
        if not isinstance(inpt, Image) or torch.rand(1) >= self.p:
            return inpt

        img = inpt.clone()
        _, h, w = img.shape
        ph, pw = min(self.patch_size, h), min(self.patch_size, w)
        top = torch.randint(0, h - ph + 1, (1,)).item()
        left = torch.randint(0, w - pw + 1, (1,)).item()

        noise = torch.randn_like(img[:, top:top+ph, left:left+pw]) * self.sigma
        img[:, top:top+ph, left:left+pw] += noise
        img = img.clamp(0.0, 1.0)
        return Image(img)

@register()
class FilterSmallInstances(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, min_pixels=9, min_visibility=0.2):
        super().__init__()
        self.min_pixels = min_pixels
        self.min_visibility = min_visibility  # Reserved for masks if needed

    def _transform(self, inpt, params):
        boxes = inpt
        spatial_size = getattr(inpt, _boxes_keys[1])  # (H, W)

        if boxes.format == torchvision.tv_tensors.BoundingBoxFormat.XYXY:
            boxes = torchvision.ops.box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')

        if getattr(inpt, "normalized", False):
            h, w = spatial_size
            boxes = boxes.clone()
            boxes[:, 2] *= w
            boxes[:, 3] *= h

        area = boxes[:, 2] * boxes[:, 3]
        keep = area >= self.min_pixels
        return inpt[keep]

