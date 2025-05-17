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
class RandomFisheyeShiftCrop:
    def __init__(self, max_shift=0.25, crop_ratio=0.85, p=0.7):
        self.max_shift = max_shift
        self.crop_ratio = crop_ratio
        self.p = p

    def __call__(self, inputs):
        image, target = inputs

        if torch.rand(1).item() > self.p:
            return image, target

        # Get size from PIL image
        if hasattr(image, 'size'):
            w, h = image.size
        else:
            # fallback for tensor (C,H,W)
            _, h, w = image.shape

        shift_x = int(torch.rand(1).item() * self.max_shift * w * (1 if torch.rand(1).item() > 0.5 else -1))
        shift_y = int(torch.rand(1).item() * self.max_shift * h * (1 if torch.rand(1).item() > 0.5 else -1))

        new_w = int(w * self.crop_ratio)
        new_h = int(h * self.crop_ratio)

        left = min(max(0, (w - new_w) // 2 + shift_x), w - new_w)
        top = min(max(0, (h - new_h) // 2 + shift_y), h - new_h)

        # Crop PIL Image
        image = image.crop((left, top, left + new_w, top + new_h))

        # Convert cropped PIL image to tensor normalized [0,1]
        image = F.pil_to_tensor(image).to(torch.float32) / 255.0

        boxes = target["boxes"]

        # Convert normalized boxes to absolute if needed
        if boxes.max() <= 1.0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h

        # Adjust boxes for crop
        boxes = boxes - torch.tensor([left, top, left, top], device=boxes.device, dtype=boxes.dtype)

        # Clamp boxes inside crop bounds
        boxes[:, 0].clamp_(min=0, max=new_w)
        boxes[:, 1].clamp_(min=0, max=new_h)
        boxes[:, 2].clamp_(min=0, max=new_w)
        boxes[:, 3].clamp_(min=0, max=new_h)

        # Remove invalid boxes
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        # Filter other target tensors accordingly
        for k, v in target.items():
            if isinstance(v, torch.Tensor) and len(v) == len(keep):
                target[k] = v[keep]

        # Normalize boxes back if needed
        if boxes.max() > 1.0:
            boxes[:, [0, 2]] /= new_w
            boxes[:, [1, 3]] /= new_h

        target["boxes"] = boxes

        return image, target


@register()
class FisheyeEdgeStretchCrop:
    def __init__(self, stretch_prob=0.5, scale_y=0.9):
        self.stretch_prob = stretch_prob
        self.scale_y = scale_y

    def __call__(self, inputs):
        image, target = inputs

        if torch.rand(1).item() > self.stretch_prob:
            # Convert to tensor if PIL
            if hasattr(image, 'size'):
                image = F.pil_to_tensor(image).to(torch.float32) / 255.0
            return image, target

        # Get size from PIL image or tensor
        if hasattr(image, 'size'):
            w, h = image.size
            # Convert PIL to tensor for resizing operations
            image = F.pil_to_tensor(image).to(torch.float32) / 255.0
        else:
            _, h, w = image.shape

        mid_y = h // 2

        # Resize top and bottom halves separately
        top = F.resize(image[:, :mid_y, :], [int(mid_y * self.scale_y), w])
        bottom = F.resize(image[:, mid_y:, :], [int(mid_y * self.scale_y), w])
        image = torch.cat([top, bottom], dim=1)
        image = F.resize(image, [h, w])

        boxes = target["boxes"]

        # Convert normalized boxes to absolute if needed
        if boxes.max() <= 1.0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h

        new_boxes = boxes.clone()

        # Masks for boxes in top, bottom, or crossing middle
        top_mask = boxes[:, 3] <= mid_y
        bottom_mask = boxes[:, 1] >= mid_y
        cross_mask = ~(top_mask | bottom_mask)

        # Scale top half boxes y coords
        new_boxes[top_mask, 1] = boxes[top_mask, 1] * self.scale_y
        new_boxes[top_mask, 3] = boxes[top_mask, 3] * self.scale_y

        # Scale bottom half boxes y coords (relative to mid_y)
        new_boxes[bottom_mask, 1] = (boxes[bottom_mask, 1] - mid_y) * self.scale_y + int(mid_y * self.scale_y)
        new_boxes[bottom_mask, 3] = (boxes[bottom_mask, 3] - mid_y) * self.scale_y + int(mid_y * self.scale_y)

        # Leave cross_mask boxes unchanged for simplicity

        # After vertical scaling, image resized back to original height
        scale_back = h / (2 * int(mid_y * self.scale_y))
        new_boxes[:, 1] = new_boxes[:, 1] * scale_back
        new_boxes[:, 3] = new_boxes[:, 3] * scale_back

        # Clamp boxes to image bounds
        new_boxes[:, 0].clamp_(min=0, max=w)
        new_boxes[:, 1].clamp_(min=0, max=h)
        new_boxes[:, 2].clamp_(min=0, max=w)
        new_boxes[:, 3].clamp_(min=0, max=h)

        # Remove invalid boxes
        keep = (new_boxes[:, 2] > new_boxes[:, 0]) & (new_boxes[:, 3] > new_boxes[:, 1])
        new_boxes = new_boxes[keep]

        # Filter other target tensors accordingly
        for k, v in target.items():
            if isinstance(v, torch.Tensor) and len(v) == len(keep):
                target[k] = v[keep]

        # Normalize boxes back if needed
        if new_boxes.max() > 1.0:
            new_boxes[:, [0, 2]] /= w
            new_boxes[:, [1, 3]] /= h

        target["boxes"] = new_boxes

        return image, target