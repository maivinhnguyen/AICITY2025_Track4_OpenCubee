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
    _transformed_types = (PIL.Image.Image,)
    def __init__(self, p=0.3, source_json=None, source_img_dir=None):
        super().__init__()
        self.p = p
        self.source_img_dir = source_img_dir

        with open(source_json, 'r') as f:
            data = json.load(f)

        self.img_id_to_file = {img['id']: img['file_name'] for img in data['images']}
        self.img_id_to_anns = {}
        for ann in data['annotations']:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)
        self.image_ids = list(self.img_id_to_file.keys())

    def transform(self, inpt: PIL.Image.Image, params: dict) -> PIL.Image.Image:
        return inpt  # No changes here — everything is done in __call__

    def transform_target(self, target: dict, params: dict) -> dict:
        return target

    def _transform(self, inpt, params):
        return inpt

    def __call__(self, inpt: PIL.Image.Image, target: dict):
        if random.random() > self.p:
            return inpt, target

        donor_img_id = random.choice(self.image_ids)
        donor_file = self.img_id_to_file[donor_img_id]
        donor_path = os.path.join(self.source_img_dir, donor_file)

        try:
            donor_image = PIL.Image.open(donor_path).convert("RGB")
        except Exception:
            return inpt, target

        donor_anns = self.img_id_to_anns[donor_img_id]

        for ann in donor_anns:
            x, y, w, h = map(int, ann['bbox'])  # COCO: [x, y, w, h]
            if w < 8 or h < 8 or x < 0 or y < 0 or x + w > donor_image.width or y + h > donor_image.height:
                continue

            obj_crop = donor_image.crop((x, y, x + w, y + h))
            paste_x = random.randint(0, max(0, inpt.width - w))
            paste_y = random.randint(0, max(0, inpt.height - h))

            inpt.paste(obj_crop, (paste_x, paste_y))

            new_box = [paste_x, paste_y, w, h]
            if isinstance(target.get("boxes"), torch.Tensor):
                target["boxes"] = torch.cat([target["boxes"], torch.tensor([new_box], dtype=torch.float32)])
            else:
                target["boxes"] = torch.tensor([new_box], dtype=torch.float32)

            if isinstance(target.get("labels"), torch.Tensor):
                target["labels"] = torch.cat([target["labels"], torch.tensor([ann["category_id"]], dtype=torch.int64)])
            else:
                target["labels"] = torch.tensor([ann["category_id"]], dtype=torch.int64)

        return inpt, target
