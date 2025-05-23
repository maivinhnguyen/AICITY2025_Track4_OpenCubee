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
class CopyPaste(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        BoundingBoxes,
    )

    def __init__(
        self,
        p: float = 0.5,
        blend: bool = True,
        sigma: float = 1.0,
        min_area: float = 0.0,
        category_ids: Optional[List[int]] = None,
        small_object_threshold: float = 0.05,
        max_paste_objects: int = 3,
        paste_all_matching: bool = True,
    ) -> None:
        """
        CopyPaste augmentation for object detection.
        
        Args:
            p: Probability of applying the transform
            blend: Whether to use Gaussian blending for smooth pasting
            sigma: Standard deviation for Gaussian blending
            min_area: Minimum normalized area threshold for objects to be copied
            category_ids: List of category IDs to prioritize for copying
            small_object_threshold: Normalized area threshold to consider objects as "small"
            max_paste_objects: Maximum number of objects to paste (ignored if paste_all_matching=True)
            paste_all_matching: If True, paste all objects matching the criteria
        """
        super().__init__()
        self.p = p
        self.blend = blend
        self.sigma = sigma
        self.min_area = min_area
        self.category_ids = set(category_ids or [])
        self.small_object_threshold = small_object_threshold
        self.max_paste_objects = max_paste_objects
        self.paste_all_matching = paste_all_matching

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        apply = torch.rand(1) < self.p
        
        # Extract image, boxes, and labels from inputs
        image = None
        boxes = None
        labels = None
        
        for inp in flat_inputs:
            if isinstance(inp, (PIL.Image.Image, Image)):
                image = inp
            elif isinstance(inp, BoundingBoxes):
                boxes = inp
                # Labels might be stored as an attribute
                labels = getattr(inp, 'labels', None)
            elif isinstance(inp, dict):
                # Handle dict format
                if 'boxes' in inp:
                    boxes = inp['boxes']
                if 'labels' in inp:
                    labels = inp['labels']
                if 'image' in inp:
                    image = inp['image']
        
        return {
            "apply": apply,
            "image": image,
            "boxes": boxes,
            "labels": labels,
        }

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel using PyTorch"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        
        return gaussian

    def _apply_gaussian_blur(self, mask: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to mask using PyTorch operations"""
        if sigma <= 0:
            return mask
        
        # Ensure mask is a tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        
        # Add batch and channel dimensions if needed
        original_shape = mask.shape
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Create Gaussian kernel
        kernel_size = int(2 * math.ceil(2 * sigma) + 1)
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0).to(mask.device)  # [1, 1, K, K]
        
        # Apply convolution for blurring
        padding = kernel_size // 2
        blurred = NF.conv2d(mask, kernel, padding=padding)
        
        # Restore original shape
        if len(original_shape) == 2:
            blurred = blurred.squeeze(0).squeeze(0)
        
        return blurred

    def _select_objects_to_copy(self, image_size: Tuple[int, int], boxes: torch.Tensor, 
                               labels: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select objects based on category ID and small object criteria"""
        if boxes is None or len(boxes) == 0:
            return torch.empty((0, 4)), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        # Ensure tensor format
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes, dtype=torch.float32)
        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        
        h, w = image_size
        
        # Calculate normalized areas
        box_areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / (h * w)
        
        # Create selection mask based on criteria
        selection_mask = torch.ones(len(boxes), dtype=torch.bool)
        
        # Filter by minimum area
        selection_mask &= (box_areas >= self.min_area)
        
        # Filter by small object threshold
        if self.small_object_threshold > 0:
            selection_mask &= (box_areas < self.small_object_threshold)
        
        # Filter by category IDs
        if len(self.category_ids) > 0 and labels is not None:
            category_mask = torch.zeros(len(labels), dtype=torch.bool)
            for cat_id in self.category_ids:
                category_mask |= (labels == cat_id)
            selection_mask &= category_mask
        
        # Get valid objects
        if not selection_mask.any():
            return torch.empty((0, 4)), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        valid_boxes = boxes[selection_mask]
        valid_labels = labels[selection_mask] if labels is not None else torch.arange(selection_mask.sum())
        valid_indices = torch.where(selection_mask)[0]
        
        # Select objects to copy
        if self.paste_all_matching:
            # Copy all matching objects
            return valid_boxes, valid_labels, valid_indices
        else:
            # Copy up to max_paste_objects
            num_to_copy = min(len(valid_boxes), self.max_paste_objects)
            if num_to_copy > 0:
                selected_idx = torch.randperm(len(valid_boxes))[:num_to_copy]
                return valid_boxes[selected_idx], valid_labels[selected_idx], valid_indices[selected_idx]
        
        return torch.empty((0, 4)), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    def _extract_object_patches(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract object patches and masks from image"""
        patches = []
        masks = []
        
        # Ensure image is in correct format [C, H, W]
        if len(image.shape) == 4:
            image = image.squeeze(0)
        
        for box in boxes:
            x1, y1, x2, y2 = box.int()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[-1], x2), min(image.shape[-2], y2)
            
            if x2 > x1 and y2 > y1:
                patch = image[..., y1:y2, x1:x2].clone()
                mask = torch.ones((y2-y1, x2-x1), dtype=torch.float32, device=image.device)
                
                patches.append(patch)
                masks.append(mask)
        
        return patches, masks

    def _generate_paste_positions(self, image_shape: torch.Size, patches: List[torch.Tensor]) -> List[Tuple[int, int]]:
        """Generate valid paste positions avoiding overlaps with original objects"""
        h, w = image_shape[-2:]
        positions = []
        
        for patch in patches:
            patch_h, patch_w = patch.shape[-2:]
            
            # Calculate valid paste region
            max_x = max(1, w - patch_w)
            max_y = max(1, h - patch_h)
            
            if max_x > 0 and max_y > 0:
                # Generate random position
                paste_x = random.randint(0, max_x - 1)
                paste_y = random.randint(0, max_y - 1)
                positions.append((paste_x, paste_y))
            else:
                positions.append((0, 0))  # Fallback position
        
        return positions

    def _paste_objects(self, target_image: torch.Tensor, patches: List[torch.Tensor], 
                      masks: List[torch.Tensor], positions: List[Tuple[int, int]]) -> torch.Tensor:
        """Paste objects onto target image with optional blending"""
        if len(patches) == 0:
            return target_image
        
        result_image = target_image.clone()
        
        for patch, mask, (paste_x, paste_y) in zip(patches, masks, positions):
            patch_h, patch_w = patch.shape[-2:]
            
            # Ensure paste position is within bounds
            if paste_x + patch_w > target_image.shape[-1] or paste_y + patch_h > target_image.shape[-2]:
                continue
            
            if self.blend and self.sigma > 0:
                # Apply Gaussian blending
                blurred_mask = self._apply_gaussian_blur(mask, self.sigma)
                blurred_mask = blurred_mask.to(target_image.device)
                
                # Expand mask dimensions to match patch
                if len(patch.shape) == 3 and len(blurred_mask.shape) == 2:
                    blurred_mask = blurred_mask.unsqueeze(0).expand_as(patch)
                
                # Blend the patch
                target_region = result_image[..., paste_y:paste_y+patch_h, paste_x:paste_x+patch_w]
                blended = patch * blurred_mask + target_region * (1 - blurred_mask)
                result_image[..., paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = blended
            else:
                # Direct paste
                result_image[..., paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = patch
        
        return result_image

    def _transform_image(self, image: Any, params: Dict[str, Any]) -> Any:
        """Transform image by copying and pasting objects"""
        if not params["apply"] or params["boxes"] is None or len(params["boxes"]) == 0:
            return image
        
        # Convert image to tensor if needed
        if isinstance(image, PIL.Image.Image):
            image_tensor = F.pil_to_tensor(image).float() / 255.0
        elif isinstance(image, Image):
            image_tensor = image.as_subclass(torch.Tensor)
        else:
            image_tensor = image
        
        # Get image dimensions
        if len(image_tensor.shape) >= 2:
            image_size = image_tensor.shape[-2:]
        else:
            return image
        
        # Select objects to copy
        selected_boxes, selected_labels, _ = self._select_objects_to_copy(
            image_size, params["boxes"], params["labels"]
        )
        
        if len(selected_boxes) == 0:
            return image
        
        # Extract patches from selected objects
        patches, masks = self._extract_object_patches(image_tensor, selected_boxes)
        
        if len(patches) == 0:
            return image
        
        # Generate paste positions
        paste_positions = self._generate_paste_positions(image_tensor.shape, patches)
        
        # Paste objects at random locations
        result_tensor = self._paste_objects(image_tensor, patches, masks, paste_positions)
        
        # Convert back to original format
        if isinstance(image, PIL.Image.Image):
            result_tensor = (result_tensor * 255).clamp(0, 255).byte()
            return F.to_pil_image(result_tensor)
        elif isinstance(image, Image):
            return Image(result_tensor)
        else:
            return result_tensor

    def _transform_boxes(self, boxes: BoundingBoxes, params: Dict[str, Any]) -> BoundingBoxes:
        """Transform bounding boxes by adding pasted object boxes"""
        if not params["apply"] or params["labels"] is None or len(params["boxes"]) == 0:
            return boxes
        
        # Get spatial size
        spatial_size = getattr(boxes, _boxes_keys[1])
        h, w = spatial_size
        
        # Select objects to copy
        selected_boxes, selected_labels, _ = self._select_objects_to_copy(
            (h, w), boxes, params["labels"]
        )
        
        if len(selected_boxes) == 0:
            return boxes
        
        # Generate new box positions for pasted objects
        new_boxes = []
        new_labels = []
        
        for box, label in zip(selected_boxes, selected_labels):
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            
            # Generate random position
            max_x = max(1, w - box_w)
            max_y = max(1, h - box_h)
            
            if max_x > 0 and max_y > 0:
                new_x = random.randint(0, int(max_x) - 1)
                new_y = random.randint(0, int(max_y) - 1)
                
                new_box = torch.tensor([new_x, new_y, new_x + box_w, new_y + box_h], 
                                     dtype=box.dtype, device=box.device)
                new_boxes.append(new_box)
                new_labels.append(label)
        
        if len(new_boxes) > 0:
            # Combine original and new boxes
            all_boxes = torch.cat([boxes, torch.stack(new_boxes)])
            all_labels = torch.cat([params["labels"], torch.stack(new_labels)])
            
            # Create new BoundingBoxes object
            result_boxes = BoundingBoxes(all_boxes, format=boxes.format, canvas_size=spatial_size)
            result_boxes.labels = all_labels
            
            return result_boxes
        
        return boxes

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """Transform individual input based on its type"""
        if isinstance(inpt, (PIL.Image.Image, Image)):
            return self._transform_image(inpt, params)
        elif isinstance(inpt, BoundingBoxes):
            return self._transform_boxes(inpt, params)
        else:
            return inpt

    def forward(self, *inputs: Any) -> Any:
        """Main forward pass following DFINE's transform pattern"""
        # Flatten inputs
        flat_inputs = []
        for inp in inputs:
            if isinstance(inp, (list, tuple)):
                flat_inputs.extend(inp)
            else:
                flat_inputs.append(inp)
        
        # Get parameters
        params = self._get_params(flat_inputs)
        
        # If not applying, return inputs unchanged
        if not params["apply"]:
            return inputs if len(inputs) > 1 else inputs[0]
        
        # Apply transform to each input
        transformed = []
        for inp in inputs:
            if isinstance(inp, (list, tuple)):
                transformed.append(type(inp)(self._transform(item, params) for item in inp))
            else:
                transformed.append(self._transform(inp, params))
        
        return transformed if len(transformed) > 1 else transformed[0]