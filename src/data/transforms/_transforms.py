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
    _transformed_types = (PIL.Image.Image, BoundingBoxes, Image)  
      
    def __init__(self, p=0.5, blend=True, sigma=1.0, min_area=0.0,   
                 rare_class_ids=None, small_object_threshold=0.05) -> None:  
        super().__init__()  
        self.p = p  
        self.blend = blend  
        self.sigma = sigma  
        self.min_area = min_area  
        self.rare_class_ids = set(rare_class_ids or [])  
        self.small_object_threshold = small_object_threshold  
        self._source_cache = []  # Cache for source objects  
      
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:  
        apply = torch.rand(1) < self.p  
          
        # Extract labels and boxes  
        labels = None  
        boxes = None  
        for inp in flat_inputs:  
            if isinstance(inp, BoundingBoxes):  
                boxes = inp  
                labels = getattr(inp, 'labels', None)  
                break  
          
        return {  
            "apply": apply,  
            "labels": labels,  
            "boxes": boxes,  
            "prioritize_rare_classes": len(self.rare_class_ids) > 0,  
            "prioritize_small_objects": self.small_object_threshold > 0  
        }  
      
    def _select_objects_to_copy(self, image, boxes, labels, params):  
        """Select objects based on rare class and small object criteria"""  
        if boxes is None or len(boxes) == 0:  
            return [], [], []  
          
        # Calculate normalized areas  
        h, w = image.shape[-2:]  
        box_areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / (h * w)  
          
        # Create selection weights  
        weights = torch.ones(len(boxes))  
          
        if params["prioritize_rare_classes"] and labels is not None:  
            for i, label in enumerate(labels):  
                if label.item() in self.rare_class_ids:  
                    weights[i] *= 3.0  # Boost rare classes  
          
        if params["prioritize_small_objects"]:  
            small_mask = box_areas < self.small_object_threshold  
            weights[small_mask] *= 2.0  # Boost small objects  
          
        # Filter by minimum area  
        valid_mask = box_areas >= self.min_area  
        if not valid_mask.any():  
            return [], [], []  
          
        valid_boxes = boxes[valid_mask]  
        valid_labels = labels[valid_mask] if labels is not None else None  
        valid_weights = weights[valid_mask]  
          
        # Select objects to copy  
        num_to_copy = min(len(valid_boxes), torch.randint(1, len(valid_boxes) + 1, (1,)).item())  
        if len(valid_weights) > 0:  
            selected_indices = torch.multinomial(valid_weights, num_to_copy, replacement=False)  
            return valid_boxes[selected_indices], valid_labels[selected_indices] if valid_labels is not None else None, selected_indices  
          
        return [], [], []  
      
    def _extract_object_patches(self, image, boxes):  
        """Extract object patches from image"""  
        patches = []  
        masks = []  
          
        for box in boxes:  
            x1, y1, x2, y2 = box.int()  
            x1, y1 = max(0, x1), max(0, y1)  
            x2, y2 = min(image.shape[-1], x2), min(image.shape[-2], y2)  
              
            if x2 > x1 and y2 > y1:  
                patch = image[..., y1:y2, x1:x2].clone()  
                  
                # Create mask for the object  
                mask = torch.ones((y2-y1, x2-x1), dtype=torch.float32)  
                  
                patches.append(patch)  
                masks.append(mask)  
          
        return patches, masks  
      
    def _paste_objects(self, target_image, patches, masks, target_boxes):  
        """Paste objects onto target image"""  
        if len(patches) == 0:  
            return target_image  
          
        result_image = target_image.clone()  
        h, w = target_image.shape[-2:]  
          
        for patch, mask, box in zip(patches, masks, target_boxes):  
            # Generate random position for pasting  
            patch_h, patch_w = patch.shape[-2:]  
              
            # Ensure paste position is within image bounds  
            max_x = max(0, w - patch_w)  
            max_y = max(0, h - patch_h)  
              
            if max_x <= 0 or max_y <= 0:  
                continue  
                  
            paste_x = torch.randint(0, max_x + 1, (1,)).item()  
            paste_y = torch.randint(0, max_y + 1, (1,)).item()  
              
            # Paste the object  
            if self.blend:  
                # Apply Gaussian blending  
                from scipy.ndimage import gaussian_filter  
                import numpy as np  
                  
                mask_np = mask.numpy()  
                blurred_mask = torch.from_numpy(gaussian_filter(mask_np, sigma=self.sigma))  
                blurred_mask = blurred_mask.unsqueeze(0) if len(patch.shape) == 3 else blurred_mask  
                  
                # Blend the patch  
                target_region = result_image[..., paste_y:paste_y+patch_h, paste_x:paste_x+patch_w]  
                blended = patch * blurred_mask + target_region * (1 - blurred_mask)  
                result_image[..., paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = blended  
            else:  
                # Direct paste  
                result_image[..., paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = patch  
          
        return result_image  
      
    def _transform_image(self, image, params):  
        """Transform image by copying and pasting objects"""  
        if not params["apply"] or params["boxes"] is None:  
            return image  
          
        # Select objects to copy  
        selected_boxes, selected_labels, _ = self._select_objects_to_copy(  
            image, params["boxes"], params["labels"], params  
        )  
          
        if len(selected_boxes) == 0:  
            return image  
          
        # Extract patches from selected objects  
        patches, masks = self._extract_object_patches(image, selected_boxes)  
          
        # Paste objects at random locations  
        result_image = self._paste_objects(image, patches, masks, selected_boxes)  
          
        return result_image  
      
    def _transform_boxes(self, boxes, params):  
        """Transform bounding boxes by adding pasted object boxes"""  
        if not params["apply"] or params["labels"] is None:  
            return boxes  
          
        # Select objects to copy  
        selected_boxes, selected_labels, _ = self._select_objects_to_copy(  
            None, boxes, params["labels"], params  
        )  
          
        if len(selected_boxes) == 0:  
            return boxes  
          
        # Generate new box positions for pasted objects  
        h, w = boxes.canvas_size  
        new_boxes = []  
        new_labels = []  
          
        for box, label in zip(selected_boxes, selected_labels):  
            box_w = box[2] - box[0]  
            box_h = box[3] - box[1]  
              
            # Random position within image bounds  
            max_x = max(0, w - box_w)  
            max_y = max(0, h - box_h)  
              
            if max_x > 0 and max_y > 0:  
                new_x = torch.randint(0, int(max_x) + 1, (1,)).item()  
                new_y = torch.randint(0, int(max_y) + 1, (1,)).item()  
                  
                new_box = torch.tensor([new_x, new_y, new_x + box_w, new_y + box_h])  
                new_boxes.append(new_box)  
                new_labels.append(label)  
          
        if len(new_boxes) > 0:  
            # Combine original and new boxes  
            all_boxes = torch.cat([boxes, torch.stack(new_boxes)])  
            all_labels = torch.cat([params["labels"], torch.stack(new_labels)])  
              
            # Create new BoundingBoxes object  
            from torchvision.tv_tensors import BoundingBoxes  
            result_boxes = BoundingBoxes(all_boxes, format=boxes.format, canvas_size=boxes.canvas_size)  
            result_boxes.labels = all_labels  
              
            return result_boxes  
          
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
class RandomZoomIn(T.Transform):    
    _transformed_types = (    
        PIL.Image.Image,    
        Image,    
        Video,    
        Mask,    
        BoundingBoxes,    
    )    
    
    def __init__(    
        self,    
        min_scale: float = 1.1,    
        max_scale: float = 1.5,    
        min_size_threshold: int = 32,    
        target_size_threshold: int = 64,    
        p: float = 0.5,    
    ) -> None:    
        """    
        RandomZoomIn transform to enhance detection of small objects.    
            
        Args:    
            min_scale: Minimum zoom scale factor    
            max_scale: Maximum zoom scale factor    
            min_size_threshold: Objects smaller than this will trigger zoom-in    
            target_size_threshold: Target size to scale small objects to    
            p: Probability of applying this transform    
        """    
        super().__init__()    
        self.min_scale = min_scale    
        self.max_scale = max_scale    
        self.min_size_threshold = min_size_threshold    
        self.target_size_threshold = target_size_threshold    
        self.p = p    
    
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:    
        # Find image and boxes in inputs    
        image = None    
        boxes = None    
            
        for x in flat_inputs:    
            if isinstance(x, (PIL.Image.Image, Image)):    
                image = x    
            elif isinstance(x, BoundingBoxes):    
                boxes = x    
            
        if image is None or boxes is None or torch.rand(1) >= self.p:    
            return {"zoom_in": False}    
                
        # Get image dimensions  
        if isinstance(image, (PIL.Image.Image, Image)):  
            img_w, img_h = image.size  
        else:  
            # For tensor images  
            img_h, img_w = image.shape[-2:]  
            
        # Check if there are small objects    
        box_sizes = boxes.tensor[:, 2:] - boxes.tensor[:, :2]  # width, height    
        box_areas = box_sizes[:, 0] * box_sizes[:, 1]    
            
        # Find small objects    
        small_obj_mask = box_areas < (self.min_size_threshold ** 2)    
            
        if not torch.any(small_obj_mask):    
            return {"zoom_in": False}    
                
        # Select a random small object to focus on    
        small_boxes = boxes.tensor[small_obj_mask]    
        target_idx = torch.randint(0, small_boxes.size(0), (1,)).item()    
        target_box = small_boxes[target_idx]    
            
        # Calculate center of the box    
        center_x = (target_box[0] + target_box[2]) / 2    
        center_y = (target_box[1] + target_box[3]) / 2    
            
        # Calculate box size    
        box_w = target_box[2] - target_box[0]    
        box_h = target_box[3] - target_box[1]    
            
        # Calculate scale factor to reach target size    
        w_scale = self.target_size_threshold / box_w    
        h_scale = self.target_size_threshold / box_h    
        scale = min(max(w_scale, h_scale), self.max_scale)    
        scale = max(scale, self.min_scale)    
            
        # Calculate new crop size    
        new_w = int(img_w / scale)    
        new_h = int(img_h / scale)    
            
        # Calculate crop coordinates    
        left = max(0, int(center_x - new_w / 2))    
        top = max(0, int(center_y - new_h / 2))    
            
        # Adjust if crop goes out of bounds    
        if left + new_w > img_w:    
            left = img_w - new_w    
        if top + new_h > img_h:    
            top = img_h - new_h    
                
        return {    
            "zoom_in": True,    
            "crop": [left, top, left + new_w, top + new_h],    
            "scale": scale    
        }    
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:    
        if not params["zoom_in"]:    
            return inpt    
                
        if isinstance(inpt, (PIL.Image.Image, Image)):    
            # Crop and resize for images    
            crop = params["crop"]  
            # Get original size  
            if isinstance(inpt, (PIL.Image.Image, Image)):  
                size = inpt.size  
            else:  
                size = (inpt.shape[-1], inpt.shape[-2])  # w, h  
                  
            return F.resized_crop(    
                inpt,     
                top=crop[1],     
                left=crop[0],     
                height=crop[3]-crop[1],     
                width=crop[2]-crop[0],     
                size=size  
            )    
        elif isinstance(inpt, BoundingBoxes):    
            # Adjust bounding boxes    
            crop = params["crop"]    
            boxes = inpt.tensor.clone()    
                
            # Shift boxes based on crop    
            boxes[:, 0] = (boxes[:, 0] - crop[0]) * params["scale"]    
            boxes[:, 1] = (boxes[:, 1] - crop[1]) * params["scale"]    
            boxes[:, 2] = (boxes[:, 2] - crop[0]) * params["scale"]    
            boxes[:, 3] = (boxes[:, 3] - crop[1]) * params["scale"]    
                
            # Clip to image boundaries    
            img_size = getattr(inpt, _boxes_keys[1])    
            boxes[:, 0].clamp_(min=0, max=img_size[1])    
            boxes[:, 1].clamp_(min=0, max=img_size[0])    
            boxes[:, 2].clamp_(min=0, max=img_size[1])    
            boxes[:, 3].clamp_(min=0, max=img_size[0])    
                
            return convert_to_tv_tensor(    
                boxes, key="boxes", box_format=inpt.format.value, spatial_size=img_size    
            )    
        elif isinstance(inpt, Mask):    
            # Crop and resize for masks    
            crop = params["crop"]    
            mask = F.crop(inpt, top=crop[1], left=crop[0], height=crop[3]-crop[1], width=crop[2]-crop[0])  
              
            # Get original size  
            if isinstance(inpt, (PIL.Image.Image, Image)):  
                size = inpt.size  
            else:  
                size = (inpt.shape[-1], inpt.shape[-2])  # w, h  
                  
            return F.resize(mask, size=size, interpolation=T.InterpolationMode.NEAREST)    
                
        return inpt