# YOLOv8 Inference and JSON Export

This script performs inference with a YOLOv8 model on test images and exports the detection results in a specific JSON format for submission.

## Requirements

```bash
pip install ultralytics tqdm
```

## Usage

### Basic Usage

```bash
python inference.py --test_folder /path/to/test/images --model best.pt
```

This will run inference on all images in the test folder using the default confidence threshold of 0.25 for all classes and save the results to `detections.json`.

### Class-Specific Confidence Thresholds

To set different confidence thresholds for each class:

```bash
python inference.py --test_folder /path/to/test/images --model best.pt --conf "0.1,0.2,0.3,0.4,0.5"
```

This sets:
- Class 0: threshold 0.1
- Class 1: threshold 0.2
- Class 2: threshold 0.3
- Class 3: threshold 0.4
- Class 4: threshold 0.5

### All Options

```bash
python inference.py --test_folder /path/to/test/images --model best.pt --output custom_output.json --conf "0.1,0.2,0.3,0.4,0.5"
```

## Parameters

- `--test_folder`: Path to the folder containing test images (required)
- `--model`: Path to the YOLOv8 model checkpoint (default: "best.pt")
- `--output`: Path for the output JSON file (default: "detections.json")
- `--conf`: Confidence thresholds for each class as a comma-separated string (default: "0.25,0.25,0.25,0.25,0.25")

## Output Format

The script produces a JSON file with the following format:

```json
[
  {
    "image_id": 123,
    "category_id": 0,
    "bbox": [x1, y1, width, height],
    "score": 0.95
  },
  // Additional detections...
]
```

Where:
- `image_id` is extracted from the image filename
- `category_id` is the class ID predicted by the model
- `bbox` contains the bounding box coordinates [x, y, width, height]
- `score` is the confidence score for the detection

## Example Output

```
Loading model from best.pt...
Processing images: 100%|████████████████████| 120/120 [00:15<00:00, 7.89it/s]
Saved 532 detections to detections.json
Detections by class:
  Class 0: 145 detections (threshold: 0.1)
  Class 1: 98 detections (threshold: 0.2)
  Class 2: 113 detections (threshold: 0.3)
  Class 3: 86 detections (threshold: 0.4)
  Class 4: 90 detections (threshold: 0.5)
```