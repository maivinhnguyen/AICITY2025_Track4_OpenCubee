# --- START OF FILE coco_evaluator.py ---
import json
import os
import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Suppress specific UserWarning from matplotlib about FixedFormatter
import warnings
warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")


class DetectionEvaluator:
    def __init__(self, gt_json_path, pred_json_path, output_dir="eval_results"):
        self.gt_json_path = Path(gt_json_path)
        self.pred_json_path = Path(pred_json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.gt_json_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.gt_json_path}")
        if not self.pred_json_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {self.pred_json_path}")

        print(f"Loading Ground Truth from: {self.gt_json_path}")
        self.coco_gt = COCO(str(self.gt_json_path))
        print(f"Loading Predictions from: {self.pred_json_path}")
        self.coco_dt = self.coco_gt.loadRes(str(self.pred_json_path)) # Use loadRes for typical pred files

        self.cat_ids = self.coco_gt.getCatIds()
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco_gt.loadCats(self.cat_ids)}
        # Ensure class 0 is handled if present in predictions but not explicitly in GT categories (rare)
        # Or map prediction category_ids to the ones present in GT
        pred_cat_ids = set(ann['category_id'] for ann in self.coco_dt.dataset['annotations'])
        for pred_cat_id in pred_cat_ids:
            if pred_cat_id not in self.cat_id_to_name:
                self.cat_id_to_name[pred_cat_id] = f"class_{pred_cat_id}"


        self.coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox') # 'bbox' for object detection

        # --- Parameters for COCOeval to get detailed match info ---
        # We'll use a specific IoU for some curves, e.g., 0.5
        self.iou_thresh_for_curves = 0.5
        self.iou_idx_for_curves = np.where(np.isclose(self.coco_eval.params.iouThrs, self.iou_thresh_for_curves))[0]
        if not self.iou_idx_for_curves.size > 0:
            # If 0.5 is not exactly in iouThrs, add it or pick closest. For simplicity, we'll assume it's there or error.
            # Or, we can manually set params.iouThrs for a specific eval run
            print(f"Warning: IoU threshold {self.iou_thresh_for_curves} not found in default COCOeval params. Using first IoU threshold.")
            self.iou_idx_for_curves = np.array([0]) # Default to the first IoU threshold (usually 0.5)
        else:
            self.iou_idx_for_curves = self.iou_idx_for_curves[0]


        self.area_rng_idx = 0  # All areas
        self.max_dets_idx = 2  # Index for 100 detections (usually [1, 10, 100])

        self.results_text_file = self.output_dir / "evaluation_summary.txt"
        self.results_log = []

    def _log(self, message):
        print(message)
        self.results_log.append(message)

    def run_coco_standard_eval(self):
        self._log("\n--- Running Standard COCO Evaluation ---")
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize() # Prints to console
        # coco_eval.stats contains the 12 standard metrics
        self.standard_metrics = self.coco_eval.stats
        self._log("\nStandard COCO Metrics:")
        metric_names = [
            "AP @IoU=0.50:0.95 | area=all | maxDets=100",
            "AP @IoU=0.50      | area=all | maxDets=100",
            "AP @IoU=0.75      | area=all | maxDets=100",
            "AP @IoU=0.50:0.95 | area=small | maxDets=100",
            "AP @IoU=0.50:0.95 | area=medium | maxDets=100",
            "AP @IoU=0.50:0.95 | area=large | maxDets=100",
            "AR @IoU=0.50:0.95 | area=all | maxDets=1",
            "AR @IoU=0.50:0.95 | area=all | maxDets=10",
            "AR @IoU=0.50:0.95 | area=all | maxDets=100",
            "AR @IoU=0.50:0.95 | area=small | maxDets=100",
            "AR @IoU=0.50:0.95 | area=medium | maxDets=100",
            "AR @IoU=0.50:0.95 | area=large | maxDets=100"
        ]
        for i, val in enumerate(self.standard_metrics):
            self._log(f"{metric_names[i]:<45}: {val:.4f}")

    def get_per_class_ap(self):
        self._log("\n--- Per-Class AP @IoU=0.5 ---")
        # Ensure evaluate() and accumulate() have been run
        if not hasattr(self.coco_eval, 'eval'):
            print("Warning: COCOeval.evaluate() and accumulate() must be run first for per-class AP.")
            return {}

        per_class_ap = {}
        precisions = self.coco_eval.eval['precision']
        # precisions shape: (iou_thresholds, recall_thresholds, category_ids, area_ranges, max_detections)

        # For AP@0.5, use the first IoU threshold index (usually 0.5)
        # Use all area ranges (index 0) and maxDets=100 (index 2)
        ap_iou_0_5_idx = 0 # Index for IoU=0.5 in coco_eval.params.iouThrs

        for cat_idx, cat_id in enumerate(self.cat_ids):
            # Get precision for this category, at IoU=0.5, across all recall thresholds
            # Dimensions: T, R, K, A, M
            # T = ap_iou_0_5_idx (for IoU=0.5)
            # R = : (all recall thresholds)
            # K = cat_idx
            # A = 0 (all area ranges)
            # M = 2 (maxDets = 100)
            p = precisions[ap_iou_0_5_idx, :, cat_idx, self.area_rng_idx, self.max_dets_idx]
            ap = np.mean(p[p > -1]) # Average over valid precision values
            if np.isnan(ap): # Handle cases where a class has no GT or no detections
                ap = 0.0
            class_name = self.cat_id_to_name.get(cat_id, f"class_{cat_id}")
            per_class_ap[class_name] = ap
            self._log(f"AP@0.5 for {class_name:<20}: {ap:.4f}")
        return per_class_ap

    def _get_pr_rc_f1_data_for_curves(self):
        """
        Extracts data needed for P-R, P-Conf, R-Conf, F1-Conf curves.
        This relies on the detailed evaluation stored in coco_eval.evalImgs after evaluate()
        """
        if not hasattr(self.coco_eval, 'evalImgs') or self.coco_eval.evalImgs is None:
            print("Running coco_eval.evaluate() to get detailed match info...")
            self.coco_eval.evaluate() # This populates evalImgs

        # evalImgs is a list of dicts, one per image_id that has GT.
        # Each dict contains:
        # 'dtIds', 'gtIds', 'dtMatches', 'gtMatches', 'dtScores', 'gtIgnore', 'dtIgnore'
        # dtMatches[category_idx, detection_idx] = IoU with matched GT, or 0
        # gtMatches[category_idx, gt_idx] = IoU with matched DT, or 0
        # dtScores[category_idx, detection_idx] = score of detection

        all_preds_details = [] # Store (score, is_tp, category_id)
        gt_counts_per_class = {cat_id: 0 for cat_id in self.cat_ids}

        img_ids_with_gt = self.coco_gt.getImgIds()

        for img_id in img_ids_with_gt:
            # Get evaluation details for this image, for the specific IoU for curves
            # coco_eval.evaluate() must have been run for all IoUs
            # We need to find the evalImg entry corresponding to this img_id
            eval_img_info = None
            for ei in self.coco_eval.evalImgs:
                if ei and ei['image_id'] == img_id and ei['category_id'] in self.cat_ids : # Ensure category matches one we care about
                    # Check if this evalImg entry is for the IoU threshold we want for curves
                    # Note: evalImgs stores results for *each* (image, category, iou_threshold)
                    # We need to iterate through coco_eval.evalImgs and pick the ones matching our self.iou_thresh_for_curves
                    # This is complex because evalImgs is structured internally.
                    # A simpler way is to re-run evaluate with specific params or parse its output carefully.

                    # Let's re-evaluate for the specific IoU to simplify extraction
                    # This is less efficient but clearer for this specific task.
                    break # Found an entry for this img_id (might not be for the right cat/iou)

            # To get TP/FP status for each detection at a specific IoU (e.g., 0.5):
            # Iterate through all predictions (self.coco_dt.dataset['annotations'])
            # Match them against GTs (self.coco_gt)
            # This is what coco_eval.evaluate() does internally. We want its per-detection match result.

        # --- More direct way using sorted predictions and matching (like sklearn) ---
        # This involves re-implementing the matching logic for TP/FP counts at varying confidences
        # Or, we can parse the output of coco_eval.eval['precision'] and coco_eval.eval['scores']

        # Let's use the output of `coco_eval.eval` which is populated by `accumulate()`
        if not hasattr(self.coco_eval, 'eval') or not self.coco_eval.eval['precision'].size:
            print("Running coco_eval.accumulate() for detailed PR data...")
            if not hasattr(self.coco_eval, 'evalImgs') or self.coco_eval.evalImgs is None:
                 self.coco_eval.evaluate()
            self.coco_eval.accumulate()

        eval_data = self.coco_eval.eval
        precision_data = eval_data['precision'] # Shape: [T, R, K, A, M] (IoU, Recall, Class, Area, MaxDets)
        recall_thresholds = self.coco_eval.params.recThrs # Standard 101 recall thresholds (0.0 to 1.0)

        # For P-R Curve (using self.iou_idx_for_curves)
        pr_data_per_class = {}
        for k_idx, cat_id in enumerate(self.cat_ids):
            class_name = self.cat_id_to_name.get(cat_id, f"class_{cat_id}")
            precisions = precision_data[self.iou_idx_for_curves, :, k_idx, self.area_rng_idx, self.max_dets_idx]
            valid_pts = precisions > -1
            pr_data_per_class[class_name] = {
                'recall': recall_thresholds[valid_pts],
                'precision': precisions[valid_pts]
            }

        # For F1/P/R vs. Confidence Curves
        # We need (confidence, precision, recall) tuples.
        # `eval_data['scores']` gives confidence scores corresponding to `eval_data['precision']` points.
        # This is a bit indirect. A more direct way for P/R/F1 vs Conf:
        
        # Store (score, tp_count_at_this_score, fp_count_at_this_score, class_id)
        # This requires iterating through sorted detections and for each, determining if it's TP or FP
        # This means we need detailed match results per detection.

        # Using the `evalImgs` approach is best for TP/FP counts per detection
        # Let's prepare data for this:
        # A list of (score, is_tp_at_iou_0.5, category_id) for ALL detections
        # `is_tp_at_iou_0.5` is 1 if the detection matched a GT with IoU >= 0.5, else 0.
        
        detailed_detections = []
        gt_counts = {cat_id: 0 for cat_id in self.cat_ids}
        # We need to run evaluate with self.iou_thresh_for_curves as the *only* IoU for evalImgs
        # Or parse the existing evalImgs carefully. Let's simplify:
        
        coco_eval_for_curves = COCOeval(self.coco_gt, self.coco_dt, 'bbox')
        coco_eval_for_curves.params.iouThrs = np.array([self.iou_thresh_for_curves]) # Focus on one IoU
        # coco_eval_for_curves.params.imgIds = self.coco_gt.getImgIds() # Evaluate on all images with GT
        # coco_eval_for_curves.params.catIds = self.cat_ids # Evaluate for all categories we care about
        coco_eval_for_curves.evaluate() # This populates evalImgs with matches for iou_thresh_for_curves

        for cat_idx, cat_id in enumerate(self.cat_ids):
            # Count total GTs for this class
            gt_anns_for_cat = self.coco_gt.getAnnIds(catIds=[cat_id])
            gt_counts[cat_id] = len(gt_anns_for_cat)

            # Iterate through images that have GTs for this class
            img_ids_with_gt_for_cat = self.coco_gt.getImgIds(catIds=[cat_id])
            for img_id in img_ids_with_gt_for_cat:
                # Find the evalImg entry for this img_id and cat_id
                # evalImgs is list of dicts, one per (img_id, cat_id, iou_thresh)
                # Since we set iouThrs to a single value, it's simpler.
                eval_entry = None
                for ei in coco_eval_for_curves.evalImgs:
                    if ei and ei['image_id'] == img_id and ei['category_id'] == cat_id:
                        eval_entry = ei
                        break
                
                if eval_entry:
                    dt_scores = eval_entry['dtScores'] # Scores of detections for this class in this image
                    dt_matches_iou = eval_entry['dtMatches'][0] # IoUs of matches (iouThrs has only one value)
                                                                # dtMatches[0][j] refers to the j-th detection of this category

                    for score, match_iou in zip(dt_scores, dt_matches_iou):
                        is_tp = 1 if match_iou >= self.iou_thresh_for_curves else 0
                        detailed_detections.append({'score': score, 'is_tp': is_tp, 'category_id': cat_id})
        
        # Sort all detections by score, descending
        detailed_detections.sort(key=lambda x: x['score'], reverse=True)

        # Calculate P, R, F1 at different confidence thresholds
        conf_curve_data_per_class = {
            self.cat_id_to_name.get(cat_id, f"class_{cat_id}"): {
                'conf': [], 'precision': [], 'recall': [], 'f1': []
            } for cat_id in self.cat_ids
        }
        # Also for "all classes"
        conf_curve_data_all_classes = {'conf': [], 'precision': [], 'recall': [], 'f1': []}
        
        tp_all = 0
        fp_all = 0
        total_gt_all = sum(gt_counts.values())

        # Per-class accumulators
        tp_cls = {cat_id: 0 for cat_id in self.cat_ids}
        fp_cls = {cat_id: 0 for cat_id in self.cat_ids}

        # Iterate through sorted detections to build curves
        # This creates points at each unique confidence score where a detection occurs
        unique_scores = sorted(list(set(d['score'] for d in detailed_detections)), reverse=True)
        if not unique_scores: # Handle case with no detections
            self._log("No detections found to generate confidence curves.")
            return pr_data_per_class, conf_curve_data_per_class, conf_curve_data_all_classes


        # Add a point for confidence threshold slightly above max score (P=0, R=0) or (P=1,R=0 if no FP)
        # And a point for confidence threshold 0 (all detections included)
        processed_confidences = set()

        # Calculate P, R, F1 for a pseudo-confidence of 1.0 (or slightly above max score)
        # This represents the state before any detections are considered "above threshold"
        # For R vs Conf, P vs Conf, F1 vs Conf, we want smooth curves.
        # Iterate through confidence thresholds from high to low.
        # At each unique score s_i in detections:
        #   Consider all detections with score >= s_i
        #   Calculate TP, FP for these.

        num_detections = len(detailed_detections)
        for i in range(num_detections + 1): # Iterate from 0 to num_detections
            current_conf_threshold = detailed_detections[i-1]['score'] if i > 0 else 1.01 # Pseudo high conf for start
            if i == num_detections: # After last detection, effectively conf_thresh = 0
                 current_conf_threshold = 0.0

            # if current_conf_threshold in processed_confidences and i < num_detections: # Skip if already processed (unless it's the final 0.0 step)
            #     if i > 0 and i < num_detections and detailed_detections[i]['score'] == current_conf_threshold: # if next score is same, skip to avoid duplicate x values
            #         continue
            # processed_confidences.add(current_conf_threshold)


            # Re-calculate TP/FP based on detections with score >= current_conf_threshold
            # This is cumulative as we go down in confidence
            tp_all_at_conf = 0
            fp_all_at_conf = 0
            tp_cls_at_conf = {cat_id: 0 for cat_id in self.cat_ids}
            fp_cls_at_conf = {cat_id: 0 for cat_id in self.cat_ids}

            # More efficient: update TP/FP incrementally
            if i > 0: # Update based on the i-th detection (0-indexed)
                det = detailed_detections[i-1]
                if det['is_tp']:
                    tp_all += 1
                    tp_cls[det['category_id']] += 1
                else:
                    fp_all += 1
                    fp_cls[det['category_id']] += 1
            
            # For "all classes"
            p_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
            r_all = tp_all / total_gt_all if total_gt_all > 0 else 0
            f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0
            
            # Only add if confidence is different from previous, or it's first/last point
            if not conf_curve_data_all_classes['conf'] or conf_curve_data_all_classes['conf'][-1] != current_conf_threshold:
                conf_curve_data_all_classes['conf'].append(current_conf_threshold)
                conf_curve_data_all_classes['precision'].append(p_all)
                conf_curve_data_all_classes['recall'].append(r_all)
                conf_curve_data_all_classes['f1'].append(f1_all)

            # For per-class
            for cat_id in self.cat_ids:
                class_name = self.cat_id_to_name.get(cat_id, f"class_{cat_id}")
                p_cls = tp_cls[cat_id] / (tp_cls[cat_id] + fp_cls[cat_id]) if (tp_cls[cat_id] + fp_cls[cat_id]) > 0 else 0
                r_cls = tp_cls[cat_id] / gt_counts[cat_id] if gt_counts[cat_id] > 0 else 0
                f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls) if (p_cls + r_cls) > 0 else 0
                
                if not conf_curve_data_per_class[class_name]['conf'] or \
                   conf_curve_data_per_class[class_name]['conf'][-1] != current_conf_threshold:
                    conf_curve_data_per_class[class_name]['conf'].append(current_conf_threshold)
                    conf_curve_data_per_class[class_name]['precision'].append(p_cls)
                    conf_curve_data_per_class[class_name]['recall'].append(r_cls)
                    conf_curve_data_per_class[class_name]['f1'].append(f1_cls)
        
        # Ensure curves start at R=0 for P-R and (Conf=1.0, Metric=0) for Conf curves
        # For PR curve, COCOeval output already handles this.
        for class_name in conf_curve_data_per_class:
            if 1.01 not in conf_curve_data_per_class[class_name]['conf']: # Add high confidence, zero metric point
                 conf_curve_data_per_class[class_name]['conf'].insert(0, 1.01)
                 conf_curve_data_per_class[class_name]['precision'].insert(0,0) # Or 1.0 if no FPs expected at high conf
                 conf_curve_data_per_class[class_name]['recall'].insert(0,0)
                 conf_curve_data_per_class[class_name]['f1'].insert(0,0)
        if 1.01 not in conf_curve_data_all_classes['conf']:
            conf_curve_data_all_classes['conf'].insert(0, 1.01)
            conf_curve_data_all_classes['precision'].insert(0,0)
            conf_curve_data_all_classes['recall'].insert(0,0)
            conf_curve_data_all_classes['f1'].insert(0,0)


        return pr_data_per_class, conf_curve_data_per_class, conf_curve_data_all_classes

    def plot_metric_vs_confidence(self, conf_data, metric_name, class_name="All Classes"):
        plt.figure(figsize=(10, 7))
        
        confs = np.array(conf_data['conf'])
        metric_values = np.array(conf_data[metric_name.lower()]) # 'f1', 'precision', 'recall'

        # Sort by confidence for plotting (should already be mostly sorted but ensure)
        sort_idx = np.argsort(confs)
        confs = confs[sort_idx]
        metric_values = metric_values[sort_idx]

        plt.plot(confs, metric_values, marker='.')
        plt.xlabel("Confidence Threshold")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs. Confidence Curve - {class_name} (IoU@{self.iou_thresh_for_curves})")
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        
        # Find optimal confidence for F1 if metric is F1
        if metric_name.lower() == 'f1' and len(metric_values) > 0:
            best_f1_idx = np.argmax(metric_values)
            best_f1 = metric_values[best_f1_idx]
            best_conf = confs[best_f1_idx]
            plt.plot(best_conf, best_f1, "ro", markersize=8, label=f"Best F1={best_f1:.3f} @ Conf={best_conf:.3f}")
            plt.legend()
            self._log(f"Optimal F1 for {class_name}: {best_f1:.4f} at confidence {best_conf:.4f}")


        filename = f"{metric_name.lower()}_conf_curve_{class_name.replace(' ', '_')}.png"
        plt.savefig(self.output_dir / filename)
        self._log(f"Saved {metric_name} vs. Confidence plot to {self.output_dir / filename}")
        plt.close()

    def plot_pr_curve(self, pr_data, class_name="All Classes"):
        plt.figure(figsize=(10, 7))
        recalls = np.array(pr_data['recall'])
        precisions = np.array(pr_data['precision'])

        # Sort by recall for proper plotting
        sort_idx = np.argsort(recalls)
        recalls = recalls[sort_idx]
        precisions = precisions[sort_idx]

        plt.plot(recalls, precisions, marker='.')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {class_name} (IoU@{self.iou_thresh_for_curves})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        filename = f"pr_curve_{class_name.replace(' ', '_')}.png"
        plt.savefig(self.output_dir / filename)
        self._log(f"Saved PR curve plot to {self.output_dir / filename}")
        plt.close()

    def run_evaluation(self):
        start_time = datetime.datetime.now()
        self._log(f"Evaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.run_coco_standard_eval()
        self.get_per_class_ap()

        self._log("\n--- Generating data for curves ---")
        pr_data_per_class, conf_curves_per_class, conf_curves_all_classes = \
            self._get_pr_rc_f1_data_for_curves()

        self._log("\n--- Plotting Curves ---")
        # Plot for "All Classes"
        if conf_curves_all_classes['conf']: # Check if data exists
            self.plot_metric_vs_confidence(conf_curves_all_classes, "F1", "All Classes")
            self.plot_metric_vs_confidence(conf_curves_all_classes, "Precision", "All Classes")
            self.plot_metric_vs_confidence(conf_curves_all_classes, "Recall", "All Classes")
            
            # For PR "All Classes", average PR data or use COCOeval stats for mAP's PR
            # The COCOeval `precision` array for K=all classes (or average) can be used
            # For simplicity here, we'll plot per-class PR curves.
            # To get an "All Classes" PR curve from COCOeval output:
            avg_precisions_for_pr_all = self.coco_eval.eval['precision'][self.iou_idx_for_curves, :, :, self.area_rng_idx, self.max_dets_idx]
            avg_precisions_for_pr_all = np.mean(avg_precisions_for_pr_all[avg_precisions_for_pr_all > -1].reshape(len(self.cat_ids), -1), axis=0) # Average over classes for each recall point
            
            if avg_precisions_for_pr_all.size > 0:
                 recall_thresholds = self.coco_eval.params.recThrs
                 valid_pts_all_pr = avg_precisions_for_pr_all.shape[0] == recall_thresholds.shape[0] # crude check
                 if not valid_pts_all_pr and avg_precisions_for_pr_all.shape[0] == np.sum(avg_precisions_for_pr_all > -1e-5): # if it's already filtered
                     pass # use as is
                 elif recall_thresholds[avg_precisions_for_pr_all > -1e-5].shape[0] == avg_precisions_for_pr_all[avg_precisions_for_pr_all > -1e-5].shape[0]: # If filtering makes them same shape
                     recall_thresholds_for_plot = recall_thresholds[avg_precisions_for_pr_all > -1e-5]
                     avg_precisions_for_pr_all_for_plot = avg_precisions_for_pr_all[avg_precisions_for_pr_all > -1e-5]
                 else: # Fallback if shapes don't align after filtering
                     recall_thresholds_for_plot = recall_thresholds
                     avg_precisions_for_pr_all_for_plot = avg_precisions_for_pr_all if avg_precisions_for_pr_all.shape == recall_thresholds.shape else np.zeros_like(recall_thresholds)


                 self.plot_pr_curve({'recall': recall_thresholds_for_plot, 'precision': avg_precisions_for_pr_all_for_plot}, "All Classes (avg)")


        # Plot per-class curves
        for class_name, pr_data in pr_data_per_class.items():
            if pr_data['recall'].size > 0 and pr_data['precision'].size > 0:
                self.plot_pr_curve(pr_data, class_name)
        
        for class_name, conf_data in conf_curves_per_class.items():
            if conf_data['conf']: # Check if data exists
                self.plot_metric_vs_confidence(conf_data, "F1", class_name)
                self.plot_metric_vs_confidence(conf_data, "Precision", class_name)
                self.plot_metric_vs_confidence(conf_data, "Recall", class_name)
        
        end_time = datetime.datetime.now()
        self._log(f"\nEvaluation finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"Total evaluation time: {end_time - start_time}")

        with open(self.results_text_file, 'w') as f:
            for line in self.results_log:
                f.write(line + '\n')
        print(f"\nFull evaluation summary saved to: {self.results_text_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="COCO Style Detection Evaluation Script")
    parser.add_argument('--gt_json', type=str, required=True, help="Path to COCO format ground truth JSON file.")
    parser.add_argument('--pred_json', type=str, required=True, help="Path to COCO format prediction JSON file.")
    parser.add_argument('--output_dir', type=str, default="eval_results", help="Directory to save plots and results summary.")
    # parser.add_argument('--iou_for_curves', type=float, default=0.5, help="IoU threshold to use for generating P-R, F1-Conf etc. curves.")

    args = parser.parse_args()

    # Example usage:
    # Create dummy GT and Pred JSON files for testing if you don't have them.
    # For a real run, replace with your actual file paths.

    # Create dummy GT if it doesn't exist (for simple testing)
    if not Path(args.gt_json).exists() and "dummy_gt.json" in args.gt_json:
        print("Creating dummy ground truth for testing...")
        dummy_gt_data = {
            "images": [{"id": 1, "width": 640, "height": 480, "file_name": "img1.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [50, 50, 100, 100], "iscrowd": 0, "area": 10000},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [200, 200, 50, 50], "iscrowd": 0, "area": 2500}
            ],
            "categories": [
                {"id": 0, "name": "cat", "supercategory": "animal"},
                {"id": 1, "name": "dog", "supercategory": "animal"}
            ]
        }
        with open("dummy_gt.json", 'w') as f:
            json.dump(dummy_gt_data, f, indent=2)
        args.gt_json = "dummy_gt.json"

    # Create dummy Pred if it doesn't exist
    if not Path(args.pred_json).exists() and "dummy_pred.json" in args.pred_json:
        print("Creating dummy predictions for testing...")
        dummy_pred_data = [
            {"image_id": 1, "category_id": 0, "bbox": [55, 55, 90, 90], "score": 0.9},
            {"image_id": 1, "category_id": 0, "bbox": [60, 60, 80, 80], "score": 0.95}, # Another detection for cat
            {"image_id": 1, "category_id": 1, "bbox": [210, 210, 40, 40], "score": 0.8},
            {"image_id": 1, "category_id": 0, "bbox": [300, 300, 30, 30], "score": 0.7} # A false positive cat
        ]
        with open("dummy_pred.json", 'w') as f:
            json.dump(dummy_pred_data, f, indent=2)
        args.pred_json = "dummy_pred.json"


    evaluator = DetectionEvaluator(
        gt_json_path=args.gt_json,
        pred_json_path=args.pred_json,
        output_dir=args.output_dir
    )
    evaluator.run_evaluation()

# --- END OF FILE coco_evaluator.py ---