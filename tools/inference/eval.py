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
        
        self.results_text_file = self.output_dir / "evaluation_summary.txt"
        self.results_log = []

        if not self.gt_json_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.gt_json_path}")
        if not self.pred_json_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {self.pred_json_path}")

        self._log(f"Loading Ground Truth from: {self.gt_json_path}")
        self.coco_gt = COCO(str(self.gt_json_path))
        self._log(f"Loading Predictions from: {self.pred_json_path}")
        try:
            with open(self.pred_json_path, 'r') as f:
                preds_list = json.load(f)
            if isinstance(preds_list, dict) and 'annotations' in preds_list:
                 self._log("Warning: Prediction JSON seems to be in full COCO format. Using 'annotations' field.")
                 preds_list = preds_list['annotations']
            self.coco_dt = self.coco_gt.loadRes(preds_list)
        except Exception as e:
            self._log(f"Error loading prediction file {self.pred_json_path} as JSON list. Trying direct loadRes with path. Error: {e}")
            self.coco_dt = self.coco_gt.loadRes(str(self.pred_json_path))

        self.cat_ids = self.coco_gt.getCatIds()
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco_gt.loadCats(self.cat_ids)}
        
        all_pred_anns = []
        if isinstance(self.coco_dt.dataset, dict) and 'annotations' in self.coco_dt.dataset:
             all_pred_anns = self.coco_dt.dataset['annotations']
        elif isinstance(self.coco_dt.dataset, list):
             all_pred_anns = self.coco_dt.dataset

        pred_cat_ids = set()
        if all_pred_anns:
            for ann in all_pred_anns:
                if 'category_id' in ann:
                    pred_cat_ids.add(ann['category_id'])
        
        for pred_cat_id in pred_cat_ids:
            if pred_cat_id not in self.cat_id_to_name:
                self.cat_id_to_name[pred_cat_id] = f"class_{pred_cat_id}"
        
        self.coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')

        # Determine IoU indices for plots averaged over IoU 0.5-0.95 (COCO standard range)
        iou_threshold_values_in_params = self.coco_eval.params.iouThrs
        self.iou_indices_for_averaged_plots = [
            idx for idx, iou_val in enumerate(iou_threshold_values_in_params)
            if 0.5 - 1e-6 <= iou_val <= 0.95 + 1e-6 # Standard COCO range for mAP
        ]
        
        if not self.iou_indices_for_averaged_plots:
            self._log("CRITICAL Warning: No IoU thresholds in the 0.5-0.95 range found in COCOeval params. Averaged plots will be based on fallback (first IoU).")
            # Fallback: use the index for 0.5 if available, or just the first one
            idx_0_5 = np.where(np.isclose(iou_threshold_values_in_params, 0.5))[0]
            if idx_0_5.size > 0:
                self.iou_indices_for_averaged_plots = [idx_0_5[0]]
            else:
                 self.iou_indices_for_averaged_plots = [0] # Absolute fallback

        self.num_ious_for_averaged_plots = len(self.iou_indices_for_averaged_plots)
        actual_ious_used = [iou_threshold_values_in_params[i] for i in self.iou_indices_for_averaged_plots]
        self._log(f"Using {self.num_ious_for_averaged_plots} IoU thresholds for averaged plots: "
                  f"{[f'{iou:.2f}' for iou in actual_ious_used]} (standard COCO 0.5-0.95 range).")

        self.area_rng_idx = 0
        self.max_dets_idx = 2 

    def _log(self, message):
        print(message)
        self.results_log.append(message)

    def run_coco_standard_eval(self):
        self._log("\n--- Running Standard COCO Evaluation ---")
        self.coco_eval.evaluate() 
        self.coco_eval.accumulate() 
        self.coco_eval.summarize() 
        
        self.standard_metrics = self.coco_eval.stats
        self._log("\nStandard COCO Metrics:")
        metric_names = [
            "AP @IoU=0.50:0.95 | area=all | maxDets=100", "AP @IoU=0.50      | area=all | maxDets=100",
            "AP @IoU=0.75      | area=all | maxDets=100", "AP @IoU=0.50:0.95 | area=small | maxDets=100",
            "AP @IoU=0.50:0.95 | area=medium | maxDets=100", "AP @IoU=0.50:0.95 | area=large | maxDets=100",
            "AR @IoU=0.50:0.95 | area=all | maxDets=1", "AR @IoU=0.50:0.95 | area=all | maxDets=10",
            "AR @IoU=0.50:0.95 | area=all | maxDets=100", "AR @IoU=0.50:0.95 | area=small | maxDets=100",
            "AR @IoU=0.50:0.95 | area=medium | maxDets=100", "AR @IoU=0.50:0.95 | area=large | maxDets=100"
        ]
        for i, val in enumerate(self.standard_metrics):
            self._log(f"{metric_names[i]:<45}: {val:.4f}")

    def get_per_class_ap(self): # This remains AP@0.5 as per its name
        self._log("\n--- Per-Class AP @IoU=0.5 ---")
        if not hasattr(self.coco_eval, 'eval') or not self.coco_eval.eval['precision'].size:
            self._log("Warning: COCOeval.accumulate() must be run first. Trying to run.")
            if not hasattr(self.coco_eval, 'evalImgs') or self.coco_eval.evalImgs is None: self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            if not hasattr(self.coco_eval, 'eval') or not self.coco_eval.eval['precision'].size:
                 self._log("Error: Failed to populate coco_eval.eval. Cannot compute per-class AP.")
                 return {}

        per_class_ap = {}
        precisions_data = self.coco_eval.eval['precision']
        iou_0_5_idx = np.where(np.isclose(self.coco_eval.params.iouThrs, 0.5))[0]
        if not iou_0_5_idx.size > 0:
            self._log("Warning: IoU=0.5 not found. Using index 0 for AP@0.5.")
            iou_0_5_idx = 0
        else:
            iou_0_5_idx = iou_0_5_idx[0]

        for cat_idx, cat_id in enumerate(self.cat_ids):
            p = precisions_data[iou_0_5_idx, :, cat_idx, self.area_rng_idx, self.max_dets_idx]
            ap = np.mean(p[p > -1]) 
            if np.isnan(ap): ap = 0.0
            class_name = self.cat_id_to_name.get(cat_id, f"class_{cat_id}")
            per_class_ap[class_name] = ap
            self._log(f"AP@0.5 for {class_name:<20}: {ap:.4f}")
        return per_class_ap

    def _get_gt_counts_map(self):
        gt_counts_map = {cat_id: 0 for cat_id in self.cat_ids}
        if not hasattr(self.coco_eval, 'evalImgs') or self.coco_eval.evalImgs is None:
            self._log("Error in _get_gt_counts_map: coco_eval.evalImgs not populated. Run evaluate() first.")
            return gt_counts_map

        target_area_rng_val = self.coco_eval.params.areaRng[self.area_rng_idx]
        target_max_dets_val = self.coco_eval.params.maxDets[self.max_dets_idx]
        temp_img_cat_gt_counted = {}

        for ei in self.coco_eval.evalImgs:
            if (ei and
                ei['aRng'] == target_area_rng_val and
                ei['maxDet'] == target_max_dets_val):
                cat_id = ei['category_id']
                img_id = ei['image_id']
                
                if (img_id, cat_id) not in temp_img_cat_gt_counted:
                    if cat_id in gt_counts_map:
                        num_gt_in_ei = sum(1 for ignored_flag in ei['gtIgnore'] if not ignored_flag)
                        gt_counts_map[cat_id] += num_gt_in_ei
                    temp_img_cat_gt_counted[(img_id, cat_id)] = True
        return gt_counts_map

    def _get_pr_rc_f1_data_for_averaged_curves(self):
        actual_ious_used_str = "0.50:0.05:0.95" if self.num_ious_for_averaged_plots == 10 and \
                               np.allclose(self.coco_eval.params.iouThrs[self.iou_indices_for_averaged_plots],
                                           np.linspace(0.5, 0.95, 10)) \
                               else ", ".join([f"{self.coco_eval.params.iouThrs[i]:.2f}" for i in self.iou_indices_for_averaged_plots])
        
        self._log(f"\n--- Generating data for P-R/F1-Conf curves (Averaged over IoUs: {actual_ious_used_str}) ---")
        
        if not self.iou_indices_for_averaged_plots:
            self._log("Error: No IoU indices selected for averaged plots. Cannot proceed.")
            empty_pr = {cn: {'recall': np.array([]), 'precision': np.array([])} for cn in self.cat_id_to_name.values()}
            empty_conf = {cn: {'conf': [], 'precision': [], 'recall': [], 'f1': []} for cn in self.cat_id_to_name.values()}
            empty_conf_all = {'conf': [], 'precision': [], 'recall': [], 'f1': []}
            return empty_pr, empty_conf, empty_conf_all
            
        pr_data_per_class_avg_iou = {}
        if hasattr(self.coco_eval, 'eval') and self.coco_eval.eval['precision'].size:
            precisions_from_eval = self.coco_eval.eval['precision']
            recall_thresholds = self.coco_eval.params.recThrs

            for k_idx, cat_id in enumerate(self.cat_ids):
                class_name = self.cat_id_to_name.get(cat_id, f"class_{cat_id}")
                prec_selected_ious_k = precisions_from_eval[self.iou_indices_for_averaged_plots, :, k_idx, self.area_rng_idx, self.max_dets_idx]
                
                prec_selected_ious_k_nan = np.where(prec_selected_ious_k == -1, np.nan, prec_selected_ious_k)
                avg_precisions_k = np.nanmean(prec_selected_ious_k_nan, axis=0)
                avg_precisions_k = np.nan_to_num(avg_precisions_k, nan=-1.0)

                valid_pts = avg_precisions_k > -1
                pr_data_per_class_avg_iou[class_name] = {
                    'recall': recall_thresholds[valid_pts],
                    'precision': avg_precisions_k[valid_pts]
                }
        else:
            self._log("Warning: coco_eval.eval not populated. Averaged P-R curves from COCOeval output cannot be generated.")
            pr_data_per_class_avg_iou = {self.cat_id_to_name.get(cat_id, f"class_{cat_id}"): {'recall': np.array([]), 'precision': np.array([])} for cat_id in self.cat_ids}

        if not hasattr(self.coco_eval, 'evalImgs') or self.coco_eval.evalImgs is None:
            self._log("Error: coco_eval.evalImgs not populated. Cannot generate averaged confidence curves.")
            empty_conf = {cn: {'conf': [], 'precision': [], 'recall': [], 'f1': []} for cn in self.cat_id_to_name.values()}
            empty_conf_all = {'conf': [], 'precision': [], 'recall': [], 'f1': []}
            return pr_data_per_class_avg_iou, empty_conf, empty_conf_all

        gt_counts_map = self._get_gt_counts_map()
        detailed_detections_for_conf_curves = []
        
        target_area_rng_val = self.coco_eval.params.areaRng[self.area_rng_idx]
        target_max_dets_val = self.coco_eval.params.maxDets[self.max_dets_idx]

        for ei in self.coco_eval.evalImgs:
            if not (ei and ei['aRng'] == target_area_rng_val and ei['maxDet'] == target_max_dets_val):
                continue

            cat_id = ei['category_id']
            if cat_id not in self.cat_ids: continue

            dt_scores = ei['dtScores']
            dt_matches_all_ious_indices = ei['dtMatches']
            
            for det_idx in range(len(dt_scores)):
                score = dt_scores[det_idx]
                tp_sum_for_this_det = 0
                
                for t_param_idx in self.iou_indices_for_averaged_plots:
                    match_gt_id = dt_matches_all_ious_indices[t_param_idx, det_idx]
                    if match_gt_id > 0:
                        tp_sum_for_this_det += 1
                
                detailed_detections_for_conf_curves.append({
                    'score': score, 'category_id': cat_id,
                    'tp_sum_across_ious': tp_sum_for_this_det,
                    'num_ious_considered': self.num_ious_for_averaged_plots
                })
        
        detailed_detections_for_conf_curves.sort(key=lambda x: x['score'], reverse=True)

        conf_curve_data_per_class_avg_iou = {
            self.cat_id_to_name.get(cat_id, f"class_{cat_id}"): 
                {'conf': [1.01], 'precision': [0.0], 'recall': [0.0], 'f1': [0.0]} 
            for cat_id in self.cat_ids
        }
        conf_curve_data_all_classes_avg_iou = {'conf': [1.01], 'precision': [0.0], 'recall': [0.0], 'f1': [0.0]}

        current_total_tps_agg_all = 0
        current_total_fps_agg_all = 0
        current_total_tps_agg_cls = {cat_id: 0 for cat_id in self.cat_ids}
        current_total_fps_agg_cls = {cat_id: 0 for cat_id in self.cat_ids}

        total_gts_overall_scaled = sum(gt_counts_map.values()) * self.num_ious_for_averaged_plots
        total_gts_cls_scaled = {cid: count * self.num_ious_for_averaged_plots for cid, count in gt_counts_map.items()}

        if not detailed_detections_for_conf_curves:
            self._log("No detections found to generate averaged confidence curves.")
        
        num_processed_dets = len(detailed_detections_for_conf_curves)
        for i in range(num_processed_dets):
            det = detailed_detections_for_conf_curves[i]
            current_conf_threshold = det['score']
            
            fp_sum_for_this_det = det['num_ious_considered'] - det['tp_sum_across_ious']

            current_total_tps_agg_all += det['tp_sum_across_ious']
            current_total_fps_agg_all += fp_sum_for_this_det
            
            if det['category_id'] in current_total_tps_agg_cls:
                current_total_tps_agg_cls[det['category_id']] += det['tp_sum_across_ious']
                current_total_fps_agg_cls[det['category_id']] += fp_sum_for_this_det

            is_last_det = (i == num_processed_dets - 1)
            conf_changed = (i + 1 < num_processed_dets and current_conf_threshold > detailed_detections_for_conf_curves[i+1]['score'])

            if is_last_det or conf_changed:
                p_all = current_total_tps_agg_all / (current_total_tps_agg_all + current_total_fps_agg_all) if (current_total_tps_agg_all + current_total_fps_agg_all) > 0 else 0
                r_all = current_total_tps_agg_all / total_gts_overall_scaled if total_gts_overall_scaled > 0 else 0
                f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0
                
                conf_curve_data_all_classes_avg_iou['conf'].append(current_conf_threshold)
                conf_curve_data_all_classes_avg_iou['precision'].append(p_all)
                conf_curve_data_all_classes_avg_iou['recall'].append(r_all)
                conf_curve_data_all_classes_avg_iou['f1'].append(f1_all)

                for cat_id_loop in self.cat_ids:
                    class_name = self.cat_id_to_name.get(cat_id_loop, f"class_{cat_id_loop}")
                    tp_c = current_total_tps_agg_cls[cat_id_loop]
                    fp_c = current_total_fps_agg_cls[cat_id_loop]
                    gt_total_c_scaled = total_gts_cls_scaled.get(cat_id_loop, 0)

                    p_cls = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
                    r_cls = tp_c / gt_total_c_scaled if gt_total_c_scaled > 0 else 0
                    f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls) if (p_cls + r_cls) > 0 else 0
                    
                    conf_curve_data_per_class_avg_iou[class_name]['conf'].append(current_conf_threshold)
                    conf_curve_data_per_class_avg_iou[class_name]['precision'].append(p_cls)
                    conf_curve_data_per_class_avg_iou[class_name]['recall'].append(r_cls)
                    conf_curve_data_per_class_avg_iou[class_name]['f1'].append(f1_cls)
        
        for data_dict_list in [conf_curve_data_all_classes_avg_iou] + list(conf_curve_data_per_class_avg_iou.values()):
            if not data_dict_list['conf'] or data_dict_list['conf'][-1] != 0.0:
                data_dict_list['conf'].append(0.0)
                data_dict_list['precision'].append(data_dict_list['precision'][-1] if data_dict_list['precision'] else 0.0)
                data_dict_list['recall'].append(data_dict_list['recall'][-1] if data_dict_list['recall'] else 0.0)
                data_dict_list['f1'].append(data_dict_list['f1'][-1] if data_dict_list['f1'] else 0.0)

        return pr_data_per_class_avg_iou, conf_curve_data_per_class_avg_iou, conf_curve_data_all_classes_avg_iou


    def calculate_per_class_average_max_f1(self):
        actual_ious_used_str = "0.50:0.05:0.95" if self.num_ious_for_averaged_plots == 10 and \
                               np.allclose(self.coco_eval.params.iouThrs[self.iou_indices_for_averaged_plots],
                                           np.linspace(0.5, 0.95, 10)) \
                               else ", ".join([f"{self.coco_eval.params.iouThrs[i]:.2f}" for i in self.iou_indices_for_averaged_plots])
        self._log(f"\n--- Calculating Per-Class Average Max F1 Score (over IoUs: {actual_ious_used_str}) ---")


        if not hasattr(self.coco_eval, 'evalImgs') or self.coco_eval.evalImgs is None:
            self._log("coco_eval.evalImgs not found. Ensure evaluate() has been run.")
            if not hasattr(self.coco_eval, 'stats'): self.coco_eval.evaluate()
            if not hasattr(self.coco_eval, 'evalImgs') or self.coco_eval.evalImgs is None:
                self._log("Error: coco_eval.evalImgs still not populated. Cannot calculate Average Max F1.")
                return {}

        target_iou_indices_for_calc = self.iou_indices_for_averaged_plots # Use the standard 0.5-0.95 range
        iou_threshold_values_in_params = self.coco_eval.params.iouThrs
        
        if not target_iou_indices_for_calc:
            self._log(f"Warning: No IoU thresholds ({actual_ious_used_str}) for Average Max F1 calculation.")
            return {}
        
        gt_counts_map = self._get_gt_counts_map()
        all_max_f1s_per_class_across_ious = {
            self.cat_id_to_name.get(cat_id, f"class_{cat_id}"): []
            for cat_id in self.cat_ids
        }
        
        target_area_rng_val = self.coco_eval.params.areaRng[self.area_rng_idx]
        target_max_dets_val = self.coco_eval.params.maxDets[self.max_dets_idx]

        for t_idx_in_params in target_iou_indices_for_calc:
            # iou_val = iou_threshold_values_in_params[t_idx_in_params] # Not strictly needed for logging here

            for cat_id in self.cat_ids:
                class_name = self.cat_id_to_name.get(cat_id, f"class_{cat_id}")
                current_detailed_detections_cls_iou = []
                for ei in self.coco_eval.evalImgs:
                    if (ei and ei['category_id'] == cat_id and
                        ei['aRng'] == target_area_rng_val and
                        ei['maxDet'] == target_max_dets_val):
                        
                        dt_scores_img_cat = ei['dtScores']
                        dt_matches_for_iou_t_img_cat = ei['dtMatches'][t_idx_in_params]
                        
                        for score, match_gt_id in zip(dt_scores_img_cat, dt_matches_for_iou_t_img_cat):
                            is_tp = 1 if match_gt_id > 0 else 0
                            current_detailed_detections_cls_iou.append({'score': score, 'is_tp': is_tp})
                
                current_detailed_detections_cls_iou.sort(key=lambda x: x['score'], reverse=True)
                
                gt_count_for_class = gt_counts_map.get(cat_id, 0)
                tp, fp = 0, 0
                f1_values_at_conf_thresholds = [0.0] 

                num_dets_for_class_iou = len(current_detailed_detections_cls_iou)
                for i in range(num_dets_for_class_iou):
                    det = current_detailed_detections_cls_iou[i]
                    if det['is_tp']: tp += 1
                    else: fp += 1
                    
                    p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r_val = tp / gt_count_for_class if gt_count_for_class > 0 else 0.0
                    f1_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0
                    f1_values_at_conf_thresholds.append(f1_val)

                max_f1_for_this_class_iou = np.max(f1_values_at_conf_thresholds) if f1_values_at_conf_thresholds else 0.0
                all_max_f1s_per_class_across_ious[class_name].append(max_f1_for_this_class_iou)

        final_avg_f1_scores = {}
        self._log(f"\n--- Per-Class Average Max F1 Scores (over IoUs: {actual_ious_used_str}) Results ---")
        for class_name, f1_list in all_max_f1s_per_class_across_ious.items():
            avg_max_f1 = np.mean(f1_list) if f1_list else 0.0
            final_avg_f1_scores[class_name] = avg_max_f1
            self._log(f"{class_name:<20}: {avg_max_f1:.4f}")
            
        return final_avg_f1_scores

    def _get_avg_iou_label_for_plots(self):
        if self.num_ious_for_averaged_plots == 10 and \
           np.allclose(self.coco_eval.params.iouThrs[self.iou_indices_for_averaged_plots],
                       np.linspace(0.5, 0.95, 10)):
            return "Avg IoU 0.50:0.05:0.95"
        else:
            return f"Avg IoUs: {', '.join([f'{self.coco_eval.params.iouThrs[i]:.2f}' for i in self.iou_indices_for_averaged_plots])}"


    def plot_per_class_metric_bar_chart(self, metric_data, y_axis_label, plot_title_prefix, file_key_suffix):
        if not metric_data:
            self._log(f"No data to plot for {y_axis_label} bar chart.")
            return

        class_names = list(metric_data.keys())
        scores = list(metric_data.values())
        avg_iou_label = self._get_avg_iou_label_for_plots()
        plot_title = f"{plot_title_prefix} ({avg_iou_label}) per Class"


        plt.figure(figsize=(max(10, int(len(class_names) * 0.6)), 7))
        try:
            bar_colors = plt.cm.get_cmap('viridis')(np.linspace(0.2, 0.8, len(class_names)))
        except:
            bar_colors = 'skyblue'
        bars = plt.bar(class_names, scores, color=bar_colors)
        plt.xlabel("Class")
        plt.ylabel(y_axis_label)
        plt.title(plot_title)
        plt.xticks(rotation=45, ha="right")
        max_score_val = max(0.05, max(scores) if scores else 0.05) 
        plt.ylim([0.0, min(1.05, max_score_val * 1.15)])
        plt.grid(axis='y', linestyle='--')

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005 * plt.gca().get_ylim()[1] , f'{yval:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        filename = f"{file_key_suffix}_per_class.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        self._log(f"Saved bar chart to {save_path}")
        plt.close()

    def plot_metric_vs_confidence(self, conf_data, metric_name, class_name="All Classes"):
        plt.figure(figsize=(10, 7))
        
        confs = np.array(conf_data['conf'])
        metric_values = np.array(conf_data[metric_name.lower()])

        sort_idx = np.argsort(confs)[::-1] 
        confs = confs[sort_idx]
        metric_values = metric_values[sort_idx]

        plt.plot(confs, metric_values, marker='.')
        plt.xlabel("Confidence Threshold")
        plt.ylabel(metric_name)
        
        avg_iou_label = self._get_avg_iou_label_for_plots()
        title = f"{metric_name} vs. Confidence - {class_name} ({avg_iou_label})"
        plt.title(title)
        plt.xlim([0.0, 1.02]) 
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        
        if metric_name.lower() == 'f1' and len(metric_values) > 0:
            valid_indices = np.where(confs <= 1.0)[0]
            if len(valid_indices) > 0:
                best_f1_idx_in_valid = np.argmax(metric_values[valid_indices])
                best_f1_idx_overall = valid_indices[best_f1_idx_in_valid]
                best_f1 = metric_values[best_f1_idx_overall]
                best_conf = confs[best_f1_idx_overall]
                plt.plot(best_conf, best_f1, "ro", markersize=8, label=f"Best F1={best_f1:.3f} @ Conf={best_conf:.3f}")
                plt.legend()
                self._log(f"Optimal F1 for {class_name} ({avg_iou_label}): {best_f1:.4f} at confidence {best_conf:.4f}")

        filename = f"{metric_name.lower()}_conf_curve_avg_iou_{class_name.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(self.output_dir / filename)
        self._log(f"Saved {metric_name} vs. Confidence plot ({avg_iou_label}) to {self.output_dir / filename}")
        plt.close()

    def plot_pr_curve(self, pr_data, class_name="All Classes"):
        plt.figure(figsize=(10, 7))
        recalls = np.array(pr_data['recall'])
        precisions = np.array(pr_data['precision'])

        sort_idx = np.argsort(recalls)
        recalls = recalls[sort_idx]
        precisions = precisions[sort_idx]
        
        plt.plot(recalls, precisions, marker='.')
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        avg_iou_label = self._get_avg_iou_label_for_plots()
        title = f"Precision-Recall Curve - {class_name} ({avg_iou_label})"
        plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        filename = f"pr_curve_avg_iou_{class_name.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(self.output_dir / filename)
        self._log(f"Saved PR curve plot ({avg_iou_label}) to {self.output_dir / filename}")
        plt.close()

    def run_evaluation(self):
        start_time = datetime.datetime.now()
        self._log(f"Evaluation started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.run_coco_standard_eval() 
        self.get_per_class_ap() 

        avg_max_f1_data = self.calculate_per_class_average_max_f1()
        if avg_max_f1_data:
            file_key = "avg_max_f1_std_coco_ious"
            self.plot_per_class_metric_bar_chart(avg_max_f1_data,
                                             "Average Max F1 Score",
                                             "Average Max F1 Score", # Title prefix, IoU range added by func
                                             file_key)

        pr_data_avg, conf_curves_per_class_avg, conf_curves_all_classes_avg = \
            self._get_pr_rc_f1_data_for_averaged_curves()

        avg_iou_label_log = self._get_avg_iou_label_for_plots()
        self._log(f"\n--- Plotting Averaged Curves ({avg_iou_label_log}) ---")
        
        if conf_curves_all_classes_avg['conf'] and len(conf_curves_all_classes_avg['conf']) > 1:
            self.plot_metric_vs_confidence(conf_curves_all_classes_avg, "F1", "All Classes")
            self.plot_metric_vs_confidence(conf_curves_all_classes_avg, "Precision", "All Classes")
            self.plot_metric_vs_confidence(conf_curves_all_classes_avg, "Recall", "All Classes")
            
            if hasattr(self.coco_eval, 'eval') and self.coco_eval.eval['precision'].size:
                prec_data = self.coco_eval.eval['precision'][self.iou_indices_for_averaged_plots, :, :, self.area_rng_idx, self.max_dets_idx]
                prec_data_nan = np.where(prec_data == -1, np.nan, prec_data)
                
                avg_prec_over_ious_then_classes = np.nanmean(np.nanmean(prec_data_nan, axis=0), axis=1)
                avg_prec_over_ious_then_classes = np.nan_to_num(avg_prec_over_ious_then_classes, nan=0.0)

                self.plot_pr_curve(
                    {'recall': self.coco_eval.params.recThrs, 'precision': avg_prec_over_ious_then_classes}, 
                    f"All Classes (Avg over Classes)" # IoU range added by plot_pr_curve
                )

        for class_name_key in self.cat_id_to_name.values():
            pr_data = pr_data_avg.get(class_name_key)
            if pr_data and pr_data['recall'].size > 0 and pr_data['precision'].size > 0:
                self.plot_pr_curve(pr_data, class_name_key)
            
            conf_data = conf_curves_per_class_avg.get(class_name_key)
            if conf_data and conf_data['conf'] and len(conf_data['conf']) > 1 :
                self.plot_metric_vs_confidence(conf_data, "F1", class_name_key)
                self.plot_metric_vs_confidence(conf_data, "Precision", class_name_key)
                self.plot_metric_vs_confidence(conf_data, "Recall", class_name_key)
        
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
    parser.add_argument('--pred_json', type=str, required=True, help="Path to COCO format prediction JSON file (list of detection dicts).")
    parser.add_argument('--output_dir', type=str, default="eval_results", help="Directory to save plots and results summary.")

    args = parser.parse_args()

    dummy_gt_filename = "dummy_gt.json"
    if Path(args.gt_json).name == dummy_gt_filename and not Path(args.gt_json).exists():
        print(f"Creating dummy ground truth: {dummy_gt_filename}")
        dummy_gt_data = {
            "images": [
                {"id": 1, "width": 640, "height": 480, "file_name": "img1.jpg"},
                {"id": 2, "width": 640, "height": 480, "file_name": "img2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [50, 50, 100, 100], "iscrowd": 0, "area": 10000},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [200, 200, 50, 50], "iscrowd": 0, "area": 2500},
                {"id": 3, "image_id": 2, "category_id": 0, "bbox": [10, 10, 30, 30], "iscrowd": 0, "area": 900},
                {"id": 4, "image_id": 2, "category_id": 0, "bbox": [70, 70, 30, 30], "iscrowd": 0, "area": 900}
            ],
            "categories": [
                {"id": 0, "name": "cat", "supercategory": "animal"},
                {"id": 1, "name": "dog", "supercategory": "animal"}
            ]}
        with open(dummy_gt_filename, 'w') as f: json.dump(dummy_gt_data, f, indent=2)
        args.gt_json = dummy_gt_filename

    dummy_pred_filename = "dummy_pred.json"
    if Path(args.pred_json).name == dummy_pred_filename and not Path(args.pred_json).exists():
        print(f"Creating dummy predictions: {dummy_pred_filename}")
        dummy_pred_data = [
            {"image_id": 1, "category_id": 0, "bbox": [55, 55, 90, 90], "score": 0.95},
            {"image_id": 1, "category_id": 0, "bbox": [60, 60, 80, 80], "score": 0.92}, 
            {"image_id": 1, "category_id": 1, "bbox": [210, 210, 40, 40], "score": 0.88},
            {"image_id": 1, "category_id": 0, "bbox": [300, 300, 30, 30], "score": 0.70},
            {"image_id": 2, "category_id": 0, "bbox": [12, 12, 28, 28], "score": 0.99},
            {"image_id": 2, "category_id": 0, "bbox": [72, 72, 28, 28], "score": 0.90},
            {"image_id": 2, "category_id": 1, "bbox": [100, 100, 50, 50], "score": 0.60}
        ]
        with open(dummy_pred_filename, 'w') as f: json.dump(dummy_pred_data, f, indent=2)
        args.pred_json = dummy_pred_filename

    evaluator = DetectionEvaluator(
        gt_json_path=args.gt_json,
        pred_json_path=args.pred_json,
        output_dir=args.output_dir
    )
    evaluator.run_evaluation()

# --- END OF FILE coco_evaluator.py ---