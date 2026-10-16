"""
Benchmark Evaluation Engine for Active Fire Triage.
Computes Confusion Matrix, Precision, Recall, F1, Specificity,
and Threshold Tradeoff Curves against ground-truth labelled datasets.

Architecture Overview:
    This module evaluates the triage ranking system's ability to distinguish
    true wildfires from non-wildfire thermal anomalies (agricultural burns,
    industrial flares, solar glint) using ground-truth annotated datasets.

    The evaluation workflow:
        1. Filters the dataset for records that have a 'ground_truth_label' field.
        2. Applies a binary classification at a configurable priority_score
           threshold: score ≥ threshold → predicted wildfire (positive);
           score < threshold → predicted non-wildfire (negative).
        3. Computes the confusion matrix (TP, FP, FN, TN) and derived metrics
           (accuracy, precision, recall, specificity, F1 score).
        4. Breaks down performance by subgroup (true_wildfire, agricultural_burn,
           industrial_flare, false_alarm_glint).
        5. Sweeps across threshold values (20–90 in steps of 5) to generate
           precision-recall and F1 tradeoff curves.
        6. Documents known scientific limitations of satellite-based fire detection.

    Ground Truth Label Definitions:
        - 'true_wildfire': Confirmed uncontrolled wildfire verified by ground
          reports, incident command, or multi-source corroboration.
        - 'agricultural_burn': Permitted, controlled agricultural stubble or
          land-clearing burn.
        - 'industrial_flare': Gas flare from oil/gas facility, refinery, or
          metallurgical plant.
        - 'false_alarm_glint': Solar glint off reflective surfaces (water,
          metal roofs, solar panels) or cloud-edge thermal artifact.
"""

import math

# ===== EVALUATION ENGINE =====

class EvaluationEngine:
    """Evaluates triage ranking effectiveness against ground-truth annotations.

    This engine treats fire triage as a binary classification problem:
        - Positive class: 'true_wildfire' (actual uncontrolled fire)
        - Negative class: Everything else (agricultural_burn, industrial_flare,
          false_alarm_glint)

    The priority_score from ExplainableTriageRanker serves as the classifier's
    output, and the threshold parameter controls the decision boundary.
    """
    
    @staticmethod
    def evaluate(scored_records, threshold=60.0):
        """Evaluates triage accuracy against ground-truth labels.

        Args:
            scored_records: List of record dicts that have been through the
                full triage pipeline (must have 'priority_score' and optionally
                'ground_truth_label' fields).
            threshold: Priority score threshold for binary classification.
                Records with priority_score ≥ threshold are predicted as
                wildfires (positive); below threshold as non-wildfire (negative).
                Default: 60.0.

        Returns:
            Dict containing:
                - has_data: Boolean indicating whether ground-truth labels were found.
                - sample_size: Number of labelled records evaluated.
                - threshold: The threshold value used.
                - confusion_matrix: Dict with tp, fp, fn, tn counts.
                - metrics: Dict with accuracy, precision, recall, specificity, f1_score.
                - subgroups: Per-label breakdown of flagged_high vs flagged_low.
                - threshold_curve: List of metric dicts at each threshold (20–90).
                - scientific_limitations: List of known limitations affecting detection.
        """
        # Filter for records with ground truth — unlabelled records cannot be evaluated
        labelled = [r for r in scored_records if r.get('ground_truth_label')]
        
        if not labelled:
            return {
                "has_data": False,
                "message": "No ground-truth labels found in the current dataset."
            }
        
        # ===== CONFUSION MATRIX COMPUTATION =====
        # In the context of satellite fire detection:
        #   TP (True Positive):  Record IS a true wildfire AND the triage system
        #                        scored it ≥ threshold (correctly flagged as high priority)
        #   FP (False Positive): Record is NOT a wildfire (e.g., agricultural burn,
        #                        industrial flare, glint) BUT the triage system
        #                        scored it ≥ threshold (incorrectly flagged as high priority)
        #   FN (False Negative): Record IS a true wildfire BUT the triage system
        #                        scored it < threshold (missed — dangerously under-triaged)
        #   TN (True Negative):  Record is NOT a wildfire AND the triage system
        #                        scored it < threshold (correctly deprioritized)
        tp = 0 # True Wildfire & Score >= Threshold
        fp = 0 # Non-Wildfire (Agri, Flare, Glint) & Score >= Threshold
        fn = 0 # True Wildfire & Score < Threshold
        tn = 0 # Non-Wildfire & Score < Threshold
        
        # Subgroup counts: track performance broken down by each ground-truth
        # label type, allowing analysts to see which types of non-wildfire
        # anomalies are most often confused with real fires
        subgroup_breakdown = {
            'true_wildfire': {'total': 0, 'flagged_high': 0, 'flagged_low': 0},
            'agricultural_burn': {'total': 0, 'flagged_high': 0, 'flagged_low': 0},
            'industrial_flare': {'total': 0, 'flagged_high': 0, 'flagged_low': 0},
            'false_alarm_glint': {'total': 0, 'flagged_high': 0, 'flagged_low': 0}
        }
        
        for r in labelled:
            label = r['ground_truth_label']
            # Binary ground truth: only 'true_wildfire' counts as a positive
            is_actual_wildfire = (label == 'true_wildfire')
            # Binary prediction: score at or above threshold = predicted wildfire
            is_pred_wildfire = (r['priority_score'] >= threshold)
            
            # Track per-subgroup statistics
            if label in subgroup_breakdown:
                subgroup_breakdown[label]['total'] += 1
                if is_pred_wildfire:
                    subgroup_breakdown[label]['flagged_high'] += 1
                else:
                    subgroup_breakdown[label]['flagged_low'] += 1
            
            # Populate the confusion matrix
            if is_actual_wildfire and is_pred_wildfire:
                tp += 1
            elif not is_actual_wildfire and is_pred_wildfire:
                fp += 1
            elif is_actual_wildfire and not is_pred_wildfire:
                fn += 1
            else:
                tn += 1
        
        # ===== PERFORMANCE METRICS =====
        total = tp + fp + fn + tn
        
        # Accuracy = (TP + TN) / Total
        # Overall fraction of correct predictions (both positive and negative).
        # Can be misleading with imbalanced classes.
        accuracy = round((tp + tn) / total, 3) if total > 0 else 0.0
        
        # Precision = TP / (TP + FP)
        # Of all records the system flagged as high priority, what fraction
        # were actually true wildfires? High precision = few false alarms.
        precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        
        # Recall (Sensitivity) = TP / (TP + FN)
        # Of all actual true wildfires, what fraction did the system correctly
        # flag as high priority? High recall = few missed fires.
        # In fire safety, recall is typically prioritized over precision
        # (missing a real fire is worse than a false alarm).
        recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        
        # Specificity = TN / (TN + FP)
        # Of all non-wildfire records, what fraction did the system correctly
        # deprioritize? High specificity = non-fires are correctly filtered out.
        specificity = round(tn / (tn + fp), 3) if (tn + fp) > 0 else 0.0
        
        # F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
        # Harmonic mean of precision and recall. Balances the tradeoff between
        # false alarms and missed fires. F1 = 1.0 is perfect; F1 = 0.0 is worst.
        f1 = round(2 * (precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0.0
        
        # ===== THRESHOLD TRADEOFF CURVE =====
        # Sweep across threshold values from 20 to 90 (step 5) to show how
        # precision, recall, and F1 change as the decision boundary moves.
        #
        # At LOW thresholds (e.g., 20):
        #   - Almost everything is flagged → high recall (few missed fires)
        #     but low precision (many false alarms)
        # At HIGH thresholds (e.g., 90):
        #   - Only the strongest signals are flagged → high precision (few false
        #     alarms) but low recall (many real fires missed)
        # The optimal threshold balances these tradeoffs based on operational
        # requirements (e.g., prioritize recall for safety-critical applications).
        threshold_curve = []
        for t in range(20, 95, 5):
            # Recompute confusion matrix at each candidate threshold
            c_tp = sum(1 for r in labelled if r['ground_truth_label'] == 'true_wildfire' and r['priority_score'] >= t)
            c_fp = sum(1 for r in labelled if r['ground_truth_label'] != 'true_wildfire' and r['priority_score'] >= t)
            c_fn = sum(1 for r in labelled if r['ground_truth_label'] == 'true_wildfire' and r['priority_score'] < t)
            c_tn = sum(1 for r in labelled if r['ground_truth_label'] != 'true_wildfire' and r['priority_score'] < t)
            
            # Precision at this threshold (default 1.0 when no positive predictions)
            c_prec = round(c_tp / (c_tp + c_fp), 3) if (c_tp + c_fp) > 0 else 1.0
            # Recall at this threshold
            c_rec = round(c_tp / (c_tp + c_fn), 3) if (c_tp + c_fn) > 0 else 0.0
            # F1 at this threshold
            c_f1 = round(2 * (c_prec * c_rec) / (c_prec + c_rec), 3) if (c_prec + c_rec) > 0 else 0.0
            # False Positive Rate = FP / (FP + TN) — used for ROC-style analysis
            c_fpr = round(c_fp / (c_fp + c_tn), 3) if (c_fp + c_tn) > 0 else 0.0
            
            threshold_curve.append({
                'threshold': t,
                'precision': c_prec,
                'recall': c_rec,
                'f1': c_f1,
                'false_positive_rate': c_fpr
            })
        
        # ===== SCIENTIFIC LIMITATIONS =====
        # These are fundamental physical and operational limitations of
        # satellite-based thermal fire detection that cannot be overcome
        # by algorithmic improvements alone. They must be communicated
        # to analysts to prevent over-reliance on satellite data.
        return {
            "has_data": True,
            "sample_size": total,
            "threshold": threshold,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn
            },
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1
            },
            "subgroups": subgroup_breakdown,
            "threshold_curve": threshold_curve,
            "scientific_limitations": [
                {
                    "factor": "Cloud Obscuration & Attenuation",
                    "impact": "High (False Negatives)",
                    # Thermal infrared radiation (3.9 µm and 11 µm) cannot penetrate
                    # optically thick clouds or dense convective smoke columns.
                    # Fires burning beneath cloud cover are completely invisible to
                    # the satellite sensor, leading to systematic under-counting
                    # (false negatives) during cloudy/smoky conditions.
                    "detail": "Satellites cannot observe thermal infrared signatures through dense cloud cover or thick convective smoke. Fires burning beneath clouds will be omitted entirely."
                },
                {
                    "factor": "Orbital Revisit Interval (Latency)",
                    "impact": "Moderate to High",
                    # Polar sun-synchronous satellites (NOAA-20, Suomi-NPP, Terra, Aqua)
                    # each pass over a given location only 2 times per day (ascending
                    # and descending nodes). Combined, the VIIRS constellation provides
                    # ~4 passes/day, but gaps of 3-6 hours exist. A wildfire that ignites,
                    # runs, and is suppressed between passes will never be detected.
                    "detail": "Polar sun-synchronous satellites (Terra/Aqua/VIIRS) only image a given location 2 to 4 times per 24 hours. Rapidly evolving wildfire runs between passes are unobserved."
                },
                {
                    "factor": "Pixel Footprint & Sub-Pixel Blending",
                    "impact": "Moderate",
                    # A single VIIRS I-band pixel covers ~375m × 375m ≈ 14 hectares
                    # at nadir. The fire could be a small intense spot (100 m²) or a
                    # large diffuse warm area filling the pixel. The reported brightness
                    # temperature is an area-weighted average of the fire and non-fire
                    # portions of the pixel. The location is the pixel center, not the
                    # fire's actual position — geolocation uncertainty is ±190m at nadir,
                    # worse at swath edge.
                    "detail": "A single 375m VIIRS pixel can represent a small 100 m² high-intensity crown fire or an extensive diffuse warm soil surface. Location is reported at the pixel center, not the exact fire flank."
                },
                {
                    "factor": "Industrial & Agricultural False Triggers",
                    "impact": "Moderate (False Positives)",
                    # Legitimate thermal anomalies that are not uncontrolled wildfires:
                    # - Gas flare stacks at oil/gas facilities burn continuously at
                    #   >1000 K and produce strong, persistent thermal signatures.
                    # - Agricultural stubble burning is intentional and legal in many
                    #   regions but produces genuine fire detections.
                    # - Blast furnaces and steel mills emit strong mid-IR radiation.
                    # These are "true" thermal anomalies but not emergencies.
                    "detail": "Refinery flare stacks, metallurgical blast furnaces, and permitted agricultural burns produce valid thermal anomalies but are not uncontrolled wildfires."
                },
                {
                    "factor": "Off-Nadir Pixel Distortions (Bow-Tie Effect)",
                    "impact": "Low to Moderate",
                    # As the scan mirror sweeps from nadir to swath edge, the
                    # instantaneous field of view (IFOV) elongates in both scan
                    # and track directions. At the extreme edge of VIIRS's 3040 km
                    # swath, pixels can be stretched by ~2.5× their nadir size.
                    # This "bow-tie" effect (named for the shape of overlapping
                    # scan lines) degrades spatial precision and can cause a
                    # single fire to be detected as multiple adjacent pixels,
                    # or conversely, multiple small fires to merge into one pixel.
                    "detail": "Pixels near swath edges are stretched by up to 2.5x in scan and track directions, lowering spatial precision."
                }
            ]
        }
