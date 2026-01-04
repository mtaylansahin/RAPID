"""Analysis manager that orchestrates single-run analysis and visualization."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.analysis.data_structures import StabilityMetrics, DEFAULT_STABILITY_GROUPS
from src.analysis.heatmap import HeatmapPlotter
from src.analysis.metrics_plots import MetricsPlotter
from src.analysis.reports import ReportGenerator


class AnalysisManager:
    """Manages single-run analysis including visualizations and reports.

    Integrates with the Evaluator to produce:
    - Interaction heatmaps (GT vs Predictions)
    - Stability-stratified metrics plots
    - Text and JSON reports
    - CSV data exports
    """

    def __init__(self, output_dir: Path, publication_mode: bool = False):
        """
        Initialize analysis manager.

        Args:
            output_dir: Directory to save all analysis outputs
            publication_mode: If True, add figure panel labels and output SVG
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.publication_mode = publication_mode

        self.heatmap_plotter = HeatmapPlotter(self.output_dir, publication_mode=publication_mode)
        self.metrics_plotter = MetricsPlotter(self.output_dir, publication_mode=publication_mode)
        self.report_generator = ReportGenerator(self.output_dir)

    def run_analysis(
        self,
        predictions: List[Tuple[int, int, int, int, float, int]],
        ground_truth: List[Tuple[int, int, int, int]],
        train_data: np.ndarray,
        valid_data: np.ndarray,
        test_timesteps: List[int],
        overall_metrics: Dict[str, float],
        per_timestep_metrics: Dict[str, Dict[int, float]],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, List[Path]]:
        """
        Run complete analysis pipeline.

        Args:
            predictions: List of (e1, rel, e2, t, score, pred) tuples
            ground_truth: List of (e1, rel, e2, t) tuples for test set
            train_data: Training data array
            valid_data: Validation data array
            test_timesteps: List of test timesteps
            overall_metrics: Overall evaluation metrics
            per_timestep_metrics: Per-timestep metrics
            dataset_info: Dataset metadata

        Returns:
            Dict mapping output type to list of generated file paths
        """
        print("\n" + "=" * 60)
        print("Running Analysis & Visualization")
        print("=" * 60)

        outputs = {"plots": [], "reports": []}

        # 1. Identify relevant pairs (Active in GT or Predicted as 1)
        gt_pairs = set()
        for e1, rel, e2, t in ground_truth:
            gt_pairs.add(f"{min(e1, e2)}_{max(e1, e2)}")

        pred_active_pairs = set()
        for e1, rel, e2, t, score, pred in predictions:
            if pred == 1:
                pred_active_pairs.add(f"{min(e1, e2)}_{max(e1, e2)}")

        relevant_pairs = gt_pairs.union(pred_active_pairs)

        # Optimize ground truth lookup
        gt_set = set(ground_truth)

        # Organize predictions by pair, FILTERING for relevant pairs
        pred_by_pair: Dict[
            str, List[Tuple[int, int]]
        ] = {}  # pair -> [(pred, label), ...]

        for e1, rel, e2, t, score, pred in predictions:
            pair = f"{min(e1, e2)}_{max(e1, e2)}"

            # Skip irrelevant pairs (dormant negatives)
            if pair not in relevant_pairs:
                continue

            if pair not in pred_by_pair:
                pred_by_pair[pair] = []

            label = 1 if (e1, rel, e2, t) in gt_set or (e2, rel, e1, t) in gt_set else 0
            pred_by_pair[pair].append((pred, label))

        # Compute consistent overall metrics (constrained scope)
        print("Computing overall metrics (constrained scope)...")
        overall_stats = self._compute_overall_metrics_from_preds(pred_by_pair)
        overall_metrics_dict = overall_stats.to_dict()

        # Compute stability-stratified metrics
        print("Computing stability-stratified metrics...")

        # Compute pair frequencies in training data
        train_pair_freq = self._compute_pair_frequencies(train_data)

        # Compute pair frequencies in test ground truth
        test_pair_freq = self._compute_test_frequencies(ground_truth, test_timesteps)

        # Compute metrics by training frequency
        stability_train = self._compute_metrics_by_frequency(
            pred_by_pair, train_pair_freq
        )

        # Compute metrics by test frequency
        stability_test = self._compute_metrics_by_frequency(
            pred_by_pair, test_pair_freq
        )

        # Generate heatmaps
        print("Generating heatmaps...")
        heatmap_paths = self.heatmap_plotter.generate_plots(
            predictions=predictions,  # Heatmap handles its own filtering/overlay
            ground_truth=ground_truth,
            train_data=train_data,
            valid_data=valid_data,
            test_timesteps=test_timesteps,
        )
        outputs["plots"].extend(heatmap_paths)

        # Generate metrics plots
        print("Generating metrics plots...")
        metrics_paths = self.metrics_plotter.generate_plots(
            stability_metrics_train=stability_train,
            stability_metrics_test=stability_test,
            overall_metrics=overall_metrics_dict,
        )
        outputs["plots"].extend(metrics_paths)

        # Generate reports
        print("Generating reports...")
        report_paths = self.report_generator.generate_all_reports(
            predictions=predictions,
            ground_truth=ground_truth,
            overall_metrics=overall_metrics_dict,
            per_timestep_metrics=per_timestep_metrics,
            stability_metrics_train=stability_train,
            stability_metrics_test=stability_test,
            dataset_info=dataset_info,
        )
        outputs["reports"].extend(report_paths)

        # Summary
        print("\n✓ Analysis complete!")
        print(f"  Plots: {len(outputs['plots'])}")
        print(f"  Reports: {len(outputs['reports'])}")
        print(f"  Output directory: {self.output_dir}")

        return outputs

    def _compute_stability_metrics(
        self,
        predictions: List[Tuple[int, int, int, int, float, int]],
        ground_truth: List[Tuple[int, int, int, int]],
        train_data: np.ndarray,
        test_timesteps: List[int],
    ) -> Tuple[Dict[str, StabilityMetrics], Dict[str, StabilityMetrics]]:
        """
        Compute stability-stratified metrics.

        Categorizes pairs into Rare/Transient/Stable based on frequency
        in training or test data.

        Restricts eavluation to relevant pairs (Active in GT or Predicted Positive),
        matching RNN-MD's evaluation protocol.
        """
        # 1. Identify relevant pairs (Active in GT or Predicted as 1)
        gt_pairs = set()
        for e1, rel, e2, t in ground_truth:
            gt_pairs.add(f"{min(e1, e2)}_{max(e1, e2)}")

        pred_active_pairs = set()
        for e1, rel, e2, t, score, pred in predictions:
            if pred == 1:
                pred_active_pairs.add(f"{min(e1, e2)}_{max(e1, e2)}")

        relevant_pairs = gt_pairs.union(pred_active_pairs)
        print(
            f"Restricting evaluation to {len(relevant_pairs)} relevant pairs (Active in GT or Predicted Positive)"
        )

        # Compute pair frequencies in training data
        train_pair_freq = self._compute_pair_frequencies(train_data)

        # Compute pair frequencies in test ground truth
        test_pair_freq = self._compute_test_frequencies(ground_truth, test_timesteps)

        # Organize predictions by pair, FILTERING for relevant pairs
        pred_by_pair: Dict[
            str, List[Tuple[int, int]]
        ] = {}  # pair -> [(pred, label), ...]

        # Optimize ground truth lookup
        gt_set = set(ground_truth)

        for e1, rel, e2, t, score, pred in predictions:
            pair = f"{min(e1, e2)}_{max(e1, e2)}"

            # Skip irrelevant pairs (dormant negatives)
            if pair not in relevant_pairs:
                continue

            if pair not in pred_by_pair:
                pred_by_pair[pair] = []

            label = 1 if (e1, rel, e2, t) in gt_set or (e2, rel, e1, t) in gt_set else 0
            pred_by_pair[pair].append((pred, label))

        # Compute metrics by training frequency
        stability_train = self._compute_metrics_by_frequency(
            pred_by_pair, train_pair_freq
        )

        # Compute metrics by test frequency
        stability_test = self._compute_metrics_by_frequency(
            pred_by_pair, test_pair_freq
        )

        return stability_train, stability_test

    def _compute_pair_frequencies(self, data: np.ndarray) -> Dict[str, float]:
        """Compute per-pair frequency as percentage of timesteps."""
        if len(data) == 0:
            return {}

        # Get unique timesteps
        timesteps = set(int(row[3]) for row in data)
        n_timesteps = len(timesteps)

        if n_timesteps == 0:
            return {}

        # Count timesteps per pair
        pair_timesteps: Dict[str, set] = {}
        for row in data:
            e1, _, e2, t = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            pair = f"{min(e1, e2)}_{max(e1, e2)}"
            if pair not in pair_timesteps:
                pair_timesteps[pair] = set()
            pair_timesteps[pair].add(t)

        # Convert to frequency percentage
        return {
            pair: (len(ts) / n_timesteps) * 100 for pair, ts in pair_timesteps.items()
        }

    def _compute_test_frequencies(
        self,
        ground_truth: List[Tuple[int, int, int, int]],
        test_timesteps: List[int],
    ) -> Dict[str, float]:
        """Compute per-pair frequency in test ground truth."""
        n_timesteps = len(test_timesteps)
        if n_timesteps == 0:
            return {}

        pair_timesteps: Dict[str, set] = {}
        for e1, rel, e2, t in ground_truth:
            pair = f"{min(e1, e2)}_{max(e1, e2)}"
            if pair not in pair_timesteps:
                pair_timesteps[pair] = set()
            pair_timesteps[pair].add(t)

        return {
            pair: (len(ts) / n_timesteps) * 100 for pair, ts in pair_timesteps.items()
        }

    def _compute_metrics_by_frequency(
        self,
        pred_by_pair: Dict[str, List[Tuple[int, int]]],
        pair_freq: Dict[str, float],
    ) -> Dict[str, StabilityMetrics]:
        """Compute metrics including MCC and Mean Pairwise F1 for each stability group."""
        # Group pairs by stability
        groups = {g.name: [] for g in DEFAULT_STABILITY_GROUPS}
        # Iterate over all relevant pairs that we have predictions for
        for pair, preds_labels in pred_by_pair.items():
            freq = pair_freq.get(pair, 0.0)

            for group in DEFAULT_STABILITY_GROUPS:
                if group.min_freq <= freq < group.max_freq:
                    # Storing (pred, label) tuples, but we also need pair-level grouping for Mean Pairwise F1
                    # So we store list of (pair, preds_labels)
                    groups[group.name].append((pair, preds_labels))
                    break

        # Compute metrics per group
        result = {}
        for group_name, pair_data_list in groups.items():
            # pair_data_list is List[Tuple[pair, List[(pred, label)]]]

            # Flatten for overall metrics
            all_preds = []
            all_labels = []

            # For Mean Pairwise F1
            pair_f1s = []

            for pair, preds_labels in pair_data_list:
                p_list = [p for p, _ in preds_labels]
                l_list = [l for _, l in preds_labels]

                all_preds.extend(p_list)
                all_labels.extend(l_list)

                # Calculate Pairwise F1
                tp_p = sum(1 for p, l in zip(p_list, l_list) if p == 1 and l == 1)
                fp_p = sum(1 for p, l in zip(p_list, l_list) if p == 1 and l == 0)
                fn_p = sum(1 for p, l in zip(p_list, l_list) if p == 0 and l == 1)

                prec_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
                rec_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0
                f1_p = (
                    2 * prec_p * rec_p / (prec_p + rec_p)
                    if (prec_p + rec_p) > 0
                    else 0.0
                )
                pair_f1s.append(f1_p)

            pair_count = len(pair_data_list)
            mean_pairwise_f1 = float(np.mean(pair_f1s)) if pair_f1s else 0.0

            # Calculate Global Metrics for the group
            tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
            tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # MCC
            numer = (tp * tn) - (fp * fn)
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = numer / denom if denom > 0 else 0.0

            result[group_name] = StabilityMetrics(
                group_name=group_name,
                pair_count=pair_count,
                recall=recall,
                precision=precision,
                f1=f1,
                mcc=mcc,
                tpr=recall,
                fpr=fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                mean_pairwise_f1=mean_pairwise_f1,
            )

        return result

    def _compute_overall_metrics_from_preds(
        self, pred_by_pair: Dict[str, List[Tuple[int, int]]]
    ) -> StabilityMetrics:
        """Compute overall metrics from predictions dict."""
        all_preds = []
        all_labels = []
        pair_f1s = []

        for pair, preds_labels in pred_by_pair.items():
            p_list = [p for p, _ in preds_labels]
            l_list = [l for _, l in preds_labels]

            all_preds.extend(p_list)
            all_labels.extend(l_list)

            # Pairwise F1
            tp_p = sum(1 for p, l in zip(p_list, l_list) if p == 1 and l == 1)
            fp_p = sum(1 for p, l in zip(p_list, l_list) if p == 1 and l == 0)
            fn_p = sum(1 for p, l in zip(p_list, l_list) if p == 0 and l == 1)

            prec_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
            rec_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0
            f1_p = (
                2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0.0
            )
            pair_f1s.append(f1_p)

        mean_pairwise_f1 = float(np.mean(pair_f1s)) if pair_f1s else 0.0

        # Global metrics
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        numer = (tp * tn) - (fp * fn)
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numer / denom if denom > 0 else 0.0

        return StabilityMetrics(
            group_name="Overall",
            pair_count=len(pred_by_pair),
            recall=recall,
            precision=precision,
            f1=f1,
            mcc=mcc,
            tpr=recall,
            fpr=fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            mean_pairwise_f1=mean_pairwise_f1,
        )
