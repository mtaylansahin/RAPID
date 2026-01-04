"""Report generation for analysis outputs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from src.analysis.data_structures import EvaluationResult, StabilityMetrics


class ReportGenerator:
    """Generates text and structured reports from evaluation results."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_reports(
        self,
        predictions: List[Tuple[int, int, int, int, float, int]],
        ground_truth: List[Tuple[int, int, int, int]],
        overall_metrics: Dict[str, float],
        per_timestep_metrics: Dict[str, Dict[int, float]],
        stability_metrics_train: Dict[str, StabilityMetrics],
        stability_metrics_test: Dict[str, StabilityMetrics],
        dataset_info: Dict[str, Any],
    ) -> List[Path]:
        """
        Generate all report files.

        Args:
            predictions: List of (e1, rel, e2, t, score, pred) tuples
            ground_truth: List of (e1, rel, e2, t) tuples
            overall_metrics: Dict with auroc, auprc, precision, recall, f1, mcc
            per_timestep_metrics: Dict with 'f1' and 'auprc' sub-dicts
            stability_metrics_train: Metrics by training frequency
            stability_metrics_test: Metrics by test frequency
            dataset_info: Dataset metadata

        Returns:
            List of generated report paths
        """
        generated_reports = []

        # Performance metrics text file
        path = self._write_performance_txt(
            overall_metrics,
            per_timestep_metrics,
            stability_metrics_train,
            stability_metrics_test,
            dataset_info,
        )
        generated_reports.append(path)

        # Structured JSON
        path = self._write_metrics_json(
            overall_metrics,
            per_timestep_metrics,
            stability_metrics_train,
            stability_metrics_test,
            dataset_info,
        )
        generated_reports.append(path)

        # Ground truth CSV
        path = self._write_ground_truth_csv(ground_truth)
        generated_reports.append(path)

        # Predictions CSV
        path = self._write_predictions_csv(predictions)
        generated_reports.append(path)

        return generated_reports

    def _write_performance_txt(
        self,
        overall: Dict[str, float],
        per_timestep: Dict[str, Dict[int, float]],
        stability_train: Dict[str, StabilityMetrics],
        stability_test: Dict[str, StabilityMetrics],
        dataset_info: Dict[str, Any],
    ) -> Path:
        """Write human-readable performance metrics."""
        path = self.output_dir / "PerformanceMetrics.txt"

        lines = []
        lines.append("=" * 60)
        lines.append("RAPID Evaluation Results")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        # Dataset info
        lines.append("DATASET INFORMATION")
        lines.append("-" * 40)
        for key, value in dataset_info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Overall metrics
        lines.append("OVERALL METRICS")
        lines.append("-" * 40)
        lines.append(f"  AUROC:     {overall.get('auroc', 0):.4f}")
        lines.append(f"  AUPRC:     {overall.get('auprc', 0):.4f}")
        lines.append(f"  Precision: {overall.get('precision', 0):.4f}")
        lines.append(f"  Recall:    {overall.get('recall', 0):.4f}")
        lines.append(f"  F1 Score:  {overall.get('f1', 0):.4f}")
        lines.append(f"  MCC:       {overall.get('mcc', 0):.4f}")
        lines.append("")

        # Per-timestep summary
        if per_timestep.get("f1"):
            f1_values = list(per_timestep["f1"].values())
            lines.append("PER-TIMESTEP SUMMARY")
            lines.append("-" * 40)
            lines.append(f"  Mean F1:   {sum(f1_values) / len(f1_values):.4f}")
            lines.append(f"  Min F1:    {min(f1_values):.4f}")
            lines.append(f"  Max F1:    {max(f1_values):.4f}")
            lines.append("")

        # Stability metrics (training frequency)
        if stability_train:
            lines.append("STABILITY METRICS (Training Frequency)")
            lines.append("-" * 40)
            for group in ["Rare", "Transient", "Stable"]:
                if group in stability_train:
                    m = stability_train[group]
                    lines.append(f"  {group} (n={m.pair_count:,}):")
                    lines.append(f"    Precision: {m.precision:.4f}")
                    lines.append(f"    Recall:    {m.recall:.4f}")
                    lines.append(f"    F1:        {m.f1:.4f}")
            lines.append("")

        # Stability metrics (test frequency)
        if stability_test:
            lines.append("STABILITY METRICS (Test Frequency)")
            lines.append("-" * 40)
            for group in ["Rare", "Transient", "Stable"]:
                if group in stability_test:
                    m = stability_test[group]
                    lines.append(f"  {group} (n={m.pair_count:,}):")
                    lines.append(f"    Precision: {m.precision:.4f}")
                    lines.append(f"    Recall:    {m.recall:.4f}")
                    lines.append(f"    F1:        {m.f1:.4f}")
            lines.append("")

        lines.append("=" * 60)

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    def _write_metrics_json(
        self,
        overall: Dict[str, float],
        per_timestep: Dict[str, Dict[int, float]],
        stability_train: Dict[str, StabilityMetrics],
        stability_test: Dict[str, StabilityMetrics],
        dataset_info: Dict[str, Any],
    ) -> Path:
        """Write structured JSON for downstream analysis."""
        path = self.output_dir / "metrics_structured.json"

        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "dataset": dataset_info,
            },
            "overall": overall,
            "per_timestep": {
                "f1": {str(k): v for k, v in per_timestep.get("f1", {}).items()},
                "auprc": {str(k): v for k, v in per_timestep.get("auprc", {}).items()},
            },
            "stability_by_training_freq": {
                k: v.to_dict() for k, v in stability_train.items()
            },
            "stability_by_test_freq": {
                k: v.to_dict() for k, v in stability_test.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path

    def _write_ground_truth_csv(
        self,
        ground_truth: List[Tuple[int, int, int, int]],
    ) -> Path:
        """Write ground truth interactions to CSV."""
        path = self.output_dir / "ground_truth.csv"

        records = []
        for e1, rel, e2, t in ground_truth:
            records.append(
                {
                    "entity1": e1,
                    "relation": rel,
                    "entity2": e2,
                    "timestep": t,
                    "pair": f"{min(e1, e2)}_{max(e1, e2)}",
                }
            )

        df = pd.DataFrame(records)
        df.to_csv(path, index=False)

        return path

    def _write_predictions_csv(
        self,
        predictions: List[Tuple[int, int, int, int, float, int]],
    ) -> Path:
        """Write model predictions to CSV."""
        path = self.output_dir / "prediction.csv"

        records = []
        for e1, rel, e2, t, score, pred in predictions:
            records.append(
                {
                    "entity1": e1,
                    "relation": rel,
                    "entity2": e2,
                    "timestep": t,
                    "score": score,
                    "predicted": pred,
                    "pair": f"{min(e1, e2)}_{max(e1, e2)}",
                }
            )

        df = pd.DataFrame(records)
        df.to_csv(path, index=False)

        return path
