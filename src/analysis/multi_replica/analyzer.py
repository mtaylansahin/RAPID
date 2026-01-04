"""Multi-replica analyzer for aggregating results across replicas."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.analysis.data_structures import (
    AggregatedMetrics,
    MetricStats,
    OverallStats,
    StabilityGroupStats,
    StabilityMetrics,
)


class MultiReplicaAnalyzer:
    """Discovers and aggregates metrics from multiple replica runs."""

    def __init__(self):
        self.replica_metrics: List[Dict] = []
        self.replica_paths: List[str] = []

    def discover_replica_directories(
        self,
        results_dir: Path,
        pattern: str = "replica*/analysis",
    ) -> List[Path]:
        """
        Discover replica analysis directories matching pattern.

        Args:
            results_dir: Base directory containing replica subdirectories
            pattern: Glob pattern to find analysis directories

        Returns:
            List of discovered analysis directory paths
        """
        results_dir = Path(results_dir)
        if not results_dir.exists():
            print(f"Warning: Results directory does not exist: {results_dir}")
            return []

        paths = sorted(results_dir.glob(pattern))
        print(f"Discovered {len(paths)} replica directories")
        return paths

    def load_replica_metrics(self, replica_paths: List[Path]) -> bool:
        """
        Load metrics from multiple replica directories.

        Expects each directory to contain 'metrics_structured.json'.

        Args:
            replica_paths: List of paths to replica analysis directories

        Returns:
            True if at least one replica loaded successfully
        """
        self.replica_metrics = []
        self.replica_paths = []

        for path in replica_paths:
            path = Path(path)
            metrics_file = path / "metrics_structured.json"

            if not metrics_file.exists():
                print(f"Warning: No metrics file found in {path}")
                continue

            try:
                with open(metrics_file) as f:
                    data = json.load(f)
                self.replica_metrics.append(data)
                self.replica_paths.append(str(path))
                print(f"Loaded metrics from {path}")
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")

        return len(self.replica_metrics) > 0

    def aggregate_metrics(
        self,
        experiment_name: Optional[str] = None,
    ) -> Optional[AggregatedMetrics]:
        """
        Aggregate metrics across all loaded replicas.

        Returns:
            AggregatedMetrics with mean/std/min/max across replicas
        """
        if not self.replica_metrics:
            print("No replica metrics loaded")
            return None

        if experiment_name is None:
            experiment_name = Path(self.replica_paths[0]).parent.name

        # Aggregate overall metrics
        overall_stats = self._aggregate_overall()

        # Aggregate stability metrics
        train_freq_stats = self._aggregate_stability("stability_by_training_freq")
        test_freq_stats = self._aggregate_stability("stability_by_test_freq")

        return AggregatedMetrics(
            experiment_name=experiment_name,
            n_replicas=len(self.replica_metrics),
            replica_paths=self.replica_paths,
            overall_stats=overall_stats,
            training_frequency_stats=train_freq_stats,
            test_frequency_stats=test_freq_stats,
        )

    def _aggregate_overall(self) -> OverallStats:
        """Aggregate overall metrics across replicas."""
        metric_names = ["auroc", "auprc", "precision", "recall", "f1", "mcc"]
        values = {name: [] for name in metric_names}

        for data in self.replica_metrics:
            overall = data.get("overall", {})
            for name in metric_names:
                if name in overall:
                    values[name].append(overall[name])

        return OverallStats(
            n_replicas=len(self.replica_metrics),
            auroc=MetricStats.from_values(values["auroc"]),
            auprc=MetricStats.from_values(values["auprc"]),
            precision=MetricStats.from_values(values["precision"]),
            recall=MetricStats.from_values(values["recall"]),
            f1=MetricStats.from_values(values["f1"]),
            mcc=MetricStats.from_values(values["mcc"]),
        )

    def _aggregate_stability(
        self,
        key: str,
    ) -> Dict[str, StabilityGroupStats]:
        """Aggregate stability-stratified metrics across replicas."""
        # Collect values per group
        group_values: Dict[str, Dict[str, List[float]]] = {}

        for data in self.replica_metrics:
            stability_data = data.get(key, {})
            for group_name, group_metrics in stability_data.items():
                if group_name not in group_values:
                    group_values[group_name] = {
                        "recall": [],
                        "precision": [],
                        "f1": [],
                        "mcc": [],
                        "pair_count": [],
                    }

                for metric in ["recall", "precision", "f1", "mcc", "pair_count"]:
                    if metric in group_metrics:
                        group_values[group_name][metric].append(group_metrics[metric])

        # Convert to StabilityGroupStats
        result = {}
        for group_name, values in group_values.items():
            result[group_name] = StabilityGroupStats(
                group_name=group_name,
                n_replicas=len(self.replica_metrics),
                recall=MetricStats.from_values(values["recall"]),
                precision=MetricStats.from_values(values["precision"]),
                f1=MetricStats.from_values(values["f1"]),
                mcc=MetricStats.from_values(values["mcc"]),
                pair_count=MetricStats.from_values(values["pair_count"]),
            )

        return result

    def get_summary(self) -> Dict:
        """Get summary of loaded replicas."""
        return {
            "total_replicas": len(self.replica_metrics),
            "replica_paths": self.replica_paths,
        }
