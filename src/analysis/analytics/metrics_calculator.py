"""Metrics calculation for analysis results."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from src.core.utils import get_logger
from ..data.data_processor import ProcessedData


logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    TP: int
    FP: int
    FN: int
    TN: int
    Recall: float
    Precision: float
    TPR: float
    FPR: float
    F1: float
    MCC: float


@dataclass
class StabilityBinDetail(PerformanceMetrics):
    """Container for performance metrics for a stability bin, including pair count."""
    pair_count: int
    mean_pairwise_f1: float
    baseline_f1: Optional[float] = None
    baseline_mean_pairwise_f1: Optional[float] = None


@dataclass
class MetricsReport:
    """Container for comprehensive metrics analysis."""
    
    baseline_metrics: Optional[PerformanceMetrics]
    model_metrics: PerformanceMetrics
    metrics_by_stability: Dict[str, StabilityBinDetail]
    metrics_by_test_stability: Dict[str, StabilityBinDetail]
    metrics_over_time: pd.DataFrame
    cumulative_errors: pd.DataFrame
    mean_pairwise_f1: float
    baseline_mean_pairwise_f1: Optional[float]


class MetricsCalculator:
    """Calculates various performance metrics and analyses."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = get_logger(__name__)
    
    def calculate_comprehensive_metrics(
        self,
        processed_data: ProcessedData
    ) -> MetricsReport:
        """Calculate comprehensive metrics report.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            Complete metrics report
        """
        self.logger.info("Calculating comprehensive metrics")
        
        try:
            # Prepare evaluation series
            gt_eval, pred_eval, baseline_eval = self._prepare_evaluation_series(processed_data)
            
            # Define all pairs and timestamps for complete indices
            # For MODEL metrics, only include pairs from GT and MODEL predictions
            model_pairs = gt_eval.index.get_level_values(0).unique()
            if not pred_eval.empty:
                model_pairs = model_pairs.union(pred_eval.index.get_level_values(0).unique())
            
            # Use separate complete indices for model and baseline to avoid excluding pairs
            model_complete_index = pd.MultiIndex.from_product(
                [model_pairs, processed_data.test_timestamps],
                names=['pair', 'time_stamp']
            )
            
            # Calculate baseline metrics
            baseline_metrics = None
            if baseline_eval is not None and not baseline_eval.empty:
                # Build complete index that covers any pairs present in baseline
                baseline_pairs = baseline_eval.index.get_level_values(0).unique().union(
                    gt_eval.index.get_level_values(0).unique()
                )
                baseline_complete_index = pd.MultiIndex.from_product(
                    [baseline_pairs, processed_data.test_timestamps],
                    names=['pair', 'time_stamp']
                )
                baseline_metrics = self._calculate_metrics(
                    gt_eval, baseline_eval, baseline_complete_index
                )
            
            # Calculate model metrics
            model_metrics = self._calculate_metrics(
                gt_eval, pred_eval, model_complete_index
            )
            
            # Calculate metrics by stability
            metrics_by_stability = self._calculate_metrics_by_stability(
                gt_eval, pred_eval, processed_data
            )
            # Calculate metrics by test-based stability
            metrics_by_test_stability = self._calculate_metrics_by_test_stability(
                gt_eval, pred_eval, processed_data
            )
            
            # Calculate metrics over time using the model pairs scope
            metrics_over_time = self._calculate_metrics_over_time(
                gt_eval, pred_eval, processed_data, model_pairs
            )
            
            # Calculate cumulative errors
            cumulative_errors = self._calculate_cumulative_errors(
                gt_eval, pred_eval, processed_data.test_timestamps
            )
            
            # Calculate Mean Pairwise F1 Score
            mean_pairwise_f1 = self.calculate_mean_pairwise_f1(processed_data)
            
            # Calculate baseline Mean Pairwise F1 Score
            baseline_mean_pairwise_f1 = None
            if not processed_data.baseline_full.empty:
                baseline_mean_pairwise_f1 = self.calculate_mean_pairwise_f1_for_baseline(processed_data)
            
            report = MetricsReport(
                baseline_metrics=baseline_metrics,
                model_metrics=model_metrics,
                metrics_by_stability=metrics_by_stability,
                metrics_by_test_stability=metrics_by_test_stability,
                metrics_over_time=metrics_over_time,
                cumulative_errors=cumulative_errors,
                mean_pairwise_f1=mean_pairwise_f1,
                baseline_mean_pairwise_f1=baseline_mean_pairwise_f1
            )
            
            self.logger.info("Comprehensive metrics calculation completed")
            return report
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            raise
    
    def _prepare_evaluation_series(
        self,
        processed_data: ProcessedData
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """Prepare evaluation series for metrics calculation.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            Tuple of (ground_truth_series, predictions_series, baseline_series)
        """
        # Create evaluation series with pair-time index
        gt_eval = processed_data.ground_truth_full.set_index(['pair', 'time_stamp'])['present']
        pred_eval = processed_data.predictions_full.set_index(['pair', 'time_stamp'])['present']
        
        baseline_eval = None
        if not processed_data.baseline_full.empty:
            baseline_eval = processed_data.baseline_full.set_index(['pair', 'time_stamp'])['present']
        
        return gt_eval, pred_eval, baseline_eval
    
    def _calculate_metrics(
        self,
        gt_series: pd.Series,
        pred_series: pd.Series,
        complete_index: pd.MultiIndex
    ) -> PerformanceMetrics:
        """Calculate standard performance metrics.
        
        Args:
            gt_series: Ground truth series
            pred_series: Predictions series
            complete_index: A MultiIndex covering all (pair, time_stamp) combinations for the scope
            
        Returns:
            PerformanceMetrics object
        """
        # Reindex series to the complete_index, filling missing values with 0 (negative)
        gt_reindexed = gt_series.reindex(complete_index, fill_value=0)
        pred_reindexed = pred_series.reindex(complete_index, fill_value=0)
        
        # Calculate confusion matrix components
        TP = ((pred_reindexed == 1) & (gt_reindexed == 1)).sum()
        FP = ((pred_reindexed == 1) & (gt_reindexed == 0)).sum()
        FN = ((pred_reindexed == 0) & (gt_reindexed == 1)).sum()
        TN = ((pred_reindexed == 0) & (gt_reindexed == 0)).sum()

        # Sanity check (optional, can be logged)
        # total_calculated = TP + FP + FN + TN
        # if total_calculated != len(complete_index):
        #     self.logger.warning(
        #         f"Metric calculation sanity check failed: "
        #         f"TP+FP+FN+TN ({total_calculated}) != len(complete_index) ({len(complete_index)})"
        #     )
        
        # Calculate metrics
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        TPR = Recall
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        F1 = 2 * ((Precision * Recall) / (Precision + Recall)) if (Precision + Recall) > 0 else 0
        
        # Calculate MCC
        mcc_denom = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** (1/2)
        MCC = (TP * TN - FP * FN) / mcc_denom if mcc_denom > 0 else 0
        
        return PerformanceMetrics(
            TP=int(TP), FP=int(FP), FN=int(FN), TN=int(TN),
            Recall=float(Recall), Precision=float(Precision),
            TPR=float(TPR), FPR=float(FPR), F1=float(F1), MCC=float(MCC)
        )
    
    def _calculate_metrics_by_stability(
        self,
        gt_eval: pd.Series,
        pred_eval: pd.Series,
        processed_data: ProcessedData
    ) -> Dict[str, StabilityBinDetail]:
        """Calculate metrics grouped by stability bins.
        
        Args:
            gt_eval: Ground truth evaluation series
            pred_eval: Predictions evaluation series
            processed_data: Processed data with stability information
            
        Returns:
            Dictionary mapping stability labels to StabilityBinDetail objects
        """
        metrics_by_bin = {}
        
        if processed_data.stability_bins is None:
            self.logger.warning("No stability bins available, skipping stability analysis")
            return metrics_by_bin
            
        # Prepare baseline evaluation series
        baseline_eval = None
        if not processed_data.baseline_full.empty:
            baseline_eval = processed_data.baseline_full.set_index(['pair', 'time_stamp'])['present']
        
        # Get evaluation pairs from ground truth AND model predictions (to include FP-only pairs)
        eval_pairs_union = gt_eval.index.get_level_values(0).unique().union(
            pred_eval.index.get_level_values(0).unique()
        )
        
        # Align stability bins with evaluation pairs for MODEL metrics
        stability_bins_aligned_model = (
            processed_data.stability_bins
            .reindex(eval_pairs_union)
            .cat.add_categories('Undefined')
            .fillna('Undefined')
        )
        
        # Prepare baseline pairs union (GT ∪ BASELINE) if baseline exists
        baseline_pairs_union = None
        if not processed_data.baseline_full.empty:
            baseline_pairs_union = set(processed_data.baseline_full['pair'].unique())
            baseline_pairs_union = pd.Index(baseline_pairs_union).union(
                gt_eval.index.get_level_values(0).unique()
            )
        
        for bin_label in stability_bins_aligned_model.cat.categories:
            # MODEL: pairs_in_bin based on GT ∪ MODEL PRED pairs
            pairs_in_bin = stability_bins_aligned_model[stability_bins_aligned_model == bin_label].index
            pair_count_for_bin = len(pairs_in_bin)
            
            if pairs_in_bin.empty:
                # Still create an entry with zero metrics if needed, or skip
                # metrics_by_bin[bin_label] = StabilityBinDetail(TP=0,FP=0,FN=0,TN=0,Recall=0,Precision=0,TPR=0,FPR=0,F1=0,MCC=0, pair_count=0)
                continue
            
            # Filter evaluation series to include only pairs in current bin
            gt_bin_filtered = gt_eval[gt_eval.index.get_level_values(0).isin(pairs_in_bin)]
            pred_bin_filtered = pred_eval[pred_eval.index.get_level_values(0).isin(pairs_in_bin)]
            
            # Create complete index for this bin
            current_bin_complete_index = pd.MultiIndex.from_product(
                [pairs_in_bin, processed_data.test_timestamps], names=['pair', 'time_stamp']
            )
            
            if current_bin_complete_index.empty and (gt_bin_filtered.empty and pred_bin_filtered.empty):
                 # If no interactions possible and no data, create zero metrics
                 perf_metrics_obj = PerformanceMetrics(TP=0,FP=0,FN=0,TN=0,Recall=0,Precision=0,TPR=0,FPR=0,F1=0,MCC=0)
                 bin_mean_pairwise_f1 = 0.0
                 baseline_bin_f1 = 0.0
                 baseline_bin_mean_pairwise_f1 = 0.0
            elif current_bin_complete_index.empty and not (gt_bin_filtered.empty and pred_bin_filtered.empty):
                 self.logger.warning(f"Bin {bin_label} has data but complete index is empty. Skipping.")
                 continue
            else:
                perf_metrics_obj = self._calculate_metrics(
                    gt_bin_filtered, pred_bin_filtered, current_bin_complete_index
                )
                
                # Calculate Mean Pairwise F1 for this stability bin
                bin_mean_pairwise_f1 = self._calculate_mean_pairwise_f1_for_bin(
                    processed_data, pairs_in_bin, "model"
                )
                
                # Calculate baseline metrics for this stability bin if baseline data exists
                baseline_bin_f1 = None
                baseline_bin_mean_pairwise_f1 = None
                if not processed_data.baseline_full.empty:
                    # BASELINE: define its own pairs_in_bin set
                    if bin_label == 'Undefined':
                        # Pairs not present in training stability bins
                        train_pairs = processed_data.stability_bins.index
                        if baseline_pairs_union is not None:
                            baseline_pairs_in_bin = baseline_pairs_union.difference(train_pairs)
                        else:
                            baseline_pairs_in_bin = pd.Index([])
                    else:
                        # Pairs that belong to this bin in training, intersected with baseline evaluation scope
                        train_pairs_in_bin = processed_data.stability_bins[processed_data.stability_bins == bin_label].index
                        if baseline_pairs_union is not None:
                            baseline_pairs_in_bin = pd.Index(train_pairs_in_bin).intersection(baseline_pairs_union)
                        else:
                            baseline_pairs_in_bin = pd.Index(train_pairs_in_bin)

                    # Calculate baseline F1 for this bin
                    if len(baseline_pairs_in_bin) > 0:
                        baseline_bin_gt = gt_eval[gt_eval.index.get_level_values(0).isin(baseline_pairs_in_bin)]
                        baseline_bin_pred = baseline_eval[baseline_eval.index.get_level_values(0).isin(baseline_pairs_in_bin)]
                        baseline_bin_complete_index = pd.MultiIndex.from_product(
                            [baseline_pairs_in_bin, processed_data.test_timestamps], names=['pair', 'time_stamp']
                        )
                        
                        if not baseline_bin_complete_index.empty and not (baseline_bin_gt.empty and baseline_bin_pred.empty):
                            baseline_metrics = self._calculate_metrics(
                                baseline_bin_gt, baseline_bin_pred, baseline_bin_complete_index
                            )
                            baseline_bin_f1 = baseline_metrics.F1
                    
                    baseline_bin_mean_pairwise_f1 = self._calculate_mean_pairwise_f1_for_bin(
                        processed_data, baseline_pairs_in_bin, "baseline"
                    )

            metrics_by_bin[bin_label] = StabilityBinDetail(
                **vars(perf_metrics_obj), 
                pair_count=pair_count_for_bin,
                mean_pairwise_f1=bin_mean_pairwise_f1,
                baseline_f1=baseline_bin_f1,
                baseline_mean_pairwise_f1=baseline_bin_mean_pairwise_f1
            )
        
        return metrics_by_bin
    
    def _calculate_metrics_by_test_stability(
        self,
        gt_eval: pd.Series,
        pred_eval: pd.Series,
        processed_data: ProcessedData
    ) -> Dict[str, StabilityBinDetail]:
        """Calculate metrics grouped by test-based stability bins.
        
        Bins are derived from test set frequency of pairs.
        """
        metrics_by_bin = {}
        
        if processed_data.stability_bins_test is None:
            self.logger.warning("No test-based stability bins available, skipping test stability analysis")
            return metrics_by_bin
            
        # Prepare baseline evaluation series
        baseline_eval = None
        if not processed_data.baseline_full.empty:
            baseline_eval = processed_data.baseline_full.set_index(['pair', 'time_stamp'])['present']
        
        # Get evaluation pairs from ground truth AND model predictions (to include FP-only pairs)
        eval_pairs_union = gt_eval.index.get_level_values(0).unique().union(
            pred_eval.index.get_level_values(0).unique()
        )
        
        # Align test stability bins with evaluation pairs for MODEL metrics
        stability_bins_aligned_model = (
            processed_data.stability_bins_test
            .reindex(eval_pairs_union)
            .cat.add_categories('Undefined')
            .fillna('Undefined')
        )
        
        # Baseline scope
        baseline_pairs_union = None
        if not processed_data.baseline_full.empty:
            baseline_pairs_union = set(processed_data.baseline_full['pair'].unique())
            baseline_pairs_union = pd.Index(baseline_pairs_union).union(
                gt_eval.index.get_level_values(0).unique()
            )
        
        for bin_label in stability_bins_aligned_model.cat.categories:
            pairs_in_bin = stability_bins_aligned_model[stability_bins_aligned_model == bin_label].index
            pair_count_for_bin = len(pairs_in_bin)
            
            if pairs_in_bin.empty:
                continue
            
            gt_bin_filtered = gt_eval[gt_eval.index.get_level_values(0).isin(pairs_in_bin)]
            pred_bin_filtered = pred_eval[pred_eval.index.get_level_values(0).isin(pairs_in_bin)]
            
            current_bin_complete_index = pd.MultiIndex.from_product(
                [pairs_in_bin, processed_data.test_timestamps], names=['pair', 'time_stamp']
            )
            
            if current_bin_complete_index.empty and (gt_bin_filtered.empty and pred_bin_filtered.empty):
                perf_metrics_obj = PerformanceMetrics(TP=0,FP=0,FN=0,TN=0,Recall=0,Precision=0,TPR=0,FPR=0,F1=0,MCC=0)
                bin_mean_pairwise_f1 = 0.0
                baseline_bin_f1 = 0.0
                baseline_bin_mean_pairwise_f1 = 0.0
            elif current_bin_complete_index.empty and not (gt_bin_filtered.empty and pred_bin_filtered.empty):
                self.logger.warning(f"Test stability bin {bin_label} has data but complete index is empty. Skipping.")
                continue
            else:
                perf_metrics_obj = self._calculate_metrics(
                    gt_bin_filtered, pred_bin_filtered, current_bin_complete_index
                )
                
                bin_mean_pairwise_f1 = self._calculate_mean_pairwise_f1_for_bin(
                    processed_data, pairs_in_bin, "model"
                )
                
                # Calculate baseline metrics for this stability bin if baseline data exists
                baseline_bin_f1 = None
                baseline_bin_mean_pairwise_f1 = None
                if not processed_data.baseline_full.empty:
                    if bin_label == 'Undefined':
                        # Pairs not present in test stability bins
                        test_pairs = processed_data.stability_bins_test.index
                        if baseline_pairs_union is not None:
                            baseline_pairs_in_bin = baseline_pairs_union.difference(test_pairs)
                        else:
                            baseline_pairs_in_bin = pd.Index([])
                    else:
                        test_pairs_in_bin = processed_data.stability_bins_test[processed_data.stability_bins_test == bin_label].index
                        if baseline_pairs_union is not None:
                            baseline_pairs_in_bin = pd.Index(test_pairs_in_bin).intersection(baseline_pairs_union)
                        else:
                            baseline_pairs_in_bin = pd.Index(test_pairs_in_bin)
                    
                    # Calculate baseline F1 for this bin
                    if len(baseline_pairs_in_bin) > 0:
                        baseline_bin_gt = gt_eval[gt_eval.index.get_level_values(0).isin(baseline_pairs_in_bin)]
                        baseline_bin_pred = baseline_eval[baseline_eval.index.get_level_values(0).isin(baseline_pairs_in_bin)]
                        baseline_bin_complete_index = pd.MultiIndex.from_product(
                            [baseline_pairs_in_bin, processed_data.test_timestamps], names=['pair', 'time_stamp']
                        )
                        
                        if not baseline_bin_complete_index.empty and not (baseline_bin_gt.empty and baseline_bin_pred.empty):
                            baseline_metrics = self._calculate_metrics(
                                baseline_bin_gt, baseline_bin_pred, baseline_bin_complete_index
                            )
                            baseline_bin_f1 = baseline_metrics.F1
                    
                    baseline_bin_mean_pairwise_f1 = self._calculate_mean_pairwise_f1_for_bin(
                        processed_data, baseline_pairs_in_bin, "baseline"
                    )
            
            metrics_by_bin[bin_label] = StabilityBinDetail(
                **vars(perf_metrics_obj),
                pair_count=pair_count_for_bin,
                mean_pairwise_f1=bin_mean_pairwise_f1,
                baseline_f1=baseline_bin_f1,
                baseline_mean_pairwise_f1=baseline_bin_mean_pairwise_f1
            )
        
        return metrics_by_bin
    
    def _calculate_metrics_over_time(
        self,
        gt_eval: pd.Series,
        pred_eval: pd.Series,
        processed_data: ProcessedData,
        all_pairs_for_scope: pd.Index
    ) -> pd.DataFrame:
        """Calculate metrics cumulatively over time.
        
        Args:
            gt_eval: Ground truth evaluation series
            pred_eval: Predictions evaluation series
            processed_data: Processed data with timestamps
            all_pairs_for_scope: All unique pairs relevant to this gt_eval/pred_eval scope
            
        Returns:
            DataFrame with metrics over time
        """
        metrics_over_time = defaultdict(list)
        timestamps_sorted = sorted(processed_data.test_timestamps)
        
        # Slices will be taken from original gt_eval and pred_eval
        # These series might be sparse. Reindexing will happen in _calculate_metrics.
        
        # Get time values from the full index of gt_eval (or pred_eval, assuming they cover similar time ranges)
        # This is to ensure the mask 'times <= t' works on a comprehensive time series.
        # However, gt_eval can be sparse. Using processed_data.test_timestamps is more robust for iteration.

        # gt_eval and pred_eval are passed directly.
        # Filtering by time will be done on these (potentially sparse) series.
        # The complete_index for each _calculate_metrics call will handle filling gaps.

        for t_idx, t in enumerate(timestamps_sorted):
            # Filter data up to current time t
            # Need to handle MultiIndex filtering carefully.
            # We are filtering the original gt_eval and pred_eval.
            current_timestamps = timestamps_sorted[:t_idx + 1]

            # Create complete index for this cumulative step
            # The pairs are all_pairs_for_scope (e.g. all pairs seen in gt_eval or pred_eval globally)
            # The times are timestamps up to current t
            cumulative_complete_index = pd.MultiIndex.from_product(
                [all_pairs_for_scope, current_timestamps],
                names=['pair', 'time_stamp']
            )
            
            if cumulative_complete_index.empty:
                # If no interactions possible yet (e.g., t_idx= -1 or no pairs), create zero metrics
                metrics = PerformanceMetrics(TP=0,FP=0,FN=0,TN=0,Recall=0,Precision=0,TPR=0,FPR=0,F1=0,MCC=0)
            else:
                # gt_eval and pred_eval are passed as is. _calculate_metrics will reindex them
                # using cumulative_complete_index. This means gt_eval and pred_eval should contain
                # all data, and _calculate_metrics will effectively select and densify relevant parts.
                 metrics = self._calculate_metrics(
                    gt_eval, pred_eval, cumulative_complete_index
                )
            
            # Store metrics
            metrics_over_time['Time'].append(t)
            metrics_over_time['Recall'].append(metrics.Recall)
            metrics_over_time['Precision'].append(metrics.Precision)
            metrics_over_time['F1'].append(metrics.F1)
            metrics_over_time['MCC'].append(metrics.MCC)
            metrics_over_time['TPR'].append(metrics.TPR)
            metrics_over_time['FPR'].append(metrics.FPR)
        
        return pd.DataFrame(metrics_over_time)
    
    def _calculate_cumulative_errors(
        self,
        gt_eval: pd.Series,
        pred_eval: pd.Series,
        timestamps: List[int]
    ) -> pd.DataFrame:
        """Calculate cumulative errors over time.
        
        Args:
            gt_eval: Ground truth evaluation series
            pred_eval: Predictions evaluation series
            timestamps: List of timestamps
            
        Returns:
            DataFrame with cumulative error information
        """
        errors_over_time = defaultdict(list)
        timestamps_sorted = sorted(timestamps)
        
        # Align indices
        common_index = gt_eval.index.intersection(pred_eval.index)
        gt_aligned = gt_eval[common_index]
        pred_aligned = pred_eval[common_index]
        
        # Get time values from index
        times = gt_aligned.index.get_level_values(1)
        
        cumulative_fp = 0
        cumulative_fn = 0
        
        for t in timestamps_sorted:
            # Filter data at current time
            mask_t = times == t
            gt_t = gt_aligned[mask_t]
            pred_t = pred_aligned[mask_t]
            
            fp_t = ((pred_t == 1) & (gt_t == 0)).sum()
            fn_t = ((pred_t == 0) & (gt_t == 1)).sum()
            
            cumulative_fp += fp_t
            cumulative_fn += fn_t
            
            errors_over_time['Time'].append(t)
            errors_over_time['Cumulative_FP'].append(cumulative_fp)
            errors_over_time['Cumulative_FN'].append(cumulative_fn)
            errors_over_time['Cumulative_Errors'].append(cumulative_fp + cumulative_fn)
        
        return pd.DataFrame(errors_over_time)
    
    def calculate_per_edge_f1(
        self,
        processed_data: ProcessedData
    ) -> Optional[pd.DataFrame]:
        """Calculate F1 score for each edge (pair).
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            DataFrame with per-edge F1 scores or None if calculation fails
        """
        try:
            # Use the generalized method for consistency
            f1_df = self._calculate_per_edge_f1_generic(
                processed_data.ground_truth_full,
                processed_data.predictions_full,
                "pred"
            )
            
            if f1_df is not None:
                self.logger.info(f"Calculated F1 scores for {len(f1_df)} pairs")
            else:
                self.logger.warning("No per-edge F1 scores calculated")
            
            return f1_df
            
        except Exception as e:
            self.logger.error(f"Failed to calculate per-edge F1: {e}")
            return None
    
    def calculate_mean_pairwise_f1(
        self,
        processed_data: ProcessedData
    ) -> float:
        """Calculate Mean Pairwise F1 Score.
        
        This metric calculates F1 scores for all interactions over time for each pair,
        then averages these F1 scores across all pairs.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            Mean Pairwise F1 Score (float between 0 and 1)
        """
        try:
            self.logger.info("Calculating Mean Pairwise F1 Score")
            
            # Use the generalized method for consistency
            f1_df = self._calculate_per_edge_f1_generic(
                processed_data.ground_truth_full,
                processed_data.predictions_full,
                "pred"
            )
            
            if f1_df is None or f1_df.empty:
                self.logger.warning("No per-edge F1 scores available for Mean Pairwise F1 calculation")
                return 0.0
            
            # Calculate mean of all pair F1 scores
            mean_pairwise_f1 = f1_df['F1'].mean()
            
            self.logger.info(f"Mean Pairwise F1 Score: {mean_pairwise_f1:.4f} (averaged over {len(f1_df)} pairs)")
            
            return float(mean_pairwise_f1)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Mean Pairwise F1: {e}")
            return 0.0
    
    def calculate_mean_pairwise_f1_for_baseline(
        self,
        processed_data: ProcessedData
    ) -> float:
        """Calculate Mean Pairwise F1 Score for baseline predictions.
        
        Args:
            processed_data: Processed analysis data
            
        Returns:
            Baseline Mean Pairwise F1 Score (float between 0 and 1)
        """
        try:
            self.logger.info("Calculating Baseline Mean Pairwise F1 Score")
            
            if processed_data.baseline_full.empty:
                self.logger.warning("No baseline data available for Mean Pairwise F1 calculation")
                return 0.0
            
            # Use the generalized method for consistency
            f1_df = self._calculate_per_edge_f1_generic(
                processed_data.ground_truth_full,
                processed_data.baseline_full,
                "baseline"
            )
            
            if f1_df is None or f1_df.empty:
                self.logger.warning("No per-edge F1 scores available for Baseline Mean Pairwise F1 calculation")
                return 0.0
            
            baseline_mean_pairwise_f1 = f1_df['F1'].mean()
            
            self.logger.info(f"Baseline Mean Pairwise F1 Score: {baseline_mean_pairwise_f1:.4f} (averaged over {len(f1_df)} pairs)")
            
            return float(baseline_mean_pairwise_f1)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Baseline Mean Pairwise F1: {e}")
            return 0.0
    
    def _calculate_per_edge_f1_generic(
        self,
        ground_truth_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        prediction_type: str = "pred",
        pairs_filter: Optional[pd.Index] = None
    ) -> Optional[pd.DataFrame]:
        """Generic method to calculate F1 score for each edge (pair).
        
        Args:
            ground_truth_df: Ground truth dataframe
            predictions_df: Predictions dataframe
            prediction_type: Suffix to use for prediction columns
            pairs_filter: Optional filter to limit to specific pairs
            
        Returns:
            DataFrame with per-edge F1 scores or None if calculation fails
        """
        try:
            # Merge ground truth and predictions
            # Build full grids to ensure negatives are included, then outer-join
            gt_full = ground_truth_df.add_suffix('_gt')
            pred_full = predictions_df.add_suffix(f'_{prediction_type}')
            merged = pd.merge(
                gt_full,
                pred_full,
                left_on=['pair_gt', 'time_stamp_gt'],
                right_on=[f'pair_{prediction_type}', f'time_stamp_{prediction_type}'],
                how='outer'
            )
            
            if merged.empty:
                return None
            
            # Fill missing pair id and presents before filtering so prediction-only pairs are kept
            merged['pair_gt'] = merged['pair_gt'].fillna(merged[f'pair_{prediction_type}'])
            merged['present_gt'] = merged['present_gt'].fillna(0).astype(int)
            merged[f'present_{prediction_type}'] = merged[f'present_{prediction_type}'].fillna(0).astype(int)

            # Filter to specific pairs if requested
            if pairs_filter is not None and not pairs_filter.empty:
                merged = merged[merged['pair_gt'].isin(pairs_filter)]
                if merged.empty:
                    return None

            per_edge_stats = []
            for pair, group in merged.groupby('pair_gt'):
                gt = group['present_gt']
                pred = group[f'present_{prediction_type}']
                
                TP = ((pred == 1) & (gt == 1)).sum()
                FP = ((pred == 1) & (gt == 0)).sum()
                FN = ((pred == 0) & (gt == 1)).sum()
                
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                per_edge_stats.append({'pair': pair, 'F1': f1})
            
            if not per_edge_stats:
                return None
            
            return pd.DataFrame(per_edge_stats).set_index('pair')
            
        except Exception as e:
            self.logger.error(f"Failed to calculate per-edge F1 ({prediction_type}): {e}")
            return None
    
    def _calculate_mean_pairwise_f1_for_bin(
        self,
        processed_data: ProcessedData,
        pairs_in_bin: pd.Index,
        prediction_type: str = "model"
    ) -> float:
        """Calculate Mean Pairwise F1 Score for a specific set of pairs (stability bin).
        
        Args:
            processed_data: Processed analysis data
            pairs_in_bin: Index of pairs to calculate for
            prediction_type: Either "model" or "baseline"
            
        Returns:
            Mean Pairwise F1 Score for the given pairs
        """
        try:
            if pairs_in_bin.empty:
                return 0.0
            
            # Select appropriate prediction data and suffix
            if prediction_type == "baseline":
                pred_data = processed_data.baseline_full
                suffix = "baseline"
            else:
                pred_data = processed_data.predictions_full
                suffix = "pred"
            
            # Use the generalized method for consistency
            f1_df = self._calculate_per_edge_f1_generic(
                processed_data.ground_truth_full,
                pred_data,
                suffix,
                pairs_in_bin
            )
            
            if f1_df is None or f1_df.empty:
                return 0.0
            
            bin_mean_pairwise_f1 = f1_df['F1'].mean()
            
            return float(bin_mean_pairwise_f1)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Mean Pairwise F1 for bin: {e}")
            return 0.0
    
    def write_metrics_report(
        self,
        report: MetricsReport,
        output_file_path: str
    ) -> None:
        """Write comprehensive metrics report to file.
        
        Args:
            report: Metrics report to write
            output_file_path: Path to output file
        """
        try:
            with open(output_file_path, "w") as f:
                f.write("=== RAPID Performance Metrics Report ===\n\n")
                
                # Baseline metrics
                if report.baseline_metrics:
                    f.write("--- BASELINE Performance ---\n")
                    self._write_metrics_to_file(f, report.baseline_metrics, "BASELINE")
                    if report.baseline_mean_pairwise_f1 is not None:
                        f.write(f"Mean Pairwise F1 Score: {report.baseline_mean_pairwise_f1:.4f}\n")
                    f.write("\n")
                else:
                    f.write("--- BASELINE Performance ---\n")
                    f.write("No baseline predictions available.\n\n")
                
                # Model metrics
                f.write("--- MODEL Performance (ALL Interactions) ---\n")
                self._write_metrics_to_file(f, report.model_metrics, "MODEL")
                f.write(f"Mean Pairwise F1 Score: {report.mean_pairwise_f1:.4f}\n")
                f.write("\n")
                
                # Stability-based metrics
                if report.metrics_by_stability:
                    f.write("--- Performance by Interaction Stability ---\n")
                    for stability_label, stability_detail in report.metrics_by_stability.items():
                        output_label = stability_label.replace("Moderate", "Transient")
                        if stability_label == "Undefined":
                            output_label = "Not in Train"
                        
                        f.write(f"\nMetrics for {output_label} interactions (N={stability_detail.pair_count} pairs):\n")
                        self._write_metrics_to_file(f, stability_detail, output_label)
                        f.write(f"Mean Pairwise F1 Score: {stability_detail.mean_pairwise_f1:.4f}\n")
                        if stability_detail.baseline_mean_pairwise_f1 is not None:
                            f.write(f"Baseline Mean Pairwise F1 Score: {stability_detail.baseline_mean_pairwise_f1:.4f}\n")
                    f.write("\n")
            
            self.logger.info(f"Metrics report written to {output_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write metrics report: {e}")

    def write_structured_metrics(
        self,
        report: MetricsReport,
        output_file_path: str,
        experiment_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Write structured metrics to JSON file for easy parsing.
        
        Args:
            report: MetricsReport to write
            output_file_path: Path to write the structured JSON file
            experiment_metadata: Optional metadata about the experiment
        """
        import json
        from datetime import datetime
        
        try:
            self.logger.info(f"Writing structured metrics to {output_file_path}")
            
            # Helper function to convert PerformanceMetrics to dict
            def metrics_to_dict(metrics: PerformanceMetrics) -> Dict[str, Any]:
                return {
                    "recall": float(metrics.Recall),
                    "precision": float(metrics.Precision),
                    "f1": float(metrics.F1),
                    "mcc": float(metrics.MCC),
                    "tpr": float(metrics.TPR),
                    "fpr": float(metrics.FPR),
                    "tp": int(metrics.TP),
                    "fp": int(metrics.FP),
                    "fn": int(metrics.FN),
                    "tn": int(metrics.TN)
                }
            
            # Helper function to convert StabilityBinDetail to dict
            def stability_detail_to_dict(detail: StabilityBinDetail) -> Dict[str, Any]:
                result = metrics_to_dict(detail)
                result.update({
                    "pair_count": int(detail.pair_count),
                    "mean_pairwise_f1": float(detail.mean_pairwise_f1),
                    "baseline_f1": float(detail.baseline_f1) if detail.baseline_f1 is not None else None,
                    "baseline_mean_pairwise_f1": float(detail.baseline_mean_pairwise_f1) if detail.baseline_mean_pairwise_f1 is not None else None
                })
                return result
            
            # Build structured data
            structured_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "format_version": "1.0",
                    **(experiment_metadata or {})
                },
                "overall_metrics": {
                    "model": metrics_to_dict(report.model_metrics),
                    "baseline": metrics_to_dict(report.baseline_metrics) if report.baseline_metrics else None,
                    "mean_pairwise_f1": {
                        "model": float(report.mean_pairwise_f1),
                        "baseline": float(report.baseline_mean_pairwise_f1) if report.baseline_mean_pairwise_f1 is not None else None
                    }
                },
                "stability_metrics": {
                    "training_frequency": {
                        bin_label: stability_detail_to_dict(detail)
                        for bin_label, detail in report.metrics_by_stability.items()
                    } if report.metrics_by_stability else {},
                    "test_frequency": {
                        bin_label: stability_detail_to_dict(detail)
                        for bin_label, detail in report.metrics_by_test_stability.items()
                    } if report.metrics_by_test_stability else {}
                }
            }
            
            # Add time series data if available
            if hasattr(report, 'metrics_over_time') and not report.metrics_over_time.empty:
                # Convert DataFrame to list of dicts for JSON serialization
                time_series_data = []
                for _, row in report.metrics_over_time.iterrows():
                    time_point = {}
                    for col in report.metrics_over_time.columns:
                        value = row[col]
                        # Handle different data types
                        if pd.isna(value):
                            time_point[col.lower()] = None
                        elif isinstance(value, (int, float)):
                            time_point[col.lower()] = float(value)
                        else:
                            time_point[col.lower()] = str(value)
                    time_series_data.append(time_point)
                
                structured_data["time_series_metrics"] = time_series_data
            
            # Write JSON file with proper formatting
            with open(output_file_path, 'w') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Structured metrics written to {output_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write structured metrics: {e}")
    
    def _write_metrics_to_file(
        self,
        file_handle,
        metrics: PerformanceMetrics,
        label: str
    ) -> None:
        """Write metrics to file handle.
        
        Args:
            file_handle: Open file handle
            metrics: Metrics to write
            label: Label for the metrics
        """
        file_handle.write(
            f"Recall: {metrics.Recall:.4f}, "
            f"Precision: {metrics.Precision:.4f}, "
            f"TPR: {metrics.TPR:.4f}, "
            f"FPR: {metrics.FPR:.4f}, "
            f"F1: {metrics.F1:.4f}, "
            f"MCC: {metrics.MCC:.4f}\n"
        ) 
