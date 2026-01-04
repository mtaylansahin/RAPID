#!/usr/bin/env python3
"""Command-line interface for multi-replica analysis."""

import sys
from pathlib import Path

import argparse
import json
from typing import List

from src.core.utils import setup_logging, get_logger
from src.analysis.multi_replica.analyzer import MultiReplicaAnalyzer
from src.analysis.multi_replica.plotter import MultiReplicaPlotter


def main():
    """Main CLI function for multi-replica analysis."""
    
    parser = argparse.ArgumentParser(
        description="Multi-replica analysis for RAPID experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover replica directories
  python -m src.analysis.multi_replica.cli --results_dir /path/to/results
  
  # Specify explicit paths
  python -m src.analysis.multi_replica.cli --replica_paths path1/analysis path2/analysis path3/analysis
  
  # Custom output and pattern
  python -m src.analysis.multi_replica.cli --results_dir /path/to/results --pattern "replica*/analysis" --output multi_replica_plots
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--results_dir', 
        type=str,
        help='Base directory to search for replica analysis directories'
    )
    input_group.add_argument(
        '--replica_paths', 
        nargs='+', 
        type=str,
        help='Explicit list of replica analysis directory paths'
    )
    
    # Configuration options
    parser.add_argument(
        '--pattern', 
        type=str, 
        default='*/analysis',
        help='Pattern to find analysis directories (default: */analysis)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='multi_replica_analysis',
        help='Output directory for plots and results (default: multi_replica_analysis)'
    )
    parser.add_argument(
        '--export_json', 
        type=str,
        help='Export aggregated metrics to JSON file'
    )
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting multi-replica analysis")
        
        # Initialize analyzer
        analyzer = MultiReplicaAnalyzer()
        
        # Get replica paths
        if args.replica_paths:
            replica_paths = args.replica_paths
            logger.info(f"Using {len(replica_paths)} explicitly provided replica paths")
        else:
            replica_paths = analyzer.discover_replica_directories(args.results_dir, args.pattern)
            logger.info(f"Discovered {len(replica_paths)} replica directories")
        
        if not replica_paths:
            logger.error("No replica directories found or specified")
            return 1
        
        # Load replica metrics
        success = analyzer.load_replica_metrics(replica_paths)
        if not success:
            logger.error("Failed to load any replica metrics")
            return 1
        
        # Print loading summary
        summary = analyzer.get_replica_summary()
        logger.info(f"Loaded {summary['successful_loads']}/{summary['total_replicas']} replicas successfully")
        
        if summary['failed_loads'] > 0:
            logger.warning(f"{summary['failed_loads']} replicas failed to load:")
            for detail in summary['replica_details']:
                if not detail['success']:
                    logger.warning(f"  {detail['path']}: {detail['error']}")
        
        # Aggregate metrics
        aggregated_metrics = analyzer.aggregate_metrics()
        if aggregated_metrics is None:
            logger.error("Failed to aggregate metrics")
            return 1
        
        logger.info(f"Successfully aggregated metrics across {aggregated_metrics.n_replicas} replicas")
        
        # Print summary
        print_metrics_summary(aggregated_metrics)
        
        # Export to JSON if requested
        if args.export_json:
            export_aggregated_metrics(aggregated_metrics, args.export_json)
            logger.info(f"Exported aggregated metrics to {args.export_json}")
        
        # Generate plots
        plotter = MultiReplicaPlotter(args.output)
        generated_plots = plotter.generate_all_plots(aggregated_metrics)
        
        if generated_plots:
            logger.info(f"Generated {len(generated_plots)} plots in {args.output}/:")
            for plot_path in generated_plots:
                plot_name = Path(plot_path).name
                logger.info(f"  {plot_name}")
        else:
            logger.warning("No plots were generated")
        
        logger.info("Multi-replica analysis completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Multi-replica analysis failed: {e}", exc_info=True)
        return 1


def print_metrics_summary(aggregated_metrics):
    """Print summary of aggregated metrics to console."""
    
    print("\n" + "="*60)
    print(f"MULTI-REPLICA SUMMARY ({aggregated_metrics.n_replicas} replicas)")
    print("="*60)
    
    # Overall metrics
    overall = aggregated_metrics.overall_stats
    print("\n📊 Overall Model Performance:")
    print(f"  F1 Score: {overall.model_f1.mean:.4f} ± {overall.model_f1.std:.4f}")
    print(f"  Recall:   {overall.model_recall.mean:.4f} ± {overall.model_recall.std:.4f}")
    print(f"  MCC:      {overall.model_mcc.mean:.4f} ± {overall.model_mcc.std:.4f}")
    
    # Stability metrics summary
    training_stats = aggregated_metrics.training_frequency_stats
    if training_stats:
        print("\n📈 Stability Performance (Training Freq.):")
        for group_name in ['Rare (<5%)', 'Moderate (5-50%)', 'Stable (>50%)']:
            if group_name in training_stats:
                stats = training_stats[group_name]
                clean_name = group_name.replace('Moderate', 'Transient')
                print(f"  {clean_name:15}: F1 = {stats.f1.mean:.4f} ± {stats.f1.std:.4f}")
    
    print("\n" + "="*60 + "\n")


def export_aggregated_metrics(aggregated_metrics, output_path: str):
    """Export aggregated metrics to JSON file."""
    
    def metric_stats_to_dict(stats):
        """Convert MetricStats to dictionary."""
        return {
            'mean': stats.mean,
            'std': stats.std,
            'min': stats.min,
            'max': stats.max,
            'n_replicas': stats.n_replicas
        }
    
    def stability_stats_to_dict(stats):
        """Convert StabilityGroupStats to dictionary."""
        result = {
            'recall': metric_stats_to_dict(stats.recall),
            'precision': metric_stats_to_dict(stats.precision),
            'f1': metric_stats_to_dict(stats.f1),
            'mcc': metric_stats_to_dict(stats.mcc),
            'tpr': metric_stats_to_dict(stats.tpr),
            'fpr': metric_stats_to_dict(stats.fpr),
            'mean_pairwise_f1': metric_stats_to_dict(stats.mean_pairwise_f1),
            'pair_count': metric_stats_to_dict(stats.pair_count),
            'group_name': stats.group_name,
            'n_replicas': stats.n_replicas
        }
        
        if stats.baseline_f1:
            result['baseline_f1'] = metric_stats_to_dict(stats.baseline_f1)
        if stats.baseline_mean_pairwise_f1:
            result['baseline_mean_pairwise_f1'] = metric_stats_to_dict(stats.baseline_mean_pairwise_f1)
        
        return result
    
    # Build export data
    export_data = {
        'metadata': {
            'experiment_name': aggregated_metrics.experiment_name,
            'n_replicas': aggregated_metrics.n_replicas,
            'replica_paths': aggregated_metrics.replica_paths
        },
        'overall_stats': {
            'model_recall': metric_stats_to_dict(aggregated_metrics.overall_stats.model_recall),
            'model_precision': metric_stats_to_dict(aggregated_metrics.overall_stats.model_precision),
            'model_f1': metric_stats_to_dict(aggregated_metrics.overall_stats.model_f1),
            'model_mcc': metric_stats_to_dict(aggregated_metrics.overall_stats.model_mcc),
            'model_mean_pairwise_f1': metric_stats_to_dict(aggregated_metrics.overall_stats.model_mean_pairwise_f1),
            'n_replicas': aggregated_metrics.overall_stats.n_replicas
        },
        'stability_stats': {
            'training_frequency': {
                name: stability_stats_to_dict(stats)
                for name, stats in aggregated_metrics.training_frequency_stats.items()
            },
            'test_frequency': {
                name: stability_stats_to_dict(stats)
                for name, stats in aggregated_metrics.test_frequency_stats.items()
            }
        }
    }
    
    # Add baseline stats if available
    if aggregated_metrics.overall_stats.baseline_recall:
        export_data['overall_stats']['baseline_recall'] = metric_stats_to_dict(
            aggregated_metrics.overall_stats.baseline_recall
        )
        export_data['overall_stats']['baseline_precision'] = metric_stats_to_dict(
            aggregated_metrics.overall_stats.baseline_precision
        )
        export_data['overall_stats']['baseline_f1'] = metric_stats_to_dict(
            aggregated_metrics.overall_stats.baseline_f1
        )
        export_data['overall_stats']['baseline_mcc'] = metric_stats_to_dict(
            aggregated_metrics.overall_stats.baseline_mcc
        )
    
    if aggregated_metrics.overall_stats.baseline_mean_pairwise_f1:
        export_data['overall_stats']['baseline_mean_pairwise_f1'] = metric_stats_to_dict(
            aggregated_metrics.overall_stats.baseline_mean_pairwise_f1
        )
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    sys.exit(main())
