"""Configuration manager for parsing and validating experiment configurations."""

import argparse
import os
from typing import List, Union, Optional
import numpy as np

from .experiment_config import ExperimentConfig, HyperparameterConfig, DataConfig


class ConfigManager:
    """Manages configuration parsing, validation, and loading for experiments."""

    def __init__(self):
        """Initialize the configuration manager."""
        self._parser = self._create_argument_parser()

    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description='RAPID: Protein-protein interaction dynamics prediction',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Data configuration arguments
        data_group = parser.add_argument_group('Data Configuration')
        data_group.add_argument(
            '--data_dir',
            type=str,
            required=True,
            help='Directory containing the interaction files'
        )
        data_group.add_argument(
            '--replica',
            type=str,
            required=True,
            help='Replica number for the experiment'
        )
        data_group.add_argument(
            '--chain1',
            type=str,
            required=True,
            help='First chain identifier'
        )
        data_group.add_argument(
            '--chain2',
            type=str,
            required=True,
            help='Second chain identifier'
        )
        data_group.add_argument(
            '--train_ratio',
            type=float,
            required=True,
            help='Training ratio for splitting the data (0.0-1.0)'
        )
        data_group.add_argument(
            '--valid_ratio',
            type=float,
            required=True,
            help='Validation ratio for splitting the data (0.0-1.0)'
        )
        data_group.add_argument(
            '--interaction_type',
            type=str,
            default='residue',
            choices=['residue', 'atomic'],
            help='Type of interaction analysis'
        )

        # Hyperparameter configuration arguments
        hyperparam_group = parser.add_argument_group('Hyperparameter Configuration')
        hyperparam_group.add_argument(
            '--dropout',
            type=str,
            default='0.5',
            help='Dropout rate(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--learning_rate',
            type=str,
            default='0.001',
            help='Learning rate(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--batch_size',
            type=str,
            default='128',
            help='Batch size(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--pretrain_epochs',
            type=str,
            default='30',
            help='Pretraining epoch(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--train_epochs',
            type=str,
            default='10',
            help='Training epoch(s). Single value or range [start,stop,step]'
        )
        hyperparam_group.add_argument(
            '--n_hidden',
            type=str,
            default='100',
            help='Hidden unit(s). Single value or range [start,stop,step]'
        )

        # System configuration arguments
        system_group = parser.add_argument_group('System Configuration')
        system_group.add_argument(
            '--gpu',
            type=int,
            default=0,
            help='GPU device ID to use'
        )
        system_group.add_argument(
            '--seed',
            type=int,
            default=999,
            help='Random seed for reproducibility'
        )

        return parser

    def parse_configuration(self, args: Optional[List[str]] = None) -> ExperimentConfig:
        """Parse command line arguments or config file into ExperimentConfig.
        
        Args:
            args: Optional list of arguments to parse. If None, uses sys.argv
            
        Returns:
            Validated ExperimentConfig instance
            
        Raises:
            ValueError: If configuration validation fails
            FileNotFoundError: If config file is specified but not found
        """
        parsed_args = self._parser.parse_args(args)

        # Parse hyperparameters from command line
        hyperparameters = self._parse_hyperparameters(parsed_args)

        # Create data configuration
        data_config = DataConfig(
            data_directory=parsed_args.data_dir,
            replica=parsed_args.replica,
            chain1=parsed_args.chain1,
            chain2=parsed_args.chain2,
            train_ratio=parsed_args.train_ratio,
            validation_ratio=parsed_args.valid_ratio,
            interaction_type=parsed_args.interaction_type
        )

        # Create experiment configuration
        experiment_name = os.path.basename(parsed_args.data_dir.rstrip('/'))
        experiment_config = ExperimentConfig(
            experiment_name=experiment_name,
            data_config=data_config,
            hyperparameters=hyperparameters,
            gpu_device=parsed_args.gpu,
            random_seed=parsed_args.seed
        )

        return experiment_config

    def _parse_hyperparameters(self, args: argparse.Namespace) -> HyperparameterConfig:
        """Parse hyperparameter arguments into HyperparameterConfig.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Validated HyperparameterConfig instance
        """
        return HyperparameterConfig(
            dropout_rates=self._parse_parameter_range(args.dropout, float),
            learning_rates=self._parse_parameter_range(args.learning_rate, float),
            batch_sizes=self._parse_parameter_range(args.batch_size, int),
            pretrain_epochs=self._parse_parameter_range(args.pretrain_epochs, int),
            train_epochs=self._parse_parameter_range(args.train_epochs, int),
            hidden_units=self._parse_parameter_range(args.n_hidden, int)
        )

    def _parse_parameter_range(self, param_str: str, param_type: type) -> List[Union[int, float]]:
        """Parse parameter string into list of values.
        
        Supports single values or ranges in format [start,stop,step].
        
        Args:
            param_str: Parameter string to parse
            param_type: Type to convert values to (int or float)
            
        Returns:
            List of parameter values
            
        Raises:
            ValueError: If parameter format is invalid
        """
        if param_str is None:
            return []

        try:
            if param_str.startswith('[') and param_str.endswith(']'):
                # Parse range format [start,stop,step] or [val1,val2,val3,...]
                values_str = param_str[1:-1].split(',')
                values = [param_type(val.strip()) for val in values_str]

                if len(values) == 3 and param_type == float:
                    # Range format for floats
                    start, stop, step = values
                    return list(np.arange(start, stop, step))
                elif len(values) == 3 and param_type == int:
                    # Range format for ints
                    start, stop, step = values
                    return list(range(start, stop, step))
                else:
                    # List of explicit values
                    return values
            else:
                # Single value
                return [param_type(param_str)]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid parameter format '{param_str}': {e}")
