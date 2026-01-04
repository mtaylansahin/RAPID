"""Data loader for analysis input files."""

from pathlib import Path

import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.core.utils import get_logger


logger = get_logger(__name__)


@dataclass
class LoadedData:
    """Container for loaded data files."""
    
    labels: pd.DataFrame
    test_set: pd.DataFrame
    train_set: pd.DataFrame
    valid_set: pd.DataFrame
    output: pd.DataFrame


class DataLoader:
    """Handles loading and initial validation of analysis input files."""
    
    def __init__(self):
        """Initialize data loader."""
        self.logger = get_logger(__name__)
    
    def load_all_data(
        self,
        input_directory: str,
        output_file_path: str
    ) -> LoadedData:
        """Load all required data files.
        
        Args:
            input_directory: Directory containing input files
            output_file_path: Path to output/prediction file
            
        Returns:
            LoadedData container with all loaded dataframes
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If file formats are invalid
        """
        try:
            self.logger.info(f"Loading data from {input_directory}")
            
            # Load labels.txt
            labels_df = self._load_labels_file(input_directory)
            
            # Load data split files
            test_df = self._load_data_split_file(input_directory, "test.txt")
            train_df = self._load_data_split_file(input_directory, "train.txt")
            valid_df = self._load_data_split_file(input_directory, "valid.txt")
            
            # Load output/prediction file
            output_df = self._load_output_file(output_file_path)
            
            self.logger.info("Successfully loaded all data files")
            
            return LoadedData(
                labels=labels_df,
                test_set=test_df,
                train_set=train_df,
                valid_set=valid_df,
                output=output_df
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def _load_labels_file(self, input_directory: str) -> pd.DataFrame:
        """Load and process labels.txt file.
        
        Args:
            input_directory: Directory containing labels.txt
            
        Returns:
            Processed labels dataframe
        """
        labels_path = Path(input_directory) / "labels.txt"
        
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        try:
            df_labels = pd.read_csv(labels_path, sep=r"\s+", header=None)
            n_cols = df_labels.shape[1]
            columns_10 = [
                'itype', 'chain_a', 'chain_b', 'resname_a', 'resname_b',
                'resid_a', 'resid_b', 'atom_a', 'atom_b', 'time'
            ]

            if n_cols == 10:
                df_labels.columns = columns_10
                if df_labels.iloc[0].tolist() == columns_10:
                    df_labels = df_labels.iloc[1:].reset_index(drop=True)
            else:
                raise ValueError(
                    f"Unexpected labels.txt format: expected 10 columns, got {n_cols}"
                )
            
            # Add derived columns
            if 'chain_res_a' not in df_labels.columns:
                df_labels['chain_res_a'] = (
                    df_labels['chain_a'].astype(str) + df_labels['resid_a'].astype(str)
                )
            if 'chain_res_b' not in df_labels.columns:
                df_labels['chain_res_b'] = (
                    df_labels['chain_b'].astype(str) + df_labels['resid_b'].astype(str)
                )

            df_labels['chain_a_b_time'] = (
                df_labels['chain_res_a'] +
                df_labels['chain_res_b'] +
                df_labels['time'].astype(str)
            )
            df_labels['chain_a_b'] = df_labels['chain_res_a'] + df_labels['chain_res_b']
            df_labels['residue_a'] = df_labels['resid_a'].astype(str) + "-" + df_labels['resname_a']
            df_labels['residue_b'] = df_labels['resid_b'].astype(str) + "-" + df_labels['resname_b']
            
            self.logger.info(f"Loaded labels file with {len(df_labels)} rows")
            return df_labels
            
        except Exception as e:
            raise ValueError(f"Failed to parse labels file: {e}")
    
    def _load_data_split_file(self, input_directory: str, filename: str) -> pd.DataFrame:
        """Load a data split file (train/test/valid).
        
        Args:
            input_directory: Directory containing the file
            filename: Name of the file to load
            
        Returns:
            Loaded dataframe with standard columns
        """
        file_path = Path(input_directory) / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data split file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, sep=r"\s+", header=None)
            df.columns = ['subject', 'relation', 'object', 'time_stamp']
            
            self.logger.info(f"Loaded {filename} with {len(df)} rows")
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to parse {filename}: {e}")
    
    def _load_output_file(self, output_file_path: str) -> pd.DataFrame:
        """Load the output/prediction file.
        
        Args:
            output_file_path: Path to output file
            
        Returns:
            Loaded output dataframe
        """
        output_path = Path(output_file_path)
        
        if not output_path.exists():
            raise FileNotFoundError(f"Output file not found: {output_path}")
        
        try:
            df = pd.read_csv(output_path, sep=r"\s+", header=None)
            df.columns = ['subject', 'object', 'time_stamp']
            
            self.logger.info(f"Loaded output file with {len(df)} rows")
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to parse output file: {e}")
    
    def validate_data_consistency(self, data: LoadedData) -> Dict[str, any]:
        """Validate consistency across loaded data files.
        
        Args:
            data: Loaded data container
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            "valid": True,
            "issues": [],
            "statistics": {}
        }
        
        try:
            # Check for empty dataframes
            for name, df in [
                ("labels", data.labels),
                ("test", data.test_set),
                ("train", data.train_set),
                ("valid", data.valid_set),
                ("output", data.output)
            ]:
                if df.empty:
                    validation_results["issues"].append(f"{name} dataframe is empty")
                    validation_results["valid"] = False
                else:
                    validation_results["statistics"][f"{name}_rows"] = len(df)
            
            # Check time stamp consistency
            all_times = set()
            for name, df in [("test", data.test_set), ("train", data.train_set), ("valid", data.valid_set)]:
                if 'time_stamp' in df.columns:
                    times = set(df['time_stamp'].unique())
                    all_times.update(times)
                    validation_results["statistics"][f"{name}_unique_times"] = len(times)
            
            validation_results["statistics"]["total_unique_times"] = len(all_times)
            
            self.logger.info(f"Data validation completed: {'PASSED' if validation_results['valid'] else 'FAILED'}")
            
            if validation_results["issues"]:
                for issue in validation_results["issues"]:
                    self.logger.warning(f"Validation issue: {issue}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            validation_results["valid"] = False
            validation_results["issues"].append(f"Validation error: {e}")
            return validation_results 
