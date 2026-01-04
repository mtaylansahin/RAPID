"""Process management utilities for safe subprocess execution."""

import subprocess
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessResult:
    """Result of a process execution."""

    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    command: str

    @property
    def succeeded(self) -> bool:
        """Check if process completed successfully."""
        return self.return_code == 0


class ProcessError(Exception):
    """Exception raised when a process fails."""

    def __init__(self, result: ProcessResult):
        self.result = result
        super().__init__(
            f"Process failed with return code {result.return_code}: {result.stderr}"
        )


class ProcessManager:
    """Manages subprocess execution with proper error handling and logging."""

    def __init__(self, timeout: float = 3600.0):
        """Initialize process manager.
        
        Args:
            timeout: Default timeout for process execution in seconds
        """
        self.timeout = timeout
        self.logger = get_logger(__name__)

    def run_command(
            self,
            command: Union[str, List[str]],
            working_directory: Optional[Union[str, Path]] = None,
            environment: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None,
            check: bool = True,
            capture_output: bool = True
    ) -> ProcessResult:
        """Execute a command with proper error handling and logging.
        
        Args:
            command: Command to execute (string or list of arguments)
            working_directory: Working directory for command execution
            environment: Environment variables for the process
            timeout: Timeout in seconds (uses default if None)
            check: Whether to raise exception on non-zero return code
            capture_output: Whether to capture stdout and stderr
            
        Returns:
            ProcessResult containing execution details
            
        Raises:
            ProcessError: If process fails and check=True
            subprocess.TimeoutExpired: If process times out
        """
        if isinstance(command, str):
            command_list = command.split()
            command_str = command
        else:
            command_list = command
            command_str = ' '.join(command)

        env = os.environ.copy()
        if environment:
            env.update(environment)

        execution_timeout = timeout or self.timeout

        self.logger.info(f"Executing command: {command_str}")
        if working_directory:
            self.logger.info(f"Working directory: {working_directory}")

        start_time = time.time()

        try:
            result = subprocess.run(
                command_list,
                cwd=working_directory,
                env=env,
                timeout=execution_timeout,
                capture_output=capture_output,
                text=True,
                check=False
            )

            execution_time = time.time() - start_time

            process_result = ProcessResult(
                return_code=result.returncode,
                stdout=result.stdout if capture_output else "",
                stderr=result.stderr if capture_output else "",
                execution_time=execution_time,
                command=command_str
            )

            if process_result.succeeded:
                self.logger.info(
                    f"Command completed successfully in {execution_time:.2f}s"
                )
                if process_result.stdout:
                    self.logger.debug(f"stdout: {process_result.stdout}")
            else:
                self.logger.error(
                    f"Command failed with return code {process_result.return_code}"
                )
                if process_result.stderr:
                    self.logger.error(f"stderr: {process_result.stderr}")

                if check:
                    raise ProcessError(process_result)

            return process_result

        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Command timed out after {execution_time:.2f}s")
            raise
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise

    def run_python_script(
            self,
            script_path: Union[str, Path],
            args: List[str] = None,
            working_directory: Optional[Union[str, Path]] = None,
            environment: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None,
            check: bool = True,
            capture_output: bool = True
    ) -> ProcessResult:
        """Execute a Python script with arguments.
        
        Args:
            script_path: Path to the Python script
            args: Command line arguments for the script
            working_directory: Working directory for script execution
            environment: Environment variables for the process
            timeout: Timeout in seconds
            check: Whether to raise exception on non-zero return code
            capture_output: Whether to capture stdout and stderr (False for real-time output)
            
        Returns:
            ProcessResult containing execution details
        """
        command = [sys.executable, str(script_path)]
        if args:
            command.extend(args)

        return self.run_command(
            command=command,
            working_directory=working_directory,
            environment=environment,
            timeout=timeout,
            check=check,
            capture_output=capture_output
        )


class RENetProcessManager(ProcessManager):
    """Specialized process manager for RE-Net operations."""

    def __init__(self, renet_directory: Union[str, Path], timeout: float = 3600.0):
        """Initialize RE-Net process manager.
        
        Args:
            renet_directory: Path to RE-Net directory
            timeout: Default timeout for operations
        """
        super().__init__(timeout)
        self.renet_directory = Path(renet_directory)

        if not self.renet_directory.exists():
            raise FileNotFoundError(f"RE-Net directory not found: {renet_directory}")

    def generate_history_graph(self, dataset_name: str) -> ProcessResult:
        """Generate history graph for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            ProcessResult from the operation
        """
        data_directory = self.renet_directory / "data" / dataset_name
        script_path = "get_history_graph.py"  # Use relative path since we're setting working directory

        return self.run_python_script(
            script_path=script_path,
            working_directory=data_directory
        )

    def pretrain_model(
            self,
            dataset_name: str,
            dropout: float,
            n_hidden: int,
            learning_rate: float,
            max_epochs: int,
            batch_size: int,
            gpu_device: int = 0,
            maxpool: int = 1
    ) -> ProcessResult:
        """Run RE-Net pretraining.
        
        Args:
            dataset_name: Dataset name
            dropout: Dropout rate
            n_hidden: Number of hidden units
            learning_rate: Learning rate
            max_epochs: Maximum epochs
            batch_size: Batch size
            gpu_device: GPU device ID
            maxpool: Maxpool parameter
            
        Returns:
            ProcessResult from pretraining
        """
        args = [
            "-d", dataset_name,
            "--gpu", str(gpu_device),
            "--dropout", str(dropout),
            "--n-hidden", str(n_hidden),
            "--lr", str(learning_rate),
            "--max-epochs", str(max_epochs),
            "--batch-size", str(batch_size),
            "--maxpool", str(maxpool)
        ]

        return self.run_python_script(
            script_path="pretrain.py",  # Use relative path since we're setting working directory
            args=args,
            working_directory=self.renet_directory,
            capture_output=False  # Allow real-time output for training progress
        )

    def train_model(
            self,
            dataset_name: str,
            dropout: float,
            n_hidden: int,
            learning_rate: float,
            max_epochs: int,
            batch_size: int,
            gpu_device: int = 0,
            num_k: int = 5
    ) -> ProcessResult:
        """Run RE-Net training.
        
        Args:
            dataset_name: Dataset name
            dropout: Dropout rate
            n_hidden: Number of hidden units
            learning_rate: Learning rate
            max_epochs: Maximum epochs
            batch_size: Batch size
            gpu_device: GPU device ID
            num_k: num_k parameter
            
        Returns:
            ProcessResult from training
        """
        args = [
            "-d", dataset_name,
            "--gpu", str(gpu_device),
            "--dropout", str(dropout),
            "--n-hidden", str(n_hidden),
            "--lr", str(learning_rate),
            "--max-epochs", str(max_epochs),
            "--batch-size", str(batch_size),
            "--num-k", str(num_k)
        ]

        return self.run_python_script(
            script_path="train.py",  # Use relative path since we're setting working directory
            args=args,
            working_directory=self.renet_directory,
            capture_output=False  # Allow real-time output for training progress
        )

    def test_model(
            self,
            dataset_name: str,
            n_hidden: int,
            gpu_device: int = 0,
            num_k: int = 5
    ) -> ProcessResult:
        """Run RE-Net testing.
        
        Args:
            dataset_name: Dataset name
            n_hidden: Number of hidden units
            gpu_device: GPU device ID
            num_k: num_k parameter
            
        Returns:
            ProcessResult from testing
        """
        args = [
            "-d", dataset_name,
            "--gpu", str(gpu_device),
            "--n-hidden", str(n_hidden),
            "--num-k", str(num_k)
        ]

        return self.run_python_script(
            script_path="test.py",  # Use relative path since we're setting working directory
            args=args,
            working_directory=self.renet_directory
        )
