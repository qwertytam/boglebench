"""
Centralized logging configuration for BogleBench.

Provides structured logging with YAML configuration and multiple handlers
for console, file, and debug output.
"""

import glob
import inspect
import logging
import logging.config
import logging.handlers
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import ConfigManager
from .workspace import WorkspaceContext


class BogleBenchLogger:
    """Centralized logger for BogleBench portfolio analyzer."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure single logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger if not already done."""
        if not self._initialized:
            self.setup_logging()
            BogleBenchLogger._initialized = True

    def setup_logging(self, config_path: Optional[str] = None):
        """
        Set up logging configuration from YAML file.

        Args:
            config_path: Path to logging config YAML file
        """

        if config_path is None:
            config_path = self._get_default_config_path()

        config_file = Path(config_path)

        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                config = self._update_config_paths(config)
                try:
                    logging.config.dictConfig(config)
                except Exception as config_error:
                    print(
                        f"!! ERROR !!: Specific config error: {config_error}\n"
                        f"Handler config: {config.get('handlers', {})}"
                    )
                    raise

            except (OSError, ValueError) as e:
                # Fallback to basic config if YAML loading fails
                self._setup_fallback_logging()
                print(
                    f"!! ERROR !!: Failed to load logging config from "
                    f"{config_path}: {e}\n"
                    f"Falling back to basic logging configuration"
                )
        else:
            # Use default config if file doesn't exist
            print(f"Logging using config file: {config_path}")
            self._setup_default_logging()

    def _get_default_config_path(self) -> str:
        """Get default path for logging configuration."""
        # Check workspace context first

        workspace = WorkspaceContext.get_workspace()
        if workspace:
            logging_config = workspace / "config" / "logging.yaml"
            if logging_config.exists():
                return str(logging_config)

        # Check environment variable
        env_path = os.getenv("BOGLEBENCH_WORKSPACE")
        if env_path:
            return str(Path(env_path) / "config" / "logging.yaml")

        # Fallback to config manager
        config_manager = ConfigManager()
        config_dir = config_manager.get_data_path("config")
        return str(config_dir / "logging.yaml")

    def _setup_default_logging(self):
        """Set up default logging configuration from template file."""
        template_path = self._get_template_path()

        if template_path.exists():
            try:
                with open(template_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                config = self._update_config_paths(config)
                logging.config.dictConfig(config)
                return
            except (OSError, ValueError) as e:
                logging.error(
                    "Failed to load default config from template: %s", e
                )

        # Final fallback if template doesn't exist
        print("Setting up fallback logging")
        self._setup_fallback_logging()

    def _get_template_path(self) -> Path:
        """Get path to the logging configuration template."""
        # Get path to the boglebench package directory
        package_dir = Path(__file__).parent.parent
        return package_dir / "templates" / "logging_config_template.yaml"

    def _update_config_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update relative paths in config to absolute paths."""

        workspace = WorkspaceContext.get_workspace()
        if workspace:
            log_dir = workspace / "logs"
        else:
            try:
                config_manager = ConfigManager()
                log_dir = config_manager.get_data_path("logs")
            except ValueError:
                log_dir = Path(tempfile.gettempdir())

        log_dir.mkdir(exist_ok=True)

        # Update handler file paths
        handlers = config.get("handlers", {})
        for _, handler_config in handlers.items():
            if "filename" in handler_config:
                original_filename = handler_config["filename"]
                if not Path(original_filename).is_absolute():
                    new_path = str(log_dir / original_filename)
                    handler_config["filename"] = new_path

        return config

    def _get_log_file_path(self) -> str:
        """Get path for log file."""
        # Skip file logging during tests
        if "pytest" in sys.modules:
            return str(Path(tempfile.gettempdir()) / "boglebench_test.log")

        # Use workspace context
        workspace = WorkspaceContext.get_workspace()
        if workspace:
            log_dir = workspace / "logs"
            log_dir.mkdir(exist_ok=True)
            return str(log_dir / "boglebench.log")

        # Fallback through config manager
        try:
            config_manager = ConfigManager()
            log_dir = config_manager.get_data_path("logs")
            log_dir.mkdir(exist_ok=True)
            return str(log_dir / "boglebench.log")
        except ValueError:
            return str(Path(tempfile.gettempdir()) / "boglebench.log")

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module.

        Args:
            name: Logger name (typically __name__ of calling module)

        Returns:
            Configured logger instance
        """
        return logging.getLogger(f"boglebench.{name}")

    def create_default_config_file(self, output_path: Optional[str] = None):
        """
        Copy the default logging configuration template to user's config
        directory.

        Args:
            output_path: Where to create the config file
        """
        if output_path is None:
            output_path = self._get_default_config_path()

        template_path = self._get_template_path()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if template_path.exists():
            # Copy template file
            shutil.copy2(template_path, output_file)
            print(f"Created logging configuration: {output_file}")
        else:
            print(
                f"WARNING: Template file not found at {template_path}\n"
                "Creating minimal logging configuration instead"
            )

            # Minimal fallback config
            minimal_config = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "simple": {"format": "%(levelname)s - %(message)s"}
                },
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "INFO",
                        "formatter": "simple",
                    }
                },
                "root": {"level": "INFO", "handlers": ["console"]},
            }

            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    minimal_config, f, default_flow_style=False, indent=2
                )

    def _create_rotating_handler(
        self, log_file_path: str, level: str = "DEBUG"
    ) -> logging.Handler:
        """Create a rotating file handler with configuration
        from config file."""

        default_max_size_mb = 10
        default_backup_count = 5

        try:
            config_manager = ConfigManager()
            max_size_mb = config_manager.get(
                "logging.rotation.max_file_size_mb"
            )
            if isinstance(max_size_mb, dict):
                max_size_mb = max_size_mb.get("value", default_max_size_mb)

            if max_size_mb is None or max_size_mb <= 0:
                max_size_mb = default_max_size_mb

            backup_count = config_manager.get("logging.rotation.backup_count")
            if isinstance(backup_count, dict):
                backup_count = backup_count.get("value", default_backup_count)
            if backup_count is None or backup_count < 0:
                backup_count = default_backup_count

            max_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes

            handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                mode="a",
            )

        except (OSError, ValueError):
            # Fallback to basic rotating handler
            handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                mode="a",
            )

        handler.setLevel(getattr(logging, level.upper()))
        return handler

    def _setup_fallback_logging(self):
        """Enhanced fallback with rotation."""
        log_file = self._get_log_file_path()

        # Create rotating handler
        file_handler = self._create_rotating_handler(log_file)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(message)s")
        )
        root_logger.addHandler(console_handler)

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days."""

        workspace = WorkspaceContext.get_workspace()
        if not workspace:
            return

        log_dir = workspace / "logs"
        if not log_dir.exists():
            return

        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

        # Find all log files
        log_patterns = ["*.log", "*.log.*"]

        for pattern in log_patterns:
            for log_file in glob.glob(str(log_dir / pattern)):
                try:
                    if os.path.getmtime(log_file) < cutoff_time:
                        os.remove(log_file)
                        print(f"Cleaned up old log file: {log_file}")
                except OSError:
                    pass  # File might be in use or already deleted


# Convenience functions for easy import
_logger_instance = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for the calling module."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = BogleBenchLogger()

    if name is None:
        frame = inspect.currentframe()
        if frame is None:
            name = "unknown"
        else:
            frame = frame.f_back

        if frame is None:
            name = "unknown"
        else:
            name = frame.f_globals.get("__name__", "unknown")

        if name is None:
            name = "unknown"

    # Remove 'boglebench.' prefix if present
    name = name.replace("boglebench.", "")

    return _logger_instance.get_logger(name)


def setup_logging(config_path: Optional[str] = None):
    """Initialize logging system."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = BogleBenchLogger()
    _logger_instance.setup_logging(config_path)


def create_logging_config(output_path: Optional[str] = None):
    """Create default logging configuration file."""
    logger_instance = BogleBenchLogger()
    logger_instance.create_default_config_file(output_path)
