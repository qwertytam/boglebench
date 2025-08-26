"""
Centralized logging configuration for BogleBench.

Provides structured logging with YAML configuration and multiple handlers
for console, file, and debug output.
"""

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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

        # print(f"DEBUG: Using logging config: {config_path}")

        config_file = Path(config_path)

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)

                config = self._update_config_paths(config)
                logging.config.dictConfig(config)
                # print(f"DEBUG: Loaded logging config from {config_path}")
                # print(config)
            except Exception as e:
                # Fallback to basic config if YAML loading fails
                self._setup_fallback_logging()
                # print(
                #     f"DEBUG: Error: Failed to load logging config from {config_path}: {e}"
                # )
        else:
            # Use default config if file doesn't exist
            # print(
            #     f"DEBUG: Warning: Logging config file not found at {config_path}"
            # )
            self._setup_default_logging()

    def _get_default_config_path(self) -> str:
        """Get default path for logging configuration."""
        # Check workspace context first
        from .workspace import WorkspaceContext

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
        from .config import ConfigManager

        config_manager = ConfigManager()
        config_dir = config_manager.get_data_path("config")
        return str(config_dir / "logging.yaml")

    def _setup_default_logging(self):
        """Set up default logging configuration from template file."""
        template_path = self._get_template_path()

        if template_path.exists():
            try:
                with open(template_path, "r") as f:
                    config = yaml.safe_load(f)

                # print("DEBUG: Updating config paths")
                config = self._update_config_paths(config)
                logging.config.dictConfig(config)
                return
            except Exception as e:
                logging.error(
                    f"Failed to load default config from template: {e}"
                )

        # Final fallback if template doesn't exist
        # print("DEBUG: Setting up fallback logging")
        self._setup_fallback_logging()

    def _get_template_path(self) -> Path:
        """Get path to the logging configuration template."""
        # Get path to the boglebench package directory
        package_dir = Path(__file__).parent.parent
        return package_dir / "templates" / "logging_config_template.yaml"

    def _update_config_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update relative paths in config to absolute paths."""
        from .workspace import WorkspaceContext

        # print("DEBUG: Updating config paths")
        workspace = WorkspaceContext.get_workspace()
        if workspace:
            log_dir = workspace / "logs"
            # print(f"DEBUG: Using workspace log directory: {log_dir}")
        else:
            try:
                from .config import ConfigManager

                config_manager = ConfigManager()
                log_dir = config_manager.get_data_path("logs")
                # print(f"DEBUG: Using config manager log directory: {log_dir}")
            except Exception:
                import tempfile

                log_dir = Path(tempfile.gettempdir())
                # print(f"DEBUG: Using temp log directory: {log_dir}")

        log_dir.mkdir(exist_ok=True)

        # Update handler file paths
        handlers = config.get("handlers", {})
        for handler_name, handler_config in handlers.items():
            if "filename" in handler_config:
                original_filename = handler_config["filename"]
                if not Path(original_filename).is_absolute():
                    new_path = str(log_dir / original_filename)
                    handler_config["filename"] = new_path
                    # print(
                    #    f"DEBUG: Updated {handler_name} log path: {original_filename} -> {new_path}"
                    # )

        return config

    def _setup_fallback_logging(self):
        """Minimal logging setup if all else fails."""
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s - %(message)s"
        )

    def _get_log_file_path(self) -> str:
        """Get path for log file."""
        # Skip file logging during tests
        import sys

        if "pytest" in sys.modules:
            import tempfile

            return str(Path(tempfile.gettempdir()) / "boglebench_test.log")

        # Use workspace context
        from .workspace import WorkspaceContext

        workspace = WorkspaceContext.get_workspace()
        if workspace:
            log_dir = workspace / "logs"
            log_dir.mkdir(exist_ok=True)
            return str(log_dir / "boglebench.log")

        # Fallback through config manager
        try:
            from .config import ConfigManager

            config_manager = ConfigManager()
            log_dir = config_manager.get_data_path("logs")
            log_dir.mkdir(exist_ok=True)
            return str(log_dir / "boglebench.log")
        except Exception:
            import tempfile

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
        Copy the default logging configuration template to user's config directory.

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
            import shutil

            shutil.copy2(template_path, output_file)
            print(f"Created logging configuration: {output_file}")
        else:
            print(f"Warning: Template file not found at {template_path}")
            print("Creating minimal logging configuration instead")

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

            with open(output_file, "w") as f:
                yaml.dump(minimal_config, f, default_flow_style=False, indent=2)


# Convenience functions for easy import
_logger_instance = None


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance for the calling module."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = BogleBenchLogger()

    if name is None:
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown")
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
