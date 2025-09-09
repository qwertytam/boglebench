"""
Configuration management for BogleBench portfolio analyzer.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """Manages configuration for BogleBench portfolio analysis."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to config file. If None, uses default locations.
        """
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config()

    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """Resolve configuration file path."""
        if config_path:
            return Path(config_path).expanduser()

        # Check for workspace context first
        from .workspace import WorkspaceContext

        workspace = WorkspaceContext.get_workspace()
        if workspace:
            workspace_config = workspace / "config" / "config.yaml"
            if workspace_config.exists():
                return workspace_config

        # Try environment variable
        env_path = os.getenv("BOGLEBENCH_CONFIG_PATH")
        if env_path:
            return Path(env_path).expanduser()

        # Default locations (fallback)
        default_locations = [
            Path.home() / "boglebench_data" / "config" / "config.yaml",
            Path.home() / ".boglebench" / "config.yaml",
            Path.cwd() / "config.yaml",
        ]

        for location in default_locations:
            if location.exists():
                return location

        return default_locations[0]

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from template file."""
        template_path = self._get_template_path()

        if template_path.exists():
            try:
                with open(template_path, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"WARNING: Could not load config template: {e}")

        # Minimal fallback if template missing
        return {
            "data": {"base_path": "~/boglebench_data"},
            "settings": {"benchmark_ticker": "SPY"},
            "analysis": {"performance_metrics": ["total_return"]},
        }

    def _get_template_path(self) -> Path:
        """Get path to config template file."""
        package_dir = Path(__file__).parent.parent
        return package_dir / "templates" / "config_template.yaml"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use template defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    user_config = yaml.safe_load(f)

                # Merge with template defaults
                config = self._load_default_config()
                self._deep_merge(config, user_config)
                return config

            except Exception as e:
                print(
                    f"!! ERROR !!: loading config from {self.config_path}: {e}"
                    "\nUsing template configuration."
                )

        return self._load_default_config()

    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge update dict into base dict."""
        for key, value in update.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated key path (e.g., 'data.base_path')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_data_path(self, relative_path: str = "") -> Path:
        """Get absolute path for data files."""
        base_path = Path(self.get("data.base_path")).expanduser()
        return base_path / relative_path if relative_path else base_path

    def get_transactions_path(self) -> Path:
        """Get path to transactions file."""
        transactions_file = self.get("data.transactions_file")
        if not isinstance(transactions_file, str):
            transactions_file = ""
        return self.get_data_path(transactions_file)

    def get_market_data_path(self) -> Path:
        """Get path to market data cache directory."""
        return self.get_data_path(self.get("data.market_data_cache"))

    def get_output_path(self) -> Path:
        """Get path to output directory."""
        return self.get_data_path(self.get("data.output_path"))

    def create_config_file(self, config_path: Optional[str] = None) -> Path:
        """Copy template configuration file to user's location."""
        if config_path:
            path = Path(config_path).expanduser()
        else:
            path = self.config_path

        template_path = self._get_template_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        if template_path.exists():
            import shutil

            shutil.copy2(template_path, path)
            print(f"INFO: Created configuration file: {path}")
        else:
            # Fallback: create from loaded defaults
            config = self._load_default_config()
            with open(path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            print(f"INFO: Created configuration file: {path}")

        return path

    def validate_paths(self) -> bool:
        """Validate that required paths exist."""
        required_paths = [
            self.get_data_path(),
            self.get_market_data_path(),
            self.get_output_path(),
        ]

        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(path)

        if missing_paths:
            print("WARNING: Missing required directories:")
            for path in missing_paths:
                print(f"  - {path}")
            return False

        return True
