"""
Workspace context management for BogleBench.
"""

import os
from pathlib import Path
from typing import Optional


class WorkspaceContext:
    """Manages workspace context across the application."""

    _current_workspace: Optional[Path] = None

    @classmethod
    def set_workspace(cls, workspace_path: str | Path):
        """Set the current workspace path."""
        cls._current_workspace = Path(workspace_path).expanduser().resolve()

        # Set environment variable for child processes
        os.environ["BOGLEBENCH_WORKSPACE"] = str(cls._current_workspace)

    @classmethod
    def get_workspace(cls) -> Optional[Path]:
        """Get the current workspace path."""
        if cls._current_workspace:
            return cls._current_workspace

        # Check environment variable
        env_workspace = os.getenv("BOGLEBENCH_WORKSPACE")
        if env_workspace:
            cls._current_workspace = Path(env_workspace)
            return cls._current_workspace

        return None

    @classmethod
    def discover_workspace(cls, start_path: str | Path) -> Optional[Path]:
        """Discover workspace by looking for config/config.yaml."""
        current = Path(start_path).expanduser().resolve()

        # If start_path is a file, start from its parent
        if current.is_file():
            current = current.parent

        # Check current directory and parents
        for _ in range(10):  # Reasonable limit
            config_file = current / "config" / "config.yaml"

            if config_file.exists():
                cls.set_workspace(current)
                return current

            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent

        return None
