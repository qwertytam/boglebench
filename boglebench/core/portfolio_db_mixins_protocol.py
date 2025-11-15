"""
Protocol definition for database operations expected by mixin classes.

This Protocol defines the interface that mixin classes expect from the
DatabaseOperations base class. It enables proper type checking without
introducing runtime dependencies.
"""

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, List, Protocol, Union


class DatabaseProtocol(Protocol):
    """
    Protocol defining the database interface expected by mixin classes.

    This Protocol declares the methods that mixins rely on from the
    DatabaseOperations base class. By inheriting from this Protocol,
    mixins can properly type-check their usage of these methods without
    creating circular dependencies or runtime overhead.
    """

    # pylint: disable=unnecessary-ellipsis

    def get_cursor(self) -> sqlite3.Cursor:
        """
        Get a database cursor, ensuring connection is established.

        Returns:
            sqlite3.Cursor: Database cursor

        Raises:
            RuntimeError: If database connection is not established
        """
        ...

    def get_connection(self) -> sqlite3.Connection:
        """
        Get database connection, ensuring it's established.

        Returns:
            sqlite3.Connection: Database connection

        Raises:
            RuntimeError: If database connection is not established
        """
        ...

    @staticmethod
    def normalize_params(params: List[Any]) -> Union[tuple, List[Any]]:
        """
        Ensure params is an acceptable type for pandas.read_sql_query.

        Args:
            params: List of parameters

        Returns:
            Tuple or list of parameters suitable for database queries
        """
        ...

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """
        Context manager for transactions.

        Yields:
            None
        """
        yield
        ...
