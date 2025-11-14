"""
Insert methods for PortfolioDatabase.
Handles data insertion for portfolio, accounts, holdings, and symbols.
"""

from typing import Dict, List, Optional

import pandas as pd


class PortfolioInsertMixin:
    """Mixin class providing insert methods for portfolio data."""

    def insert_portfolio_summary(
        self,
        date: pd.Timestamp,
        total_value: float,
        net_cash_flow: float = 0,
        investment_cash_flow: float = 0,
        income_cash_flow: float = 0,
        portfolio_mod_dietz_return: Optional[float] = None,
        portfolio_twr_return: Optional[float] = None,
        market_value_change: Optional[float] = None,
    ):
        """Insert portfolio summary for a date."""
        cursor = self.get_cursor()
        cursor.execute(
            """
            INSERT INTO portfolio_summary (
                date, total_value, net_cash_flow, investment_cash_flow,
                income_cash_flow, portfolio_mod_dietz_return, 
                portfolio_twr_return, market_value_change
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                total_value,
                net_cash_flow,
                investment_cash_flow,
                income_cash_flow,
                portfolio_mod_dietz_return,
                portfolio_twr_return,
                market_value_change,
            ),
        )

    def insert_account_data(
        self,
        date: pd.Timestamp,
        account: str,
        total_value: float,
        cash_flow: float = 0,
        weight: float = 0,
        mod_dietz_return: Optional[float] = None,
        twr_return: Optional[float] = None,
    ):
        """Insert account data for a date."""
        cursor = self.get_cursor()
        cursor.execute(
            """
            INSERT INTO account_data (
                date, account, total_value, cash_flow, weight,
                mod_dietz_return, twr_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                account,
                total_value,
                cash_flow,
                weight,
                mod_dietz_return,
                twr_return,
            ),
        )

    def insert_holding(
        self,
        date: pd.Timestamp,
        account: str,
        symbol: str,
        quantity: float,
        value: float,
        weight: float = 0,
    ):
        """Insert holding data for a date."""
        cursor = self.get_cursor()
        cursor.execute(
            """
            INSERT INTO holdings (
                date, account, symbol, quantity, value, weight
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (date, account, symbol, quantity, value, weight),
        )

    def insert_symbol_data(
        self,
        date: pd.Timestamp,
        symbol: str,
        price: Optional[float],
        adj_price: Optional[float],
        total_quantity: float,
        total_value: float,
        weight: float = 0,
        cash_flow: float = 0,
        market_return: Optional[float] = None,
        twr_return: Optional[float] = None,
    ):
        """Insert symbol data for a date."""
        cursor = self.get_cursor()
        cursor.execute(
            """
            INSERT INTO symbol_data (
                date, symbol, price, adj_price, total_quantity, total_value,
                weight, cash_flow, market_return, twr_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date,
                symbol,
                price,
                adj_price,
                total_quantity,
                total_value,
                weight,
                cash_flow,
                market_return,
                twr_return,
            ),
        )

    def insert_day_batch(
        self,
        portfolio_summary: Dict,
        account_data: List[Dict],
        holdings: List[Dict],
        symbol_data: List[Dict],
    ):
        """
        Insert all data for a single day in a batch.

        Args:
            portfolio_summary: Dict with portfolio-level data
            account_data: List of dicts with account-level data
            holdings: List of dicts with holding-level data
            symbol_data: List of dicts with symbol-level data
        """
        self.insert_portfolio_summary(**portfolio_summary)

        for acc_data in account_data:
            self.insert_account_data(**acc_data)

        for holding in holdings:
            self.insert_holding(**holding)

        for sym_data in symbol_data:
            self.insert_symbol_data(**sym_data)

    def bulk_insert_days(
        self,
        days_data: List[Dict],
    ):
        """
        Bulk insert multiple days of data efficiently.

        Args:
            days_data: List of dicts, each containing:
                - portfolio_summary: Dict
                - account_data: List[Dict]
                - holdings: List[Dict]
                - symbol_data: List[Dict]
        """
        with self.transaction():
            for day in days_data:
                self.insert_day_batch(
                    portfolio_summary=day["portfolio_summary"],
                    account_data=day["account_data"],
                    holdings=day["holdings"],
                    symbol_data=day["symbol_data"],
                )
