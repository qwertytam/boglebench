"""
Database schema for normalized portfolio history storage with temporal symbol attributes.
"""

SCHEMA_VERSION = 2  # Incremented for temporal attributes

PORTFOLIO_SUMMARY_TABLE = """
CREATE TABLE IF NOT EXISTS portfolio_summary (
    date TIMESTAMP PRIMARY KEY,
    total_value REAL NOT NULL,
    net_cash_flow REAL DEFAULT 0,
    investment_cash_flow REAL DEFAULT 0,
    income_cash_flow REAL DEFAULT 0,
    portfolio_mod_dietz_return REAL,
    portfolio_twr_return REAL,
    market_value_change REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_portfolio_date 
ON portfolio_summary(date);
"""

ACCOUNT_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS account_data (
    date TIMESTAMP NOT NULL,
    account TEXT NOT NULL,
    total_value REAL NOT NULL,
    cash_flow REAL DEFAULT 0,
    weight REAL DEFAULT 0,
    mod_dietz_return REAL,
    twr_return REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, account),
    FOREIGN KEY (date) REFERENCES portfolio_summary(date) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_account_date 
ON account_data(date);

CREATE INDEX IF NOT EXISTS idx_account_name 
ON account_data(account);

CREATE INDEX IF NOT EXISTS idx_account_date_account 
ON account_data(date, account);
"""

HOLDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS holdings (
    date TIMESTAMP NOT NULL,
    account TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL,
    value REAL NOT NULL,
    weight REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, account, symbol),
    FOREIGN KEY (date, account) REFERENCES account_data(date, account) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_holdings_date 
ON holdings(date);

CREATE INDEX IF NOT EXISTS idx_holdings_symbol 
ON holdings(symbol);

CREATE INDEX IF NOT EXISTS idx_holdings_account_symbol 
ON holdings(account, symbol);

CREATE INDEX IF NOT EXISTS idx_holdings_date_symbol 
ON holdings(date, symbol);
"""

SYMBOL_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS symbol_data (
    date TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    price REAL,
    adj_price REAL,
    total_quantity REAL NOT NULL,
    total_value REAL NOT NULL,
    weight REAL DEFAULT 0,
    cash_flow REAL DEFAULT 0,
    market_return REAL,
    twr_return REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, symbol),
    FOREIGN KEY (date) REFERENCES portfolio_summary(date) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_symbol_date 
ON symbol_data(date);

CREATE INDEX IF NOT EXISTS idx_symbol_name 
ON symbol_data(symbol);

CREATE INDEX IF NOT EXISTS idx_symbol_date_symbol 
ON symbol_data(date, symbol);
"""

# NEW: Temporal symbol attributes table
SYMBOL_ATTRIBUTES_TABLE = """
CREATE TABLE IF NOT EXISTS symbol_attributes (
    symbol TEXT NOT NULL,
    effective_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP,  -- NULL means current/active
    asset_class TEXT,
    geography TEXT,
    region TEXT,
    sector TEXT,
    style TEXT,
    market_cap TEXT,
    fund_type TEXT,
    expense_ratio REAL,
    dividend_yield REAL,
    is_esg BOOLEAN DEFAULT 0,
    description TEXT,
    source TEXT,  -- How this data was obtained ('user', 'api', 'inferred')
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, effective_date)
);

CREATE INDEX IF NOT EXISTS idx_symbol_attr_symbol 
ON symbol_attributes(symbol);

CREATE INDEX IF NOT EXISTS idx_symbol_attr_effective_date 
ON symbol_attributes(effective_date);

CREATE INDEX IF NOT EXISTS idx_symbol_attr_end_date 
ON symbol_attributes(end_date);

CREATE INDEX IF NOT EXISTS idx_symbol_attr_symbol_dates 
ON symbol_attributes(symbol, effective_date, end_date);

CREATE INDEX IF NOT EXISTS idx_symbol_attr_asset_class 
ON symbol_attributes(asset_class);

CREATE INDEX IF NOT EXISTS idx_symbol_attr_geography 
ON symbol_attributes(geography);

CREATE INDEX IF NOT EXISTS idx_symbol_attr_sector 
ON symbol_attributes(sector);

CREATE INDEX IF NOT EXISTS idx_symbol_attr_style 
ON symbol_attributes(style);

-- Trigger to maintain end_date when new version is added
CREATE TRIGGER IF NOT EXISTS update_previous_end_date
AFTER INSERT ON symbol_attributes
FOR EACH ROW
BEGIN
    -- Set end_date of previous record to day before new effective_date
    UPDATE symbol_attributes
    SET end_date = datetime(NEW.effective_date, '-1 day')
    WHERE symbol = NEW.symbol 
      AND effective_date < NEW.effective_date
      AND end_date IS NULL
      AND effective_date != NEW.effective_date;
END;
"""

METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Useful views for common queries
VIEWS = """
-- View: Latest holdings
CREATE VIEW IF NOT EXISTS latest_holdings AS
SELECT h.* 
FROM holdings h
INNER JOIN (
    SELECT MAX(date) as max_date FROM holdings
) latest ON h.date = latest.max_date;

-- View: Latest portfolio summary
CREATE VIEW IF NOT EXISTS latest_portfolio AS
SELECT ps.*, 
       (SELECT MAX(date) FROM portfolio_summary) as is_latest
FROM portfolio_summary ps
WHERE ps.date = (SELECT MAX(date) FROM portfolio_summary);

-- View: Current (active) symbol attributes
CREATE VIEW IF NOT EXISTS current_symbol_attributes AS
SELECT 
    symbol,
    effective_date,
    asset_class,
    geography,
    region,
    sector,
    style,
    market_cap,
    fund_type,
    expense_ratio,
    dividend_yield,
    is_esg,
    description,
    source,
    created_at,
    updated_at
FROM symbol_attributes
WHERE end_date IS NULL
ORDER BY symbol;

-- View: Holdings with prices and current attributes
CREATE VIEW IF NOT EXISTS holdings_with_attributes AS
SELECT 
    h.date,
    h.account,
    h.symbol,
    h.quantity,
    h.value,
    h.weight,
    s.price,
    s.adj_price,
    s.market_return,
    s.twr_return,
    sa.asset_class,
    sa.geography,
    sa.region,
    sa.sector,
    sa.style,
    sa.market_cap,
    sa.fund_type
FROM holdings h
LEFT JOIN symbol_data s ON h.date = s.date AND h.symbol = s.symbol
LEFT JOIN LATERAL (
    SELECT * FROM symbol_attributes sa_inner
    WHERE sa_inner.symbol = h.symbol
      AND sa_inner.effective_date <= h.date
      AND (sa_inner.end_date IS NULL OR sa_inner.end_date >= h.date)
    ORDER BY sa_inner.effective_date DESC
    LIMIT 1
) sa ON TRUE;

-- View: Symbol data with current attributes
CREATE VIEW IF NOT EXISTS symbol_data_with_attributes AS
SELECT 
    sd.date,
    sd.symbol,
    sd.price,
    sd.adj_price,
    sd.total_quantity,
    sd.total_value,
    sd.weight,
    sd.cash_flow,
    sd.market_return,
    sd.twr_return,
    sa.asset_class,
    sa.geography,
    sa.region,
    sa.sector,
    sa.style,
    sa.market_cap,
    sa.fund_type
FROM symbol_data sd
LEFT JOIN LATERAL (
    SELECT * FROM symbol_attributes sa_inner
    WHERE sa_inner.symbol = sd.symbol
      AND sa_inner.effective_date <= sd.date
      AND (sa_inner.end_date IS NULL OR sa_inner.end_date >= sd.date)
    ORDER BY sa_inner.effective_date DESC
    LIMIT 1
) sa ON TRUE;
"""

ALL_TABLES = [
    PORTFOLIO_SUMMARY_TABLE,
    ACCOUNT_DATA_TABLE,
    HOLDINGS_TABLE,
    SYMBOL_DATA_TABLE,
    SYMBOL_ATTRIBUTES_TABLE,  # NEW: Temporal attributes
    METADATA_TABLE,
    VIEWS,
]


def get_schema_info() -> dict:
    """Return schema information for documentation."""
    return {
        "version": SCHEMA_VERSION,
        "tables": {
            "portfolio_summary": {
                "description": "Daily portfolio-level aggregates",
                "primary_key": ["date"],
            },
            "account_data": {
                "description": "Daily account-level aggregates",
                "primary_key": ["date", "account"],
            },
            "holdings": {
                "description": "Daily position-level data (account × symbol)",
                "primary_key": ["date", "account", "symbol"],
            },
            "symbol_data": {
                "description": "Daily symbol-level aggregates across all accounts",
                "primary_key": ["date", "symbol"],
            },
            "symbol_attributes": {
                "description": "Temporal symbol attributes with effective date ranges",
                "primary_key": ["symbol", "effective_date"],
                "temporal": True,
                "notes": "end_date=NULL means currently active. Trigger maintains end_dates.",
            },
            "metadata": {
                "description": "Schema version and configuration",
                "primary_key": ["key"],
            },
        },
    }
