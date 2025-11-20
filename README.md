# BogleBench

Portfolio performance analysis and benchmarking in the spirit of John Bogle.

Named after John C. Bogle, founder of Vanguard and champion of low-cost index investing, BogleBench helps investors analyze their portfolio performance against benchmarks using the principles Bogle advocated: long-term investing, low costs, and broad diversification.

## Features

- 📊 Portfolio performance analysis with risk-return metrics
- 📈 Individual asset performance tracking
- 🎯 Benchmark comparison (S&P 500, custom benchmarks)
- 📋 Comprehensive performance metrics (Sharpe ratio, Information ratio, etc.)
- 🔄 Automatic market data fetching
- 📓 Jupyter notebook integration
- 🛡️ Secure data separation from code
- 💰 Focus on Bogle's principles: simplicity, low costs, long-term perspective

## Installation

```bash
pip install boglebench
```

## Quick Start

### 1. Initialize Workspace

```bash
boglebench-init --path ~/my_boglebench_data
```

This creates the following structure:

```text
~/my_boglebench_data/
├── config/config.yaml          # Configuration file
├── transactions/               # Your transaction data
├── market_data/               # Cached market data
└── output/                    # Analysis results
```

### 2. Add Your Transaction Data

Add your transaction CSV file to `~/my_boglebench_data/transactions/`. Required columns:

### Required columns

- `date`: Transaction date in ISO8601 format (YYYY-MM-DD)
- `symbol`: Stock/ETF symbol
- `transaction_type`: BUY or SELL
- `shares`: Number of shares
- `price_per_share`: Price per share

### Optional columns

- `account`: Broker account name (defaults to "Default")
- `notes`: Free-form notes about the transaction

### Supported Columns for Dividend Transactions

When entering dividend transactions (`DIVIDEND` or `DIVIDEND_REINVEST`), you may optionally provide:

- `amount`: The total amount for the dividend transaction
- `dividend_per_share`: The per-share amount paid by the security (cash or equivalent).
- `dividend_type`: The type of dividend (e.g., CASH, REINVEST, SPECIAL).
- `dividend_pay_date`: The pay date of the dividend (defaults to `date` if not provided).
- `dividend_ex_date`: The ex-dividend date (optional, for auditing).
- `dividend_record_date`: The record date (optional, for auditing).

### How BogleBench Uses Dividend Data

- BogleBench will compare your provided dividends to the official data from AlphaVantage.
- It will sum all `amount` values for the same pay date/symbol and compare to AlphaVantage.
- If `dividend_per_share` is provided, it will also compare this value.
- If `dividend_type` is provided, it will check for mismatches.
- If you provide `dividend_ex_date` or `dividend_record_date`, they will be included in audit reports and output for your review.

**Important:** Dates must be in ISO8601 format `(YYYY-MM-DD)`. Examples of valid dates:
`2023-01-15`, `2024-12-31`. Invalid formats like `01/15/2023` or `15-01-2023` will
cause an error.

### Example Usage

See the included template in `boglebench/templates/sample_transactions.csv`.

### 3. Symbol Attributes (Optional but Recommended)

Symbol attributes (asset class, geography, sector, etc.) are managed separately from transactions and enable attribution analysis.

Create an attributes CSV file with these columns:

- `symbol`: Stock/ETF ticker (required)
- `effective_date`: Date these attributes become effective in ISO8601 format (YYYY-MM-DD) (required)
- `asset_class`: e.g., "Equity", "Bond", "Real Estate" (optional)
- `geography`: e.g., "US", "International", "Emerging Markets" (optional)
- `region`: e.g., "North America", "Europe", "Asia Pacific" (optional)
- `sector`: e.g., "Technology", "Healthcare", "Financials" (optional)
- `style`: e.g., "Growth", "Value", "Blend" (optional)
- `market_cap`: e.g., "Large", "Mid", "Small" (optional)
- `fund_type`: e.g., "ETF", "Mutual Fund", "Stock" (optional)

Load attributes after building portfolio history:

```python
analyzer.build_portfolio_history()
analyzer.load_symbol_attributes(csv_path='path/to/symbol_attributes.csv')
```

Attributes support temporal tracking - you can load multiple versions with different effective dates to track changes over time.

See the included template in `boglebench/templates/sample_attributions.csv`.

### 4. Run Analysis

```python
from boglebench import BogleBenchAnalyzer

# Initialize analyzer
analyzer = BogleBenchAnalyzer(config_path="~/my_boglebench_data/config/config.yaml")

# Load and analyze
analyzer.load_transactions()
analyzer.build_portfolio_history()

# Optional: Load symbol attributes for attribution analysis
analyzer.load_symbol_attributes(csv_path='path/to/symbol_attributes.csv')

results = analyzer.calculate_performance()

# View results
print(results.summary())
```

Or use the command line:

```bash
boglebench-analyze --config ~/my_boglebench_data/config/config.yaml
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  base_path: "~/my_boglebench_data"
  transactions_file: "transactions/my_transactions.csv"

settings:
  benchmark_symbol: "SPY"
  risk_free_rate: 0.02
  default_currency: "USD"

validation:
  # Short position handling - BogleBench does not support short positions
  short_position_handling: "reject"  # Options: "reject" or "cap"

analysis:
  performance_metrics:
    - total_return
    - annualized_return
    - volatility
    - sharpe_ratio
    - max_drawdown
    - information_ratio
```

### Short Position Handling

BogleBench does not support short positions (negative holdings). When a transaction would result in a short position, you can configure how the system responds:

- **`reject`** (default): Raises an error and prevents the transaction from being processed. This ensures data integrity and prevents invalid portfolio states.
- **`cap`**: Automatically adjusts the transaction quantity to the maximum allowed (equal to current holdings), resulting in zero holdings after the transaction. A warning is logged for each capped transaction.
- **`ignore`**: Logs a warning but allows the short position to occur. Use with caution as portfolio metrics may be incorrect for short positions.

Example scenarios:

- **Scenario**: You own 100 shares of AAPL and attempt to sell 150 shares
  - **reject mode**: Raises `ShortPositionError` with details about the invalid transaction
  - **cap mode**: Adjusts the sale to 100 shares, resulting in zero holdings, and logs a warning
  - **ignore mode**: Allows the sale of 150 shares, resulting in -50 shares (short position), and logs a warning

Configure this in your `config.yaml`:

```yaml
validation:
  short_position_handling: "reject"  # or "cap" or "ignore"
```

## Philosophy

BogleBench embodies John Bogle's investment philosophy:

- **Simplicity**: Clear, straightforward analysis without unnecessary complexity
- **Low Costs**: Focus on cost-efficient investing and fee impact analysis
- **Long-term Perspective**: Emphasis on long-term performance over short-term fluctuations
- **Broad Diversification**: Analysis tools for well-diversified portfolios
- **Stay the Course**: Consistent, disciplined approach to portfolio evaluation

## Requirements

- Python 3.13+
- pandas
- numpy
- alpha-vantage
- matplotlib
- seaborn
- scipy

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Notes on Calculations

This is intended for the analysis of personal portfolios, not for professional use.
As such there are a few areas where the calculation methodologies differ from what
may be found in the realm of professional fund management.

### Returns

Portfolio returns can be calculated using several industry-standard methods, such as **Modified Dietz**, **Time-Weighted Return (TWR)**, and **Internal Rate of Return (IRR)** (also called **money-weighted return**).

The key distinction between these methods lies in their treatment of **external vs. internal cash flows**. In professional fund management, **external cash flows** refer to contributions and withdrawals of capital between the investor and the portfolio (e.g., deposits, redemptions). These flows are included in performance calculations for Modified Dietz, TWR, and IRR because they represent changes in the amount of capital at work that are outside the manager’s control.

By contrast, **internal cash flows** (e.g., dividends, interest, or proceeds from asset sales that remain in the portfolio) are not treated as external and are typically reflected in the change in portfolio value.

This package assumes that the publicly listed assets being analyzed represent only a portion of the total portfolio. Therefore, internally generated cash flows (such as dividends) are treated as if they were immediately reinvested or withdrawn — effectively converting them into external flows for return calculation purposes. In other words, all internally generated cash flows are considered portfolio contributions or distributions, ensuring they are fully captured in the reported return.

---

### Example (Modified Dietz with Internal Flows)

$$
R = \frac{V_{E} - V_{B} - \sum{CF}}{V_{B} + \sum{\left(W_i \times CF_i\right)}}
$$

Where:

- $V_{B}$ = beginning portfolio value
- $V_{E}$ = ending portfolio value
- $CF_i$ = each cash flow during the period (including dividends)
- $W_i$ = weight for each cash flow, based on timing within the period
- $R$ = portfolio return

---

### Numerical Example – Modified Dietz

- Beginning Value ($V_{B}$): **10,000**
- Ending Value ($V_{E}$): **10,200**
- Dividend ($CF_1$): **+100** (received halfway through period, so $W_1 = 0.5$)

$$
R = \frac{10{,}200 - 10{,}000 - 100}{10{,}000 + (0.5 \times 100)}
= \frac{100}{10{,}050}
= 0.00995 \quad (0.995\%)
$$

If the dividend were excluded as an external flow, the return would instead be:

$$
R = \frac{10{,}200 - 10{,}000}{10{,}000}
= 0.02 \quad (2.0\%)
$$

This illustrates that including internal cash flows as external flows _reduces_ the reported return, because the dividend is treated as a distribution that leaves the portfolio rather than simply increasing ending value.

---

### Example (Time-Weighted Return)

TWR breaks the performance measurement into **subperiods** whenever there is a cash flow, then geometrically links the returns across periods. This eliminates the effect of cash flow timing.

#### Setup

- Beginning Value: **10,000**
- Dividend: **+100** at midpoint
- Ending Value: **10,200**

#### Step 1: Split into Subperiods

| Period              | Start Value             | Cash Flow | End Value | Subperiod Return                                                |
| ------------------- | ----------------------- | --------- | --------- | --------------------------------------------------------------- |
| 1 (before dividend) | 10,000                  | +0        | 10,050    | $ r_1 = \frac{10{,}050 - 10{,}000}{10{,}000} = 0.0050 $ (0.50%) |
| 2 (after dividend)  | 10,150 _(10,050 + 100)_ | +0        | 10,200    | $ r_2 = \frac{10{,}200 - 10{,}150}{10{,}150} = 0.0049 $ (0.49%) |

#### Step 2: Geometrically Link Returns

$$
R_{TWR} = (1 + r_1) \times (1 + r_2) - 1
= (1.0050 \times 1.0049) - 1
= 0.00995 \quad (0.995\%)
$$

The TWR result matches the Modified Dietz calculation when flows are small and evenly spaced, but is unaffected by the size or timing of the cash flow — making it the preferred measure for comparing manager skill across accounts with different cash flow patterns.

---

### Note on IRR (Money-Weighted Return)

IRR (or money-weighted return) is the discount rate that sets the **net present value of all cash flows (including beginning and ending value)** to zero. It is useful when the investor controls the timing and size of cash flows — for example, in **private equity**, **venture capital**, or **portfolios with large irregular contributions/withdrawals** — because it reflects the actual dollar-weighted experience of the investor. However, IRR is not ideal for comparing performance across managers, since results are heavily influenced by cash flow timing decisions.

---

### Summary

- **Modified Dietz**: Approximates IRR; easier to calculate; sensitive to flow timing.
- **TWR**: Breaks into subperiods; removes impact of external flows; preferred for manager evaluation.
- **IRR**: True money-weighted return; reflects investor experience; best for portfolios with irregular flows or where timing decisions matter.
- **This package**: Treats internal flows as external, ensuring that dividends and interest are explicitly included in return calculations.