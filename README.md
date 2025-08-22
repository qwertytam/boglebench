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

- `date`: Transaction date (YYYY-MM-DD)
- `ticker`: Stock/ETF symbol  
- `transaction_type`: BUY or SELL
- `shares`: Number of shares
- `price_per_share`: Price per share
- `account`: Broker account name (optional, defaults to "Default")

### 3. Run Analysis

```python
from boglebench import BogleBenchAnalyzer

# Initialize analyzer
analyzer = BogleBenchAnalyzer(config_path="~/my_boglebench_data/config/config.yaml")

# Load and analyze
analyzer.load_transactions()
analyzer.fetch_market_data()
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
  benchmark_ticker: "SPY"
  risk_free_rate: 0.02
  default_currency: "USD"

analysis:
  performance_metrics:
    - total_return
    - annualized_return
    - volatility
    - sharpe_ratio
    - max_drawdown
    - information_ratio
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
- yfinance
- matplotlib
- seaborn
- scipy

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
