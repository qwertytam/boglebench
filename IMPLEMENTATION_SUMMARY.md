# Short Position Handling - Implementation Summary

## Overview

This implementation adds logic to detect and handle transactions that would result in short positions (negative holdings) in BogleBench. Since BogleBench does not support short positions, users can now choose how to handle such scenarios through configuration.

## Configuration

Add the following to your `config.yaml`:

```yaml
validation:
  short_position_handling: "reject"  # Options: "reject" or "cap"
```

## Modes

### REJECT Mode (Default)

When a transaction would result in a short position, the system raises a `ShortPositionError` with detailed information:

```python
from boglebench import BogleBenchAnalyzer, ShortPositionError

try:
    analyzer = BogleBenchAnalyzer(config_path="config.yaml")
    analyzer.load_transactions()
except ShortPositionError as e:
    print(f"Short position detected: {e}")
    # Transaction would result in short position:
    #   Date: 2023-11-15
    #   Account: Schwab_401k
    #   Symbol: AAPL
    #   Current position: 125.00 shares
    #   Transaction quantity: -200.00 shares
    #   Would result in: -75.00 shares
```

**Use Case**: When you want strict validation and need to correct your transaction data manually.

### CAP Mode

When a transaction would result in a short position, the system automatically adjusts the transaction quantity to the maximum allowed (equal to current holdings):

```python
analyzer = BogleBenchAnalyzer(config_path="config.yaml")  # config has cap mode
transactions = analyzer.load_transactions()
# ⚠️  Short position detected and capped:
#    Date: 2023-11-15
#    Account: Schwab_401k
#    Symbol: AAPL
#    Original transaction: -200.00 shares
#    Current position: 125.00 shares
#    Adjusted transaction: -125.00 shares (resulting in 0 shares)
```

**Use Case**: When you want automatic correction for overselling scenarios, especially useful when:
- You have approximate transaction records
- You want to close out positions without exact share counts
- You're importing data from brokers that may have rounding differences

## Technical Details

### Files Changed

1. **boglebench/core/constants.py**
   - Added `ShortPositionHandling` enum with REJECT and CAP modes

2. **boglebench/core/short_position_handler.py** (new)
   - `ShortPositionError`: Exception raised when short position detected in REJECT mode
   - `ShortPositionHandler`: Class that detects and adjusts transactions
   - `process_transactions_with_short_check()`: Batch processing function

3. **boglebench/core/portfolio.py**
   - Updated `load_transactions()` to call short position checking
   - Added import for `ShortPositionError`

4. **boglebench/__init__.py**
   - Exported `ShortPositionError` for user access

5. **boglebench/templates/config_template.yaml**
   - Added `validation.short_position_handling` configuration option

6. **README.md**
   - Added documentation section for short position handling
   - Removed "Known Bugs" section about short positions

### Testing

**Unit Tests** (14 tests in `test_short_position_handler.py`):
- Handler initialization and validation
- Detection of short positions
- REJECT mode error raising
- CAP mode transaction adjustment
- Multi-account independence
- Edge cases (exact zero, same-day transactions)

**Integration Tests** (4 tests in `test_short_position_integration.py`):
- Full workflow with BogleBenchAnalyzer
- Normal transactions (no short positions)
- REJECT mode with real config
- CAP mode with real config
- Default behavior

**Manual Validation** (`test_manual_validation.py`):
- Uses actual sample transactions with known short position
- Tests both REJECT and CAP modes end-to-end
- Results: ✅ All tests passed

### Algorithm

1. Transactions are processed chronologically
2. For each transaction that changes quantity (BUY, SELL, etc.):
   - Track current holdings per account per symbol
   - Calculate resulting position after transaction
   - If resulting position < 0 (with small epsilon for floating point):
     - **REJECT mode**: Raise `ShortPositionError`
     - **CAP mode**: Adjust transaction quantity to `-current_position`
3. Continue with adjusted (or original) transactions

### Performance Impact

- Minimal: O(n) single pass through transactions
- Memory: O(accounts × symbols) to track holdings
- Typically processes 1000s of transactions in milliseconds

## Example Scenario

**Original Transaction Data:**
```csv
date,symbol,transaction_type,quantity,value_per_share,total_value,account
2023-01-15,AAPL,BUY,100,150.00,15000.00,Test
2023-02-15,AAPL,BUY,50,155.00,7750.00,Test
2023-11-15,AAPL,SELL,200,186.00,37200.00,Test
```

**Problem:** After first two transactions, you own 150 shares. Selling 200 would result in -50 shares (short position).

**REJECT Mode Result:**
```
ShortPositionError: Transaction would result in short position:
  Current position: 150.00 shares
  Transaction quantity: -200.00 shares
  Would result in: -50.00 shares
```

**CAP Mode Result:**
```
Transaction adjusted from SELL 200 to SELL 150
Final position: 0 shares (no short position)
```

## Migration Guide

### For Existing Users

If you have transaction data that may contain short positions:

1. **First, try REJECT mode** (default):
   ```yaml
   validation:
     short_position_handling: "reject"
   ```
   This will identify all problematic transactions.

2. **Review the errors**: Check if they're data entry errors or intentional.

3. **Choose your approach**:
   - Fix the data manually, OR
   - Use CAP mode to automatically adjust

### For New Users

The default REJECT mode is recommended for data quality. Only use CAP mode if you specifically need automatic adjustment.

## Security Considerations

- ✅ No security vulnerabilities detected by CodeQL
- ✅ All validation happens before data is persisted
- ✅ Clear error messages prevent silent data corruption
- ✅ Comprehensive logging for audit trail

## Future Enhancements

Potential future additions (not in this PR):
- Add a third mode: WARN (log warning but allow short positions)
- Track short position history for reporting
- Support for shorting as a first-class feature (major change)

## Support

For issues or questions:
1. Check the configuration documentation in `config_template.yaml`
2. Review error messages - they contain detailed context
3. Check logs for warnings in CAP mode
4. File an issue on GitHub with your configuration and error details
