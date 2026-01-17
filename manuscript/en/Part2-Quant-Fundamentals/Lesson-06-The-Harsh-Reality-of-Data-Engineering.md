# Lesson 06: The Harsh Reality of Data Engineering

> **Data problems kill more strategies than model problems.**

---

## A Nightmare Weekend

Friday night at 11 PM, Xiao Wang deployed his carefully developed US stock quantitative strategy. Backtested 80% annual return, Sharpe ratio 2.5, everything looked perfect.

Monday morning at 9 AM, he was woken by an alert message: "Strategy error, unable to fetch data."

He got up to check and found Alpha Vantage API returning **429 Too Many Requests**. His program was requesting 10 times per minute, exceeding the free account's API rate limit (5 times per minute).

After fixing it, the strategy finally started running.

Monday afternoon at 2 PM, another alert: "Strategy position anomaly."

This time it was a data problem: The API returned a **missing minute bar**, causing MACD to calculate NaN, and the program treated NaN as a sell signal, clearing all positions.

After fixing that, he decided to do a comprehensive data check. Then he discovered more problems:

- Some timestamps were UTC, others were ET (Eastern Time)
- Some bars had 0 volume (pre-market/after-hours sessions)
- Historical data and real-time data had different field names (`close` vs `Close`)

**For an entire week, he didn't write a single line of strategy code - all spent fixing data problems.**

This is the harsh reality of data engineering.

---

## 6.1 Data Source Selection

### Free Data Sources

| Data Source | Market Coverage | Pros | Cons |
|-------------|-----------------|------|------|
| **Yahoo Finance** (yfinance) | US stocks, ETFs | Free, long history | **Recent stability issues reported** (API changes/rate limits/missing data); build robust fallbacks |
| **Alpaca Markets** | US stocks | Free IEX data, developer-friendly API | Free tier has limited data coverage |
| **Binance API** | Cryptocurrency | Real-time, free | Only Binance data |
| **Alpha Vantage** | Stocks, forex, crypto | Free tier available | Free-tier quotas are often very low (limits change; verify on the official site) |
| **CCXT** | Cryptocurrency | Unified interface | Depends on exchange APIs |

> **Note**: Data vendors may change products, APIs, quotas, pricing, or even discontinue services. Treat data sources as having “delisting risk” and always have backups.

### Paid Data Sources (2024-2025 Reference Pricing)

| Data Source | Market Coverage | Monthly Cost | Features |
|-------------|-----------------|--------------|----------|
| **Bloomberg** | Global | **$2,500+** | Institutional standard, highest quality |
| **LSEG Workspace** (formerly Refinitiv) | Global | Quote-based pricing | LSEG subsidiary, former Reuters data |
| **Polygon.io** (Massive) | US stocks | **$99+** | Real-time + historical, developer-friendly |
| **Nasdaq Data Link** (formerly Quandl) | Alternative data | Per dataset pricing | Satellite, consumer data, etc. |

### How to Choose?

```
Personal learning/research:
    -> Free data sources + tolerate data issues

Small fund (< $1M AUM):
    -> Paid data (basic tier) + build data validation

Institutional (> $10M AUM):
    -> Bloomberg/Refinitiv + multi-source cross-validation
```

**Core principle**: **Data quality determines strategy ceiling**. Garbage in, garbage out.

---

## 6.2 The Painful Reality of APIs

### The Code You Think You'll Write vs Reality

**You think**:
```python
data = api.get_history("AAPL", days=365)
```

**Reality**:

<details>
<summary>Pseudocode Reference (replace with specific SDK like yfinance / alpaca / ccxt)</summary>

```python
# Note: This is pseudocode showing robust data fetching design patterns
# api, log, EmptyDataError, RateLimitError need to be replaced with your specific library

import time
from typing import Optional
import pandas as pd

def get_history_robust(
    symbol: str,
    days: int,
    max_retries: int = 5,
    backoff_base: float = 2.0
) -> Optional[pd.DataFrame]:
    """
    Robust data fetching function

    Problems handled:
    - Rate Limiting (exponential backoff retry)
    - Empty data (validation + alerting)
    - Connection errors (auto-reconnect)
    - Timeouts (reasonable timeout settings)
    """
    for attempt in range(max_retries):
        try:
            # Set timeout (replace with your API call)
            data = api.get_history(
                symbol,
                days=days,
                timeout=30
            )

            # Validate data (generic logic)
            if data is None:
                raise EmptyDataError(f"No data for {symbol}")

            if len(data) == 0:
                raise EmptyDataError(f"Empty data for {symbol}")

            if len(data) < days * 0.9:  # Less than 90% of expected data
                log.warning(f"Data incomplete: got {len(data)}, expected ~{days}")

            # Check for anomalies
            if data['close'].isnull().any():
                log.warning("NaN values detected, filling...")
                data['close'] = data['close'].ffill()

            return data

        except RateLimitError:
            wait_time = backoff_base ** attempt
            log.info(f"Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)

        except ConnectionError:
            log.warning("Connection lost, reconnecting...")
            api.reconnect()
            time.sleep(1)

        except TimeoutError:
            log.warning("Request timeout, retrying...")

        except Exception as e:
            log.error(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                raise

    return None
```

</details>

### Common API Problems

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Rate Limiting** | 429 errors | Exponential backoff + request queue |
| **Missing Data** | Empty response or partial data | Multi-source backup + missing detection |
| **Data Delay** | Timestamps lagging | Record delay + adjust strategy |
| **Inconsistent Format** | Field names change | Unified data adapter layer |
| **Unstable Connection** | Disconnects, timeouts | Heartbeat detection + auto-reconnect |

### The Real Cost of Rate Limiting

Suppose you need minute bars for 100 symbols:

| API Limit | Completion Time |
|-----------|-----------------|
| 1 request/second | 100 seconds |
| 10 requests/second | 10 seconds |
| Unlimited | < 1 second |

**High-frequency strategies need high API quotas** - this is the core value of paid data sources.

---

## 6.3 Time Alignment Issues

### Timezone Chaos

Different data sources use different timezones:

| Data Source | Default Timezone |
|-------------|------------------|
| Binance | UTC |
| Yahoo Finance | Exchange local time |
| China A-shares | UTC+8 |
| US stocks | ET (Eastern Time) |

**If not handled**:
- 9:30 A-share data + 9:30 US stock data -> Actually 12 hours apart
- Strategy uses "same moment" data, but they're actually different moments

**Solution**: **Convert everything to UTC**, store and compute in UTC.

```python
import pandas as pd

def normalize_timezone(df, source_tz='UTC'):
    """Standardize to UTC"""
    if df.index.tz is None:
        df.index = df.index.tz_localize(source_tz)
    df.index = df.index.tz_convert('UTC')
    return df
```

### Tick to Candlestick Aggregation

When aggregating tick data to candlesticks, note:

| Issue | Explanation |
|-------|-------------|
| Open price | First trade price in that period |
| Close price | Last trade price in that period |
| High price | Highest trade price in that period |
| Low price | Lowest trade price in that period |
| Volume | Sum of all trades in that period |

**Trap**: What if there's no trade in a given minute?

```
Option 1: Fill OHLC with previous minute's close price
Option 2: Mark as missing, handle later
Option 3: Exclude that time point from aggregation
```

Different approaches lead to different indicator calculations. **Must standardize handling rules**.

### Cross-Asset Data Alignment

Stocks, futures, forex have different trading hours:

| Asset | Trading Hours |
|-------|---------------|
| China A-shares | 9:30-11:30, 13:00-15:00 |
| US stocks | 9:30-16:00 ET |
| Forex | Near 24/7 |
| Cryptocurrency | 24/7 |

**If strategy needs cross-asset signals**:
- When A-shares close, US hasn't opened yet
- Use A-share close + US open? There's a time gap
- Need careful "synchronization point" design

---

## 6.4 Data Quality Issues

### Anomaly Detection and Handling

| Anomaly Type | Detection Method | Handling |
|--------------|-----------------|----------|
| Price jumps | Price change > 20% | Check if real (stock split?) |
| Volume anomaly | 0 or extreme values | Check exchange status |
| Missing values | NaN | Forward fill or delete |
| Duplicate data | Duplicate timestamps | Keep first or last |

```python
def detect_anomalies(df, price_col='close', volume_col='volume'):
    """Detect data anomalies"""
    issues = []

    # Price jumps
    returns = df[price_col].pct_change()
    jumps = returns.abs() > 0.20
    if jumps.any():
        issues.append(f"Price jumps detected: {jumps.sum()} times")

    # Zero volume
    zero_volume = df[volume_col] == 0
    if zero_volume.any():
        issues.append(f"Zero volume: {zero_volume.sum()} bars")

    # Missing values
    nulls = df.isnull().sum()
    if nulls.any():
        issues.append(f"Null values: {nulls.to_dict()}")

    # Duplicate timestamps
    duplicates = df.index.duplicated()
    if duplicates.any():
        issues.append(f"Duplicate timestamps: {duplicates.sum()}")

    return issues
```

### Stock Dividend and Split Adjustments

Stock historical prices need "adjustment":

| Event | Actual Change | Unadjusted Data | Adjusted Data |
|-------|---------------|-----------------|---------------|
| 2-for-1 split | Shares double, price halves | Price cliff before/after | Smooth and continuous |
| Dividend | Ex-dividend deduction | Gap on ex-div date | Historical prices adjusted |

**Consequences of not adjusting**:
- In backtests, strategy generates false signals on ex-dates
- Return calculations show extreme values

### Futures Roll Handling

Futures contracts expire and need "rolling":

```
January 2024: Hold IF2401 (January contract)
Before January expiry: Switch to IF2402 (February contract)

Problem: IF2401 closes at 4000, IF2402 closes at 4050
      Not real price movement, but different contracts
```

**Solutions**:
- Calculate spread when splicing, adjust historical prices
- Or use "continuous main contract" data (pre-processed by data vendor)

---

## 6.5 Survivorship Bias

### What is Survivorship Bias?

If you only backtest with "stocks that still exist today":

```
2010 stock pool: 1000 stocks
Still exist in 2024: 800 stocks
Delisted/bankrupt: 200 stocks

Your backtest only uses 800, ignoring those 200 failed companies
-> Backtest returns are inflated
```

### How Serious is Survivorship Bias?

| Research | Conclusion |
|----------|------------|
| Academic studies | Annual returns inflated 1-3% |
| Value investing strategies | Bias more severe (cheap stocks more likely to delist) |
| Small-cap strategies | Bias is largest |

### How to Avoid?

| Method | Difficulty | Effectiveness |
|--------|------------|---------------|
| Use database including delisted stocks | High | Most accurate |
| Use historical index constituents | Medium | Fairly accurate |
| Acknowledge bias in conclusions | Low | At least honest |

**Paid data sources typically provide "survivorship-bias-free" datasets** - another value of paying.

---

## 6.6 Alternative Data Introduction

### What is Alternative Data?

Traditional data: Prices, volume, financial statements
Alternative data: **Everything else**

| Type | Data Source | Application |
|------|-------------|-------------|
| **Satellite data** | Parking lot car counts, oil tank levels | Predict retail earnings, oil inventory |
| **Text data** | News, social media, earnings calls | Sentiment analysis, event-driven |
| **Credit card data** | Spending statistics | Predict company revenue |
| **Web traffic** | Website visits | Predict e-commerce performance |
| **GPS data** | Phone locations | Foot traffic analysis |

### Challenges of Alternative Data

| Challenge | Explanation |
|-----------|-------------|
| **High noise** | Signal/noise ratio far lower than price data |
| **Compliance risk** | Privacy issues, data source legality |
| **Fast alpha decay** | Once widely used, advantage disappears |
| **High cost** | Data itself expensive + processing costs |

### Multi-Agent Perspective

![Data Pipeline Architecture](assets/data-pipeline-architecture.svg)

---

## 6.7 Data Pipeline Design Principles

### Three Principles

| Principle | Explanation |
|-----------|-------------|
| **Immutability** | Never modify raw data, generate new files after processing |
| **Traceability** | Record source, processing time, processing version for each data point |
| **Redundant backup** | Cross-validate with at least two data sources |

### Data Pipeline Architecture

```
Data Collection Layer
    |
    |-> Raw Data Storage (immutable)
    |         |
    |         v
    |-> Data Cleaning Layer
    |     - Anomaly handling
    |     - Missing value filling
    |     - Timezone standardization
    |         |
    |         v
    |-> Feature Engineering Layer
    |     - Technical indicator calculation
    |     - Label generation
    |         |
    |         v
    --> Ready Data Storage -> Strategy Use
```

### Monitoring and Alerting

| Monitor Item | Suggested Threshold | Alert Action |
|--------------|---------------------|--------------|
| Data delay | > 1 minute | Warning |
| Missing data rate | > 1% | Warning |
| Anomaly ratio | > 0.1% | Investigate |
| API error rate | > 5% | Pause strategy |

---

## Code Implementation (Optional)

### Data Quality Check Framework

```python
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class DataQualityReport:
    symbol: str
    start_date: str
    end_date: str
    total_rows: int
    missing_rows: int
    null_values: dict
    anomalies: List[str]
    is_valid: bool

def check_data_quality(df: pd.DataFrame, symbol: str) -> DataQualityReport:
    """Comprehensive data quality check"""

    anomalies = []

    # Check time continuity
    # Note: This example assumes *daily* bars and uses business days as a rough expectation.
    # For minute/tick data, you must generate expected_index based on exchange trading sessions,
    # holidays, lunch breaks (if any), etc.
    idx = pd.DatetimeIndex(df.index)
    expected_index = pd.bdate_range(start=idx.min().normalize(), end=idx.max().normalize())
    expected_rows = len(expected_index)
    actual_rows = len(df)
    missing_rows = expected_rows - actual_rows

    if missing_rows > expected_rows * 0.05:
        anomalies.append(f"High missing rate: {missing_rows/expected_rows:.1%}")

    # Check NULL values
    null_counts = df.isnull().sum().to_dict()

    # Check price jumps
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        extreme_moves = (returns.abs() > 0.2).sum()
        if extreme_moves > 0:
            anomalies.append(f"Extreme price moves: {extreme_moves}")

    # Check volume
    if 'volume' in df.columns:
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            anomalies.append(f"Zero volume periods: {zero_volume}")

    # Check timestamp duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        anomalies.append(f"Duplicate timestamps: {duplicates}")

    return DataQualityReport(
        symbol=symbol,
        start_date=str(df.index[0]),
        end_date=str(df.index[-1]),
        total_rows=actual_rows,
        missing_rows=missing_rows,
        null_values=null_counts,
        anomalies=anomalies,
        is_valid=len(anomalies) == 0
    )
```

---

## Lesson Deliverables

After completing this lesson, you will have:

1. **Deep understanding of data problems** - Know that data engineering is the main battlefield of quant
2. **API best practices** - Robust data fetching code framework
3. **Data quality checking ability** - Can identify and handle common data issues
4. **Data pipeline design thinking** - Understand production-grade data system architecture

---

## Key Takeaways

- [x] Understand the trade-offs between free and paid data sources
- [x] Master methods for handling API rate limiting, missing data, etc.
- [x] Recognize hidden issues like timezone chaos, survivorship bias
- [x] Understand the value and challenges of alternative data
- [x] Understand data pipeline design principles

---

## Extended Reading

- [Background: Exchanges and Order Book Mechanics](Background/Exchanges-and-Order-Book-Mechanics.md) - Source of tick data
- [Background: Cryptocurrency Trading Characteristics](Background/Cryptocurrency-Trading-Characteristics.md) - Challenges of 24/7 data

---

## Next Lesson Preview

**Lesson 07: Backtest System Pitfalls**

Even if your data is fine, backtests can still deceive you. Look-Ahead Bias, overfitting, ignoring trading costs... These traps have caused countless seemingly perfect strategies to fail miserably in live trading. Next lesson reveals the truth about backtesting.
