# Background: US vs China Quantitative Market Differences

> A-shares (China) and US stocks have significant differences in trading rules, market structure, and data availability. Understanding these differences is a prerequisite for developing cross-market strategies.

---

## 1. Trading Rules Comparison

| Rule | A-Shares (China) | US Stocks | Hong Kong |
|------|-----------------|-----------|-----------|
| Trading Hours | 9:30-11:30, 13:00-15:00 | 9:30-16:00 ET | 9:30-12:00, 13:00-16:00 |
| Settlement | T+1 | T+1 (since May 2024) | T+2 |
| Price Limits | ±10% (ChiNext ±20%) | No limit | No fixed limits (VCM mechanism) |
| Short Selling | Securities lending (many restrictions) | Relatively free | Designated securities only, naked shorts prohibited |
| Minimum Tick | ¥0.01 | $0.01 | Dynamic (varies by price range) |
| Trading Unit | Multiples of 100 shares | 1 share | Set by issuer (lot size varies) |
| Pre/Post Market | Call auction | Pre/post market trading | Pre-opening session and closing auction |
| Main Fees | Stamp tax 0.05% (sell side) | Very low or zero commission | Stamp tax 0.1% (both sides) |

---

## 2. T+1 Strategy Impact

### 2.1 China A-Shares T+1

**Rule**: Stocks bought today can only be sold the next day

**Impact**:
- Cannot do intraday swings
- Cannot quickly stop loss
- Cannot avoid overnight risk

**Strategy Response**:
```python
# A-share strategies must consider overnight positions
def should_buy(signal, overnight_risk):
    if overnight_risk > threshold:
        return False  # Risk too high, don't open position
    return signal > 0
```

### 2.2 US Stock Day Trading

**Rules**:
- Pattern Day Trader (PDT) rule
- Account < $25,000: Maximum 3 day trades in 5 trading days
- Account ≥ $25,000: No restrictions
- *Note: FINRA has approved changes to PDT rules, planning to replace fixed amount limits with more flexible risk margins (pending SEC final approval)*

**Advantages**:
- Can do intraday strategies
- Quick stop losses
- Utilize intraday volatility

---

## 3. Price Limit System

### 3.1 A-Share Price Limits

| Board | Price Limit Range |
|-------|-------------------|
| Main Board (SSE/SZSE) | ±10% |
| ChiNext (创业板) | ±20% (since Aug 2020; no limits first 5 days after listing) |
| STAR Market (科创板) | ±20% (since Jul 2019; no limits first 5 days after listing) |
| Beijing Stock Exchange (北交所) | ±30% (since Nov 2021) |
| ST Stocks | ±5% |

### 3.2 Hong Kong Volatility Control Mechanism (VCM)

**Rules**:
- Designed to mitigate extreme intraday volatility.
- HSI/HSCEI constituents: ±10% (5-minute price deviation).
- After triggering, enters 5-minute cooling period with fixed price range trading.

**Impact**:
- Consecutive limit up/down makes execution impossible
- Cannot buy at limit up (liquidity disappears)
- Cannot sell at limit down (passive stop loss fails)

**Strategy Considerations**:
```python
# Price limit detection
def is_limit_up(price, prev_close, limit=0.10):
    return price >= prev_close * (1 + limit - 0.001)

def is_limit_down(price, prev_close, limit=0.10):
    return price <= prev_close * (1 - limit + 0.001)
```

### 3.3 US Stock Circuit Breakers

**Index Circuit Breakers**:
- Level 1 (7%): 15-minute halt
- Level 2 (13%): 15-minute halt
- Level 3 (20%): Trading halted for the day

**Individual Stock Circuit Breakers** (LULD):
- 5-minute halt when price deviation is too large

---

## 4. Short Selling Mechanisms

### 4.1 A-Share Securities Lending

**Restrictions**:
- Must open margin trading account
- Limited securities available, hard to borrow popular stocks
- High borrowing costs (8-10% annualized)
- Some stocks prohibited from lending

**Practical Impact**:
- Short strategies difficult to implement
- Market neutral strategies have high costs
- Limited hedging tools

### 4.2 US Stock Short Selling

**Process**:
1. Borrow shares
2. Sell them
3. Buy back to return

**Costs**:
- Borrow fee (0.3% - 50%+ annualized)
- Dividend compensation

**Easy-to-Borrow vs Hard-to-Borrow**:
- Large caps easy to borrow
- Small caps, popular short targets hard to borrow and expensive

---

## 5. Data Availability

### 5.1 A-Share Data

| Data Type | Source | Cost |
|-----------|--------|------|
| Daily quotes | Tushare, AKShare | Free |
| Minute data | Tushare Pro | Points system/sponsorship (~¥200+) |
| Level-2 | Brokers, Wind | ¥10000+/year |
| Financial data | Tushare, Wind | Basic free to deep paid |
| Alternative data | Third parties | High cost |

**Free Data Sources**:
- Tushare Pro: https://tushare.pro (points system, active community can get free access)
- AKShare: https://akshare.xyz (open source free, rich interfaces)
- BaoStock: http://baostock.com

### 5.2 US Stock Data

| Data Type | Source | Cost |
|-----------|--------|------|
| Daily quotes | Yahoo Finance | Free |
| Minute data | Alpha Vantage | Free/Paid |
| Level-2 | Polygon.io | $29-199/month |
| Financial data | SEC EDGAR | Free |

---

## 6. Market Participant Structure

### 6.1 A-Shares

| Participant | Share | Characteristics |
|-------------|-------|-----------------|
| Retail | ~80%+ (trading volume) | Short-term trading, emotion-driven, ~20% of holdings |
| Institutional | ~20% (trading volume) | Mutual funds, private funds, insurance, foreign (growing) |

**Impact**:
- High volatility
- Strong momentum effects
- Emotion-driven price deviations

### 6.2 US Stocks

| Participant | Share | Characteristics |
|-------------|-------|-----------------|
| Institutional | ~70-80% | Pensions, mutual funds, hedge funds, HFT (50%+ of volume) |
| Retail | ~20-25% | Increased recently due to Robinhood etc., strong "buy the dip" tendency |

**Impact**:
- Relatively rational
- Factors more persistently effective
- High passive investment share

---

## 7. Strategy Differences

### 7.1 Effective A-Share Strategies

| Strategy | Effectiveness | Reason |
|----------|---------------|--------|
| Momentum | Strong | Retail herding, limit-up effect |
| Small cap | Strong | Shell value, liquidity premium |
| Reversal | Medium | Correction after overreaction |
| Value | Weak-Medium | Retail prefer growth |

### 7.2 Effective US Stock Strategies

| Strategy | Effectiveness | Reason |
|----------|---------------|--------|
| Value | Medium | Long-term effective but cyclical |
| Momentum | Medium | Diluted by institutional trading |
| Quality | Strong | Long-term stable |
| Low volatility | Strong | Good risk-adjusted returns |

---

## 8. Technical Implementation Differences

### 8.1 Backtesting Considerations

**A-Shares**:
```python
# Factors A-share backtesting must consider
class ChinaBacktester:
    def __init__(self):
        self.t_plus_1 = True  # T+1 restriction
        self.limit_up_down = 0.10  # Price limits
        self.min_lot = 100  # Minimum trading unit

    def can_sell(self, position, trade_date):
        # Check if T+1 is satisfied
        return position.buy_date < trade_date

    def check_tradeable(self, price, prev_close):
        # Check for price limits
        if self.is_limit_up(price, prev_close):
            return False  # Cannot buy at limit up
        if self.is_limit_down(price, prev_close):
            return False  # Cannot sell at limit down
        return True
```

**US Stocks**:
```python
# US stock backtesting is simpler
class USBacktester:
    def __init__(self):
        self.t_plus_0 = True  # Can day trade
        self.fractional_shares = True  # Fractional shares allowed
```

### 8.2 Live Trading Interfaces

**A-Shares**:
- Broker proprietary APIs (requires application)
- PTrade, QMT and other programmatic interfaces
- Third-party interfaces (gray area)

**US Stocks**:
- Interactive Brokers API
- Alpaca API
- TD Ameritrade API

---

## 9. Regulatory Differences

### 9.1 A-Share Regulation

- CSRC, Shanghai and Shenzhen exchanges
- Programmatic trading requires registration
- Strict abnormal trading monitoring
- Severe insider trading penalties

### 9.2 US Stock Regulation

- SEC, FINRA
- HFT requires registration
- Reg NMS governs execution
- Pattern Day Trader rules

---

## 10. Practical Recommendations

### 10.1 A-Share Strategy Development

1. **Consider T+1**: Strategy period must be at least overnight
2. **Handle price limits**: Exclude limit up/down days in backtesting
3. **Note turnover**: A-share turnover is high, transaction costs accumulate quickly
4. **Policy risk awareness**: Policy significantly impacts A-shares

### 10.2 US Stock Strategy Development

1. **Note PDT rules**: Small accounts have day trading restrictions
2. **Consider pre/post market**: Major news often released pre/post market
3. **Borrow costs**: Short strategies must consider Hard-to-Borrow costs
4. **Liquidity tiering**: Large and small cap liquidity differs significantly

### 10.3 Cross-Market Strategies

```python
# Market rules configuration
def get_market_rules(market):
    if market == 'CN':
        return {
            't_plus': 1,
            'limit': 0.10,
            'min_lot': 100,
            'short_available': False
        }
    elif market == 'HK':
        return {
            't_plus': 2,
            'limit': 'VCM',
            'min_lot': 'Varies',
            'short_available': True
        }
    elif market == 'US':
        return {
            't_plus': 0,
            'limit': None,
            'min_lot': 1,
            'short_available': True
        }
```

---

> **Core principle**: Don't directly apply US stock strategies to A-shares, and vice versa. Each market has unique rules and participant structures; strategies must adapt to local market characteristics.
