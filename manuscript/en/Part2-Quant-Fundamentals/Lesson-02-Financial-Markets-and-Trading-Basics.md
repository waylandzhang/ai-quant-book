# Lesson 02: Financial Markets and Trading Basics

> **If your Agent doesn't understand order books, slippage, and market impact, it's just a machine that writes essays.**

---

## An Expensive Lesson

A quant newbie developed a strategy that looked very profitable: backtested annual return of 80%, Sharpe ratio of 2.0. He excitedly went live.

In the first week, the strategy signal said "buy 1000 shares of Tesla," and he did. The backtested expected cost was $250,000 ($250 x 1000 shares).

Actual execution price? $253,500.

He lost $3,500 before even starting to make money.

**What happened?**

1. **Slippage**: His market order "ate through" multiple price levels in the order book, with an average price $2.5/share higher than expected
2. **Fees**: Commission + SEC fees = approximately $50 loss per trade
3. **Market Impact**: His large order attracted attention from other traders, pushing up the price

Three months later, the strategy's live performance was 40% lower than the backtest. The strategy wasn't bad - he just miscalculated the costs.

**Goal of this lesson**: Help your Agent understand how real markets work, and avoid the tragedy of "profitable in backtest, losing money live."

---

## 2.1 Market Types

### Characteristics of Different Markets

| Market | Trading Hours | Leverage | Short Selling | Suitable Strategies |
|--------|--------------|----------|---------------|-------------------|
| **Stocks** | Exchange hours (China A-shares: 9:30-15:00) | Limited margin trading | Restricted | Multi-factor, event-driven |
| **Futures** | Nearly 24 hours | High (5-20x) | Natively supported | CTA, arbitrage |
| **Forex** | 24 hours (except weekends) | Very high (50-100x) | Natively supported | Trend following, carry trade |
| **Crypto** | 24/7 | Exchange-defined (1-100x) | Natively supported | High-frequency, cross-exchange arbitrage |

**Implications for Agents**:
- Crypto 24/7 -> Agent must run around the clock, requiring stronger automation
- Futures high leverage -> Risk Agent's stop-loss logic must be stricter
- China A-shares T+1 (regular stocks bought today can only be sold tomorrow, some ETFs excepted) -> Execution Agent needs to consider overnight risk

### Primary Market vs Secondary Market

| | Primary Market | Secondary Market |
|--|----------------|------------------|
| **Definition** | Initial securities issuance (IPO, secondary offerings) | Trading of issued securities between investors |
| **Participants** | Mainly institutions, retail IPO subscriptions | All investors |
| **Quant Opportunities** | IPO strategies, private placement arbitrage | Vast majority of quant strategies |

**This course focuses on the secondary market** - the main battlefield for quantitative trading.

### Exchanges and Matching Mechanisms

Whether it's US stocks, China A-shares, or cryptocurrencies, all trades ultimately flow to the **Matching Engine**.

The exchange isn't your counterparty; it's just a "matchmaker," matching buyers and sellers through rules:

```
Matching Rules (most exchanges):
1. Price Priority: Higher bid orders get filled first
2. Time Priority: At the same price, earlier orders get filled first
```

**Multi-Agent Perspective**:
- **Execution Agent**: Must understand matching logic, deciding between market orders and limit orders
- **Research Agent**: Identifies large fund flows through transaction data

---

## 2.2 Basic Trading Units

### Asset

An asset is what you trade. Different assets have different code conventions:

| Market | Asset Example | Code Format |
|--------|---------------|-------------|
| China A-shares | Kweichow Moutai | 600519.SH |
| US Stocks | Apple | AAPL |
| Cryptocurrency | Bitcoin | BTC/USDT |
| Futures | CSI 300 Index Futures | IF2401 |

### Time Scales: Tick -> Candlestick

Quant systems process data at different granularities:

```
Tick Data (finest)
    | aggregate
Minute Bars (1min, 5min, 15min)
    | aggregate
Daily / Weekly / Monthly (coarsest)
```

**Tick Data**: Every trade, every quote change. Essential for high-frequency strategies, high storage cost.

**OHLCV (Candlesticks)**: Standard format after aggregating ticks:
- **O**pen (opening price)
- **H**igh (highest price)
- **L**ow (lowest price)
- **C**lose (closing price)
- **V**olume (trading volume)

### Order Book

The order book shows market "depth" - how many orders are queued at different price levels:

![Order Book Structure](assets/order-book.svg)

**Spread (Bid-Ask Spread)** = Best Ask - Best Bid = $185.01 - $184.99 = **$0.02**

- Smaller spread -> Better liquidity -> Lower trading costs
- Larger spread -> Worse liquidity -> Higher slippage risk

---

## 2.3 The Real Impact of Trading Costs

> Backtested 50% annual return, losing money live? Trading costs are the culprit.

### Slippage

Backtests assume you can buy at the best price, but in live trading:

```
You want to buy 1000 shares of AAPL, order book:
  $185.01 - 200 shares  <- You eat these 200 first
  $185.05 - 500 shares  <- Then eat these 500
  $185.10 - 300 shares  <- Finally eat these 300

Actual average price = (185.01x200 + 185.05x500 + 185.10x300) / 1000 = $185.057
Expected price = $185.01
Slippage = $0.047/share = Total $47
```

### Fees: Cumulative Effect

Seemingly small fees get amplified in high-frequency trading:

```python
# Assuming 0.1% fee per trade
fee_rate = 0.001
trades_per_day = 50
trading_days = 250

# Annualized fee cost
annual_fee = fee_rate * 2 * trades_per_day * trading_days  # buy and sell each
print(f"Annualized fee cost: {annual_fee:.1%}")  # Output: 25.0%
```

**25% annualized fee cost** - this means your strategy's annual return must exceed 25% just to break even!

### Market Impact

Your large order itself changes the market. **Square Root Law** estimation:

```
Market Impact ~ Y x sigma x sqrt(Q/V)

Y = constant (typically 0.5-1.0)
sigma = daily volatility
Q = your order size
V = average daily volume
```

```python
def estimate_market_impact(order_size, daily_volume, daily_volatility, Y=0.5):
    """Estimate market impact cost"""
    participation = order_size / daily_volume
    impact = Y * daily_volatility * (participation ** 0.5)
    return impact

# Example: Order size is 1% of daily volume
impact = estimate_market_impact(
    order_size=1_000_000,
    daily_volume=100_000_000,
    daily_volatility=0.02
)
print(f"Estimated market impact: {impact:.2%}")  # Output: 0.10%
```

### Cost Summary

| Cost Type | Typical Range | Who's Most Affected |
|-----------|---------------|---------------------|
| Slippage | 0.01% - 0.5% | Large orders, illiquid assets |
| Fees | 0.01% - 0.1% per trade | High-frequency strategies |
| Market Impact | 0.05% - 1%+ | Large capital, small-cap assets |

**Multi-Agent Perspective**: The Execution Agent's core responsibility is minimizing these three costs.

### Paper Exercise: Is Your Strategy Really Profitable?

**Scenario**: You developed a US stock intraday strategy with the following backtest parameters:

| Parameter | Value |
|-----------|-------|
| Backtested Annual Return | 35% |
| Average Daily Trades | 20 (counting buys and sells separately) |
| Average Trade Size | $50,000 |
| Broker Commission | $0 (commission-free broker) |
| SEC Fee | 0.00278% (on sells) |
| Average Slippage | 0.03% |
| Trading Days | 252 days/year |

**Question: What will the live return be?**

---

**Step-by-Step Calculation**:

**Step 1: Calculate per-trade costs**
```
Per-trade slippage cost = $50,000 x 0.03% = $____
Per-trade SEC fee = $50,000 x 0.00278% = $____ (sells only)
```

**Step 2: Calculate daily costs**
```
Daily slippage = $____ x 20 trades = $____
Daily SEC = $____ x 10 trades (sells) = $____
Daily total cost = $____
```

**Step 3: Calculate annualized costs**
```
Annual cost = $____ x 252 days = $____
Total trading volume = $50,000 x 20 x 252 = $252,000,000
Annual cost rate = $____ / $252,000,000 = ____%
```

**Step 4: Calculate live return**
```
Live annual return = 35% - ____% = ____%
```

---

**Answer** (calculate first, then check):

<details>
<summary>Click to reveal answer</summary>

**Key Concept Clarification**:
- **Principal**: Your invested capital, e.g., $100,000
- **Trade Size**: Amount per trade, e.g., $50,000
- **Total Trading Volume**: Trade size x trades x days (includes leverage and turnover effects)
- **Turnover Rate**: Total trading volume / Principal, represents capital rotation

**Calculation Process**:
- Per-trade slippage = $50,000 x 0.03% = **$15**
- Per-trade SEC = $50,000 x 0.00278% = **$1.39**
- Daily slippage = $15 x 20 = **$300**
- Daily SEC = $1.39 x 10 = **$13.9**
- Daily total cost = **$313.9**
- Annual total cost = $313.9 x 252 = **$79,103**

**Cost Rate Calculation (easy to confuse!)**:
| Calculation Method | Formula | Result | Meaning |
|-------------------|---------|--------|---------|
| Relative to trading volume | $79,103 / $252,000,000 | 0.031% | Cost per trade |
| Relative to principal | $79,103 / $100,000 | **79.1%** | How much cost erodes principal |

Where: Annual trading volume = $50,000 x 20 trades x 252 days = $252,000,000 (turnover = 2520x!)

**Final Answer**:
- If strategy principal is $100,000:
- Annual cost **relative to principal** = $79,103 / $100,000 = **79.1%**
- **Live annual return = 35% - 79.1% = -44.1%**

**Conclusion: This strategy will lose big in live trading!** The 0.03% slippage ignored in backtesting (small relative to trade size) accumulates to 79% (relative to principal) - a fatal wound with high turnover.

</details>

**Reflection Questions**:
1. If slippage drops to 0.01%, can the strategy still be profitable?
2. If trading frequency drops to 5 times per day, what happens?
3. What insights does this give you for strategy design?

---

## 2.4 Strategy Lifecycle

A complete trade flows through the multi-agent system from inception to completion:

![Strategy Lifecycle](assets/strategy-lifecycle.svg)

**Detailed Flow**:

1. **Signal Generation (Signal Agent)**
   - "AAPL's MACD shows bullish divergence, recommend going long"

2. **Risk Review (Risk Agent)**
   - "Current total position is 60%, max single position is 10%, this trade can only be 10%"
   - May reject, reduce size, or approve

3. **Order Execution (Execution Agent)**
   - "Order too large, split into 10 smaller orders, send one every 30 seconds using TWAP algorithm"

4. **Execution Monitoring (Monitor Agent)**
   - "5th child order slippage exceeded threshold, pausing subsequent execution"
   - Real-time execution quality feedback

5. **Position Management and Exit (Position Agent)**
   - "Position up 5%, triggering trailing stop"
   - "Position down 2%, triggering stop-loss exit"
   - Loop complete

**Each stage can be an independent specialized Agent** - this is the advantage of multi-agent architecture: specialized division of labor, clear responsibilities, easier debugging.

---

## Lesson Deliverables

After completing this lesson, you will have:

1. **Understanding of Market Structure** - Know the characteristics and constraints of different markets (stocks/futures/crypto)
2. **Trading Cost Awareness** - Can estimate the impact of slippage, fees, and market impact on strategies
3. **Strategy Lifecycle Perspective** - Understand the complete loop from signal to exit

### Verification Checklist

Use these checkpoints to confirm you truly understand this lesson:

| Checkpoint | Verification Standard | Self-Test Method |
|------------|----------------------|------------------|
| **Cost Calculation** | Can complete paper exercise independently, error < 10% | Recalculate without looking at answers |
| **Order Book Understanding** | Can explain why large orders create slippage | Draw order book, simulate 1000-share market order execution |
| **Market Differences** | Can state 3 key differences between China A-shares vs US stocks vs crypto | Explain verbally without notes |
| **Lifecycle** | Can draw strategy flow from signal to exit | Draw on blank paper, label each Agent's role |

**If you can do these**:
- Cost calculation accurate -> You have cost awareness
- Draw order book execution process -> You understand market microstructure
- Draw complete lifecycle -> You have systems thinking

**If you cannot**:
- Re-read relevant sections
- Walk through examples with specific numbers (e.g., AAPL $185)
- Find more detailed explanations in extended reading

---

## Key Takeaways

- [x] Understand characteristics of different markets (stocks/futures/forex/crypto) and their strategy implications
- [x] Master OHLCV and order book fundamentals
- [x] Recognize the three trading cost killers: slippage, fees, market impact
- [x] Understand the complete strategy lifecycle loop: Signal -> Risk -> Execution -> Monitoring -> Exit

---

## Extended Reading

- [Background: Exchanges and Order Book Mechanics](Background/Exchanges-and-Order-Book-Mechanics.md) - Deep dive into L1/L2/L3 data differences
- [Background: HF Market Microstructure](Background/HF-Market-Microstructure.md) - Essential reading if you want to compete with HFT
- [Background: Cryptocurrency Trading Characteristics](Background/Cryptocurrency-Trading-Characteristics.md) - Unique challenges of 24/7 markets

---

## Next Lesson Preview

**Lesson 03: Math and Statistics Fundamentals**

Markets move, but how do we quantify these movements? Why do we use "returns" instead of "prices"? What is a "fat-tailed distribution," and why can the normal distribution assumption blow up your account? Find out next lesson.
