# Lesson 08: Beta, Hedging, and Market Neutrality

> **You think you're generating Alpha, but you might just be betting on direction.**

---

## A Typical Scenario (Illustrative)

> Note: The following is a synthetic example to illustrate common phenomena; numbers are illustrative and don't correspond to any specific institution/product.

In 2021, a quant team presented their performance to investors: 32% annual return, 1.8 Sharpe ratio, only 12% maximum drawdown. Investors happily invested.

A year later, investors discovered the truth:

That year, the Nasdaq 100 index rose 27%.

When they did a simple regression analysis, the results were embarrassing:
- **Beta = 1.15** (strategy highly correlated with market)
- **True Alpha = 32% - 1.15 x 27% = 1%**

In other words, 80% of this "high-return strategy's" gains came from **the market's own rise**, not any unique trading skill.

Even worse, when the market fell 33% in 2022, this strategy lost 38%. Investors finally understood: they didn't buy an "Alpha strategy" - they bought a **leveraged market bet**.

**The lesson from this story**:

> Without understanding Beta, you don't know where your returns come from; without understanding hedging, you don't know where your risks are.

---

## 8.1 Understanding Beta Again

### 8.1.1 The Essence of Beta

In the first lesson's background knowledge, we briefly introduced Alpha and Beta. Now let's understand them more deeply.

**Beta** measures the **sensitivity** of your portfolio relative to the market benchmark.

```
Beta = 1.0  -> Market up 10%, you up 10%
Beta = 1.5  -> Market up 10%, you up 15% (but you'll also lose 50% more when it falls)
Beta = 0.5  -> Market up 10%, you up 5% (half the volatility)
Beta = 0    -> Your returns are unrelated to market movement (this is "market neutral")
Beta < 0    -> Market up, you down; market down, you up (inverse)
```

### 8.1.2 Why is Beta More Important Than Alpha?

Many obsess over finding Alpha while ignoring a harsh reality:

| Dimension | Beta Returns | Alpha Returns |
|-----------|--------------|---------------|
| **Source** | Reward for bearing market risk | Reward for unique skill/information |
| **Accessibility** | Anyone can get it (buy index) | Very few can consistently get it |
| **Cost** | Extremely low (index fund fee 0.03%) | Extremely high (hedge fund 2%+20%) |
| **Capacity** | Nearly unlimited | Limited (Alpha decays) |
| **Sustainability** | Long-term stable (market risk premium) | Uncertain (strategy may fail) |

**Key insight**:

> If your strategy has Beta = 1, how much of your "strategy return" is Alpha vs Beta?
>
> For most retail investors and many "quant funds," **over 80% of returns come from Beta**.

### 8.1.3 Paper Exercise: Decompose Your Returns

**Scenario**: Your strategy performed as follows over the past year:
- Strategy return: +25%
- S&P 500 (benchmark) over same period: +18%
- Your strategy Beta (calculated via regression): 1.2

**Question**: What is your true Alpha?

```
Alpha = Strategy return - Beta x Benchmark return
Alpha = 25% - 1.2 x 18%
Alpha = 25% - 21.6%
Alpha = 3.4%
```

> Note: Strictly speaking, Alpha/Beta are typically estimated via regression on **excess returns** (subtracting the risk-free rate), where Alpha is the intercept. This simplified formula is for intuition.

**Interpretation**:
- You thought you made 25%
- Actually, 21.6% was because you took on more risk than the market (Beta = 1.2)
- Only 3.4% is your "true skill"
- If the market drops 20% next year, you might lose 24% (1.2 x 20%)

<details>
<summary>Code Implementation (for engineers)</summary>

```python
import numpy as np
import pandas as pd
from scipy import stats

def decompose_returns(strategy_returns: pd.Series,
                      benchmark_returns: pd.Series,
                      rf_rate: float = 0.02) -> dict:
    """
    Decompose strategy returns into Alpha and Beta components

    Parameters:
        strategy_returns: Daily strategy return series
        benchmark_returns: Daily benchmark return series
        rf_rate: Annualized risk-free rate

    Returns:
        Dictionary with alpha, beta, r_squared
    """
    # Convert to excess returns
    rf_daily = rf_rate / 252
    excess_strategy = strategy_returns - rf_daily
    excess_benchmark = benchmark_returns - rf_daily

    # Linear regression: R_strategy = alpha + beta * R_benchmark
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        excess_benchmark, excess_strategy
    )

    beta = slope
    alpha_daily = intercept
    alpha_annual = alpha_daily * 252  # Annualize

    # Return decomposition
    # Use compounded returns instead of simple sums (closer to typical backtest conventions)
    total_return = (1 + strategy_returns).prod() - 1
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    beta_contribution = beta * benchmark_total_return
    alpha_contribution = total_return - beta_contribution

    return {
        'beta': beta,
        'alpha_annual': alpha_annual,
        'r_squared': r_value ** 2,
        'total_return': total_return,
        'benchmark_total_return': benchmark_total_return,
        'beta_contribution': beta_contribution,
        'alpha_contribution': alpha_contribution,
        'beta_pct': beta_contribution / total_return * 100 if total_return != 0 else 0
    }

# Example usage
# result = decompose_returns(strategy_rets, spy_rets)
# print(f"Beta: {result['beta']:.2f}")
# print(f"Alpha (annualized): {result['alpha_annual']:.2%}")
# print(f"Beta contribution to returns: {result['beta_pct']:.1f}%")
```

</details>

---

## 8.2 The Essence of Hedging

### 8.2.1 What is Hedging?

The core idea of **hedging** is very simple:

> Hold positions in the opposite direction to offset risks you don't want to bear.

**Analogy**: You bought a Beijing-to-Shanghai flight ticket but worry the flight might be cancelled. You can also buy a same-time high-speed rail ticket as a "hedge" - if the flight is normal, the train ticket is wasted (hedge cost); if the flight is cancelled, the train ticket saves you.

### 8.2.2 Notional Hedging vs Beta Hedging

This is the first mistake many make: **thinking equal dollar amounts means equal risk**.

**Case**:

You hold $1M of tech stocks (Beta = 1.5). To hedge, you short $1M of S&P 500 ETF (Beta = 1.0).

**Question**: Is this a perfect hedge?

```
Your long Beta exposure: $1M x 1.5 = $1.5M
Your short Beta exposure: $1M x 1.0 = $1M
Net Beta exposure: $1.5M - $1M = $0.5M (long)

You still have $0.5M of Beta exposure unhedged!
```

**Correct Beta hedging**:

```
Short amount needed = Long amount x (Long Beta / Short instrument Beta)
                    = $1M x (1.5 / 1.0)
                    = $1.5M

Verification:
Long Beta exposure: $1M x 1.5 = $1.5M
Short Beta exposure: $1.5M x 1.0 = $1.5M
Net Beta exposure: 0
```

### 8.2.3 Paper Exercise: Calculate Hedge Ratios

| Scenario | Long Position | Long Beta | Short Instrument Beta | Short Amount Needed | Verification |
|----------|---------------|-----------|----------------------|---------------------|--------------|
| A | $500K growth stocks | 1.3 | 1.0 (SPY) | ? | ? |
| B | $1M utility stocks | 0.6 | 1.0 (SPY) | ? | ? |
| C | $800K tech stocks | 1.8 | 1.2 (QQQ) | ? | ? |

<details>
<summary>Click to reveal answers</summary>

| Scenario | Calculation | Short Amount | Net Beta |
|----------|-------------|--------------|----------|
| A | $500K x 1.3 / 1.0 | **$650K** | 500Kx1.3 - 650Kx1.0 = 0 |
| B | $1M x 0.6 / 1.0 | **$600K** | 1Mx0.6 - 600Kx1.0 = 0 |
| C | $800K x 1.8 / 1.2 | **$1.2M** | 800Kx1.8 - 1.2Mx1.2 = 0 |

**Key findings**:
- Scenario A: Need to short more than long amount because long Beta > 1
- Scenario B: Need to short less than long amount because long Beta < 1
- Scenario C: Using QQQ to hedge requires considering QQQ's own Beta

</details>

---

## 8.3 Hedging Instrument Comparison

### 8.3.1 ETF Shorting vs Index Futures Hedging

| Dimension | ETF Shorting | Index Futures |
|-----------|--------------|---------------|
| **Capital efficiency** | Low (≥150% margin, Reg T requirement) | High (only 5-15% margin) |
| **Cost** | Stock borrowing interest (1-10%/year) | Basis cost (<1% in low-rate environment, 2-4% when rates are high) |
| **Rolling** | None | Need monthly/quarterly rolls |
| **Precision** | Can match exact amounts | Fixed contract size |
| **Availability** | Depends on broker stock loan inventory | Standardized contracts, good liquidity |
| **Retail access** | Partially available | Usually requires professional account |

### 8.3.2 Basis Risk

**Basis** = Futures price - Spot price

This is the biggest hidden risk when hedging with futures:

```
Normal situation:
  Futures premium = Spot price + Carry cost (interest - dividends)
  Basis usually positive, converges to zero at expiry

Abnormal situation (during crisis):
  Massive capital rushes into futures to short
  Futures trade at large discount (futures < spot)
  Your hedge position actually loses money
```

**Example (approximate): March 2020**

> Note: The table below shows approximate values to illustrate "basis risk" mechanics, not exact historical data.

| Date | S&P 500 Spot | S&P 500 Futures | Basis | Impact |
|------|--------------|-----------------|-------|--------|
| 3/9 | 2746 | 2730 | -16 | Small discount |
| 3/12 | 2480 | 2400 | **-80** | Severe discount |
| 3/16 | 2386 | 2280 | **-106** | Extreme discount |

**Impact**: If you shorted futures to hedge, you not only suffered from spot decline, but also lost extra money as futures discount widened.

### 8.3.3 Real-World Hedging Cost Considerations

| Cost Type | Source | Annual Estimate | Notes |
|-----------|--------|-----------------|-------|
| **Stock borrow interest** | Borrowing stock/ETF to short | 1-10% | Popular stocks can be > 30% |
| **Futures basis** | Futures premium cost | 0.5-2% | Normal markets |
| **Transaction cost** | Bid-ask spread + commission | 0.1-0.3%/trade | Futures lower |
| **Roll cost** | Futures contract roll | 0.1-0.5%/roll | Monthly/quarterly |
| **Opportunity cost** | Margin/capital tied up | 2-5% | Risk-free rate |

**Key formula**:

> **Net hedged return = Alpha - Hedging cost**
>
> If Alpha < Hedging cost, the hedged strategy loses money.

---

## 8.4 Market Neutral Strategies

### 8.4.1 What is "True" Market Neutral?

**Market Neutral** means:

> Regardless of market direction, strategy returns are unaffected (Beta ~ 0)

![Market Neutral Strategy Structure](assets/market-neutral-structure.svg)

### 8.4.2 Three Levels of Market Neutrality

| Level | Definition | Difficulty | Effectiveness |
|-------|------------|------------|---------------|
| **Dollar Neutral** | Equal long and short dollar amounts | Simple | Can't truly eliminate Beta |
| **Beta Neutral** | Equal long and short Beta exposure | Medium | Eliminates market risk |
| **Factor Neutral** | Equal long and short factor exposures | Hard | Eliminates multiple systematic risks |

**The problem with Dollar Neutral**:

```
Assume:
  Long: $1M tech stocks (Beta = 1.5)
  Short: $1M utility stocks (Beta = 0.6)

Looks "market neutral" (equal dollar amounts)

Actual Beta exposure:
  Net Beta = $1M x 1.5 - $1M x 0.6 = $0.9M

You're actually long $0.9M of market exposure!
```

### 8.4.3 Why Can't Retail Investors Do Market Neutral?

| Barrier | Institution | Retail |
|---------|-------------|--------|
| **Borrow cost** | 0.5-2%/year (prime client rate) | 3-10%/year (retail rate) |
| **Borrow availability** | Prime broker relationships | Often can't borrow desired stocks |
| **Capital efficiency** | 2-4x leverage | Usually no leverage |
| **Transaction cost** | 0.01-0.05%/trade | 0.1-0.5%/trade |
| **Portfolio size** | 100+ stocks | Usually 10-20 stocks |
| **Risk infrastructure** | Real-time factor exposure monitoring | Manual tracking |

**Let's do the math**:

Assume you have a "truly effective" neutral strategy:
- Gross Alpha: 8%/year (already quite good)
- Stock borrow cost: 5%/year (retail rate)
- Transaction cost: 2%/year (500% turnover, 0.2% each)
- **Net return: 8% - 5% - 2% = 1%**

Might as well buy Treasury bonds.

### 8.4.4 How Do Institutions Make It Work?

| Advantage | Specifics |
|-----------|-----------|
| **Economies of scale** | $1B scale, fixed costs become negligible |
| **Prime broker relationships** | Borrow rates < 1%, rich stock pool |
| **Leverage** | 2-4x leverage amplifies Alpha |
| **Technology infrastructure** | Millisecond execution, real-time risk control |
| **Talent** | 10+ person team dedicated to research |

**Renaissance's Medallion Fund**:

```
Estimated operating parameters:
- Gross returns: 60-80%/year
- Fees: 5% management + 44% performance
- Net returns: ~35-40%/year
- Beta: Near 0
- Capacity: Internal money only, ~$12B (2024 estimate)
```

---

## 8.5 Common Misconceptions

### Misconception 1: "Long tech, short financials = market neutral"

**Problem**: Sector hedging != market hedging.

```
Tech stock Beta ~ 1.3
Financials Beta ~ 1.1

Equal allocation:
Net Beta = 0.5 x 1.3 - 0.5 x 1.1 = 0.1 (still long market)

Bigger problem:
You're simultaneously long "growth factor," short "value factor"
This isn't market neutral - it's a factor bet
```

### Misconception 2: "Low volatility after hedging = safe"

**Problem**: Low volatility != low risk.

```
Case: LTCM
  - Strategy volatility was low (10% annual)
  - But 25x leverage
  - Actual risk exposure = 10% x 25 = 250%
  - One "impossible event" caused bankruptcy
```

### Misconception 3: "Neutral strategy profitable in backtest = profitable live"

**Problem**: Backtests ignore many hidden costs.

```
Backtest assumes:
  x Stock borrow always available
  x Borrow cost fixed
  x No slippage
  x No market impact

Live reality:
  - Can't borrow stocks you want to short
  - Borrowed stocks get recalled
  - Insufficient liquidity causes slippage
  - Your trades get front-run
```

### Misconception 4: "Shorting is as easy as going long"

**Problem**: Shorting has natural asymmetry.

| Dimension | Long | Short |
|-----------|------|-------|
| Max loss | 100% (stock goes to zero) | **Unlimited** (stock can rise infinitely) |
| Cost | None (buy and hold) | Yes (borrow interest accrues) |
| Time | Can hold indefinitely | May be forced to return shares |
| Psychology | Can wait for recovery when losing | Forced to cover when losing |

---

## 8.6 Multi-Agent Perspective

In multi-agent quant systems, Beta management and hedging need dedicated Agents.

### 8.6.1 Hedging Agent Responsibilities

![Hedging Agent](assets/hedging-agent.svg)

### 8.6.2 Collaboration with Other Agents

| Collaborator | Collaboration Method |
|--------------|---------------------|
| **Signal Agent** | Receive position change signals, calculate new hedge requirements |
| **Risk Agent** | Report Beta exposure, receive risk budget constraints |
| **Execution Agent** | Send hedge orders, receive execution feedback |
| **Cost Agent** | Query borrow costs, get futures basis data |
| **Regime Agent** | Receive signals during crisis, increase hedge intensity |

### 8.6.3 Agent Architecture in Neutral Strategies

![Neutral Strategy Agent Architecture](assets/neutral-strategy-architecture.svg)

---

## Verification Checklist

After completing this lesson, verify your learning with these standards:

| Check Item | Pass Standard | Self-Test Method |
|------------|---------------|------------------|
| Understand Beta | Can explain what Beta = 1.2 means | Explain in your own words |
| Decompose returns | Can calculate Alpha and Beta contributions | Complete paper exercise |
| Calculate hedge ratios | Can correctly calculate short amount for Beta neutrality | Complete hedge exercise |
| Understand hedge costs | Can list at least 3 hedging costs | Explain why retail can't do neutral |
| Identify misconceptions | Can point out problems with "equal dollar hedge" | Explain Dollar Neutral's flaw |

### Comprehensive Exercise

**Design a simplified neutral strategy framework**:

1. Assume you have $1M capital, want to build a Beta neutral strategy
2. Long: Hold 5 growth stocks, average Beta = 1.4
3. Short: Use SPY to hedge
4. Questions:
   - How much SPY to short?
   - If borrow rate is 5%/year, what's the hedge cost?
   - What's the minimum gross Alpha needed to cover costs?

<details>
<summary>Click to reveal answers</summary>

1. **Short amount**:
   - Long Beta exposure = $1M x 1.4 = $1.4M
   - Need to short SPY (Beta = 1.0): $1.4M

2. **Hedge cost**:
   - Borrow interest = $1.4M x 5% = $70,000/year
   - As % of capital = $70K / $1M = 7%

3. **Breakeven Alpha**:
   - Gross Alpha > 7% just to cover borrow cost
   - Add transaction costs (assume 1%), need > 8%
   - This means your stock-picking must be very strong

**Conclusion**: For retail investors, this strategy is likely not realistic.

</details>

---

## Lesson Deliverables

After completing this lesson, you will have:

1. **Beta decomposition framework** - Understand where your returns actually come from
2. **Hedge ratio calculation methods** - Know how to correctly calculate hedge amounts
3. **Hedging cost checklist** - Understand hidden costs' impact on strategy
4. **Market neutral feasibility assessment** - Judge if neutral strategy suits you

---

## Key Takeaways

- [x] Beta measures strategy sensitivity to market, is a major source of returns
- [x] Notional hedging (equal dollars) != Beta hedging (equal Beta exposure)
- [x] Hedging costs (borrow, basis, transaction) can consume Alpha
- [x] Market neutral strategies are nearly infeasible for retail (costs, tools, scale)
- [x] "Equal long-short dollars" doesn't equal "market neutral"

---

## Extended Reading

- [Background: Alpha and Beta](../Part1-Quick-Start/Background/Alpha-and-Beta.md) - Basic definitions of Alpha and Beta
- [Lesson 15: Risk Control and Money Management](../Part4-Multi-Agent/Lesson-15-Risk-Control-and-Money-Management.md) - More on risk management
- [Background: Famous Quant Disasters](../Part1-Quick-Start/Background/Famous-Quant-Disasters.md) - LTCM and other hedging failure cases

---

## Next Lesson Preview

**Lesson 09: Supervised Learning in Quantitative Trading**

After understanding the essence of Beta and hedging, we start exploring how to use machine learning to predict markets. But remember: **Prediction is just the first step; converting prediction into tradeable Alpha is key** - and that requires deducting all costs, including the hedging costs we discussed today.
