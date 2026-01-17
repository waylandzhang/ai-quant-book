# Lesson 07: Backtest System Pitfalls

> **Backtested 100% annual return, live trading -50%. This isn't an accident - it's inevitable.**

---

## The Fall of a Perfect Strategy

In 2019, a quantitative researcher developed a "perfect" A-share strategy.

Backtest results:
- Annual return: 120%
- Sharpe ratio: 3.5
- Maximum drawdown: 8%
- Win rate: 68%

He confidently invested 5 million RMB.

**Month 1**: +2% return, as expected
**Month 2**: -5% return, getting uneasy
**Month 3**: -8% return, starting to doubt
**Month 6**: -35% cumulative loss

He carefully examined the code and found three fatal errors:

1. **Look-Ahead Bias**: The strategy used "next day's open price" to decide entry, but in reality you don't know the opening price before the market opens

2. **Overfitting**: He tested over 200 parameter combinations and picked the best one. But that was just "happens to perform well on historical data," not truly effective

3. **Ignored trading costs**: Backtest assumed 0.1% slippage, live was 0.5%; backtest didn't account for impact cost, large orders frequently got "eaten"

**Backtesting isn't predicting the future - it's describing the past. If you don't know the pitfalls, the description of the past becomes a misleading guide to the future.**

---

## 7.1 Look-Ahead Bias

### What is Look-Ahead Bias?

**Using future information** in backtests to make decisions.

Most common errors:

| Error | Explanation |
|-------|-------------|
| Using close price to decide entry | Close price is only known after market closes |
| Using same-day data for indicators | Before day ends, MACD and other indicators are uncertain |
| Using future data for labels | "Rose 3 days later" as label, but you don't know this when trading |

### Intuitive Understanding

```
Timeline:
9:30 ─────── 10:00 ─────── 11:00 ─────── Close
  |            |            |            |
  You are here |            |         <- You don't know this info
              |            ^
              |         Real-time available
              ^
           Real-time available
```

**Common backtesting mistake**: Using complete post-close data, pretending to make decisions at any point in time.

### Typical Error Cases

```python
# Wrong code (Look-Ahead Bias)
for i in range(len(data)):
    if data['close'][i] > data['open'][i]:  # Today closed green
        buy_price = data['open'][i]  # <- Wrong! You only know it's green after close
        # At open you don't know if today will close green or red

# Correct code
for i in range(1, len(data)):  # Start from second day
    if data['close'][i-1] > data['open'][i-1]:  # Yesterday closed green
        buy_price = data['open'][i]  # Enter at today's open
```

### How to Avoid?

| Principle | Implementation |
|-----------|----------------|
| **Separate signal and execution** | Signal on day T, execute on day T+1 |
| **Use only past data** | Use `shift(1)` when calculating indicators |
| **Assume worst execution price** | Use high for buys, low for sells |

---

## 7.2 Data Leakage

### What is Data Leakage?

When training models, **test set information "leaks" into the training set**.

### Leakage in Time Series

Financial data has temporal order; you can't randomly split like regular ML:

```
Wrong split:
Training set: [Jan, Mar, May, Jul, Sep, Nov]  <- Random selection
Test set: [Feb, Apr, Jun, Aug, Oct, Dec]  <- Random selection

Problem: Training includes March data, but February is in test
      Using "future" data to train, predicting "past"
```

**Correct split**:

```
Correct split:
Training set: [Jan - Aug]
Validation set: [Sep - Oct]
Test set: [Nov - Dec]

Maintain temporal order!
```

### Leakage in Feature Engineering

| Leakage Type | Example |
|--------------|---------|
| **Future returns as feature** | Using "5-day future return" as input feature |
| **Global normalization** | Normalizing with mean/std of entire dataset |
| **Label in features** | Using return-based indicators to predict returns |

```python
# Wrong: Global normalization (leaks future distribution info)
mean = df['close'].mean()  # Includes future data
std = df['close'].std()    # Includes future data
df['normalized'] = (df['close'] - mean) / std

# Correct: Rolling window normalization
df['normalized'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
```

### Leakage in Feature Importance

If a feature has extremely high importance (e.g., >50%), likely leakage:

```
Signs of leakage:
- One feature importance > 50%
- Model accuracy > 90%
- Training and test performance equally good
```

---

## 7.3 Overfitting

### What is Overfitting?

Model "memorizes" noise in historical data instead of learning real patterns.

### Signs of Overfitting

```
Training set performance: 80% annual, Sharpe 3.0
Test set performance: 5% annual, Sharpe 0.5

Huge gap = Overfitting
```

### Why is Quant Especially Prone to Overfitting?

| Reason | Explanation |
|--------|-------------|
| **Too many parameters** | 100 parameters, only 1000 samples |
| **Multiple testing** | Test 1000 strategies, some will be "effective" |
| **Low signal-to-noise** | Market noise is large, real signals weak |
| **Unstable distribution** | Past patterns may not hold in future |

### Multiple Testing Problem

Suppose you test 100 random strategies (all actually ineffective):

```
Probability of p < 0.05: 5%
Among 100 strategies, expect 5 to be "significant"

Problem: These 5 strategies look effective, but they're just lucky
```

**Bonferroni correction**: If testing n strategies, significance threshold should be 0.05/n

```
Testing 100 strategies -> threshold 0.05/100 = 0.0005
Testing 1000 strategies -> threshold 0.05/1000 = 0.00005
```

### Detecting Overfitting

| Indicator | Overfitting Signal |
|-----------|-------------------|
| Training/test gap | Training >> Test |
| Parameter sensitivity | Small parameter changes, big result changes |
| Cross-period stability | Performance varies greatly across years |
| Strategy complexity | More complex rules = more likely to overfit |

### How to Reduce Overfitting?

| Method | Explanation |
|--------|-------------|
| **Simplify model** | Simpler = harder to overfit |
| **More data** | More samples = harder to memorize noise |
| **Regularization** | L1/L2 penalizes complex models |
| **Early stopping** | Stop when validation performance drops |
| **Ensemble methods** | Multiple models vote, reduce individual bias |

---

## 7.4 Ignoring Trading Costs

### Commonly Ignored Costs

| Cost Type | Typical Value (2024-2025) | Common Backtest Assumption |
|-----------|---------------------------|---------------------------|
| **Commission** | US retail: zero; institutional: ~$0.003/share; China A-shares: ~0.02% | 0% or underestimated |
| **Slippage** | 0.01-0.5% (varies by liquidity) | 0% or fixed value |
| **Market impact** | 0.1-1%+ | Completely ignored |
| **Stock borrowing** | 0.5-50%+ annual (varies by availability) | Ignored |
| **Funding cost** | 4-5% annual (current rate environment) | Ignored |

> **Note on US "Zero Commission"**: While major brokers (Fidelity, Schwab, Robinhood) offer zero commission, there are still SEC fees (the rate changes over time; order-of-magnitude is tens of dollars per $1M sold) and hidden costs from PFOF (Payment for Order Flow).

### Real Impact of Slippage

Assume strategy trades 200 times per year, 0.2% one-way slippage:

```
Annual trading cost = 200 x 0.2% x 2 (buy+sell) = 80%

If your strategy annual return is 50%, after slippage:
Actual return = 50% - 80% = -30%
```

**High-frequency strategies being killed by slippage is the norm**.

### Market Impact Modeling

Large orders "eat through" the order book, moving execution price:

```
Square root rule estimation:

Impact cost ~ sigma x sqrt(Q/V)

sigma = daily volatility (e.g., 2%)
Q = your order size
V = average daily volume

Example:
Order size = 1% of daily volume
Impact cost ~ 2% x sqrt(0.01) = 0.2%
```

### Correct Cost Modeling

```python
def estimate_trading_cost(
    price: float,
    quantity: float,
    daily_volume: float,
    daily_volatility: float,
    commission_rate: float = 0.0003  # 3 bps
) -> dict:
    """Estimate trading costs"""

    # Commission
    commission = price * quantity * commission_rate

    # Slippage (assume market order eats 2-3 levels)
    slippage_rate = 0.001 * (quantity / daily_volume) ** 0.5
    slippage = price * quantity * slippage_rate

    # Market impact
    participation = quantity / daily_volume
    impact_rate = daily_volatility * (participation ** 0.5)
    market_impact = price * quantity * impact_rate

    total = commission + slippage + market_impact

    return {
        'commission': commission,
        'slippage': slippage,
        'market_impact': market_impact,
        'total': total,
        'total_rate': total / (price * quantity)
    }
```

---

## 7.5 Correct Backtesting Methods

### Walk-Forward Validation

**Core idea**: Simulate the real strategy development process - train on past, test on "future," roll forward.

```
Round 1:
  Train: [Jan-Jun]  ->  Test: [Jul]
Round 2:
  Train: [Feb-Jul]  ->  Test: [Aug]
Round 3:
  Train: [Mar-Aug]  ->  Test: [Sep]
...

Each round is an "out-of-sample" test
```

**Advantages**:
- Simulates real conditions
- Multiple OOS tests, more reliable results
- Can detect parameter stability

### Out-of-Sample Testing

**Strictly reserve some data that never participates in development**:

```
Development phase data: [2015-2022]
  |-- Training set: [2015-2019]
  |-- Validation set: [2020-2021]
  --> Parameter tuning, model selection all done here

Final test data: [2022-2023]
  --> Use only once!
     Once seen, it's no longer OOS
```

**Key principle**: OOS data can only be used once. If you adjust strategy based on OOS results, it becomes IS (in-sample).

### Monte Carlo Simulation

**Test strategy robustness through random perturbation**:

```
Original backtest: 30% annual

Monte Carlo simulation (1000 runs):
- Randomly shuffle trade order
- Randomly adjust entry time +/-1 day
- Randomly adjust costs +/-20%

Result distribution:
  5th percentile: 8% annual
  50th percentile: 25% annual
  95th percentile: 45% annual

-> Real performance likely between 8%-45%
```

**If most Monte Carlo outcomes are losses, strategy is unreliable**.

### Backtest Quality Gate

> **This is one of the most important checklists in this book.** Every item must pass before going live. Print it and hang it on your wall.

#### Layer 1: Data Integrity

| # | Check Item | Pass Criteria | Failure Consequence |
|---|------------|---------------|---------------------|
| 1.1 | Data coverage | Train+test >= 5 years, includes at least 1 bull/bear cycle | Untested in extreme markets, stability unknown |
| 1.2 | Adjustment/roll handling | Using adjusted prices, continuous main contract for futures | False signals, inflated returns |
| 1.3 | Survivorship bias | Includes delisted stock data | Historical returns inflated 50%+ |
| 1.4 | Timezone alignment | All sources unified (UTC or local) | Cross-market signals confused |

#### Layer 2: Temporal Integrity

| # | Check Item | Pass Criteria | Failure Consequence |
|---|------------|---------------|---------------------|
| 2.1 | Look-Ahead Bias | Signal on day T, execute on T+1 | Backtest returns inflated 2-10x |
| 2.2 | Data leakage | Strict temporal train/test separation | Overfitting undetectable |
| 2.3 | Feature calculation | All features use shift(1) or earlier | Used future information |
| 2.4 | Label definition | Labels use only data before current time | Label leakage |

#### Layer 3: Overfitting Detection

| # | Check Item | Pass Criteria | Failure Consequence |
|---|------------|---------------|---------------------|
| 3.1 | OOS performance | OOS return > 50% of training return | Severe overfitting |
| 3.2 | Parameter stability | +/-20% parameter change, <30% return change | Parameter sensitive, not reproducible |
| 3.3 | Multiple testing | Testing n strategies, p-value threshold = 0.05/n | False positive strategies |
| 3.4 | Cross-period stability | Each year Sharpe > 0.5, no large swings | Only works in specific periods |

#### Layer 4: Cost Modeling

| # | Check Item | Pass Criteria | Failure Consequence |
|---|------------|---------------|---------------------|
| 4.1 | Commission | Includes real rates (US retail: zero, institutional: ~$0.003/share; China: ~0.02%) | Underestimated costs |
| 4.2 | Slippage | Conservative assumption (recommend 0.1-0.3%) | HFT strategies lose money |
| 4.3 | Market impact | Large orders consider square root model | Capacity inflated |
| 4.4 | Funding costs | Margin shorts consider borrowing fees | Short strategy returns inflated |

#### Layer 5: Validation Methods

| # | Check Item | Pass Criteria | Failure Consequence |
|---|------------|---------------|---------------------|
| 5.1 | Walk-Forward | At least 10 rolling validations | Single validation unreliable |
| 5.2 | Monte Carlo | 90% of simulations > 0 | Too much luck component |
| 5.3 | Stress test | Tested on 2008, 2020, 2022 crises | Blows up during crisis |
| 5.4 | Return decay | Backtest return x 0.5 still acceptable | Live expectations too high |

**How to use**:
1. Every item must pass; any failure means no live trading
2. Check passed items; note specific issues for failures
3. After fixes, rerun entire check process

**Quick self-check** (ask yourself after each backtest):
- "Did I use future information?" -> Check Layer 2
- "Does it work in different periods?" -> Check Layer 3
- "Still profitable after costs?" -> Check Layer 4
- "Is there a luck component?" -> Check Layer 5

---

## 7.6 Common Misconceptions

**Misconception 1: High backtest returns mean good strategy**

Most dangerous assumption. High backtest returns could be from overfitting, Look-Ahead Bias, underestimated costs. What really matters is OOS performance and parameter stability.

**Misconception 2: Tested 100 strategies, pick the best one**

Classic multiple testing problem. Testing 100 random strategies, expect 5 to be "significant" (p<0.05), but they're just lucky. Correct approach: p-value threshold = 0.05/100 = 0.0005.

**Misconception 3: Backtesting with close price execution is reasonable**

Not reasonable. Close price is only known after close. In actual trading, you place orders before close, execution could be close price +/- slippage. Correct: Signal on day T, execute on T+1.

**Misconception 4: Good paper trading means ready for live**

Not enough. Paper trading typically has ideal slippage, no market impact, 100% fills. Live needs gradual deployment: Paper trading -> 1-5% capital live -> gradually increase.

---

## 7.7 From Backtest to Live Trading

> **Industry consensus**: Average performance degradation from backtest to live trading is **30-50%**. This isn't pessimism - it's hard-earned industry experience validated by countless live deployments.

### Why is Live Always Worse Than Backtest?

| Factor | Backtest | Live | Return Impact |
|--------|----------|------|---------------|
| Execution price | Close or assumed | Actual (usually worse) | -5~20% |
| Slippage | Fixed assumption (0.1%) | Varies with market (0.2-0.5%) | -10~30% |
| Market impact | Completely ignored | Large orders significantly increase costs | -5~50% |
| Fill rate | Assume 100% | Partial fills possible | -5~15% |
| Latency | Ignored | 50-500ms | -2~10% |
| Failures | Don't exist | Network down, API errors | Unpredictable |
| Psychology | Doesn't exist | Fear and greed | Unpredictable |
| Model overfitting | Invisible | Exposed | -20~80% |

### Industry Data on Performance Degradation

Based on live tracking statistics from multiple quantitative institutions:

```
+-----------------------------------------------------------+
|         Backtest to Live Performance Degradation          |
+-----------------------------------------------------------+
|  Degradation Range  |  Proportion  |  Main Causes         |
+-----------------------------------------------------------+
|  < 20%              |  10%         |  Low-freq strategies, excellent cost modeling  |
|  20-30%             |  25%         |  Normal degradation range  |
|  30-50%             |  40%         |  Typical case (industry average)  |
|  50-70%             |  15%         |  Underestimated costs, mild overfitting  |
|  > 70%              |  10%         |  Severe overfitting, severely underestimated costs  |
+-----------------------------------------------------------+
```

**Key insight**: If your strategy shows amazing backtest returns (annual >100%), it will almost certainly be heavily discounted in live trading.

### Expected Return Decay Formula

**Conservative estimation method** (recommended):

```
Expected live return = Backtest return x 0.5 - Hidden costs

Hidden costs include:
- Data latency: -2~5%
- Execution difference: -3~10%
- Model decay: -5~15%
```

**Scenario-based estimation**:

| Strategy Type | Backtest Annual | Optimistic Expectation | Conservative Expectation | Decay Factor |
|---------------|-----------------|------------------------|--------------------------|--------------|
| Low-freq value | 30% | 20% | 12% | 0.4-0.65 |
| Mid-freq momentum | 50% | 30% | 18% | 0.35-0.6 |
| HFT market-making | 100% | 40% | 20% | 0.2-0.4 |
| ML factor | 80% | 35% | 15% | 0.2-0.45 |

**Core principle**: **If backtest returns halved are still acceptable, then consider live trading**.

### Breakdown of Degradation Causes

```
Backtest Return 100%
    |
    +-- Underestimated Trading Costs ------------------- -20%
    |     - Slippage assumed 0.1%, actual 0.3%
    |     - Market impact ignored
    |
    +-- Overfitting ------------------------------------ -25%
    |     - Parameters over-optimized on historical data
    |     - Multiple testing not corrected
    |
    +-- Execution Difference --------------------------- -10%
    |     - Latency causes price drift
    |     - Partial fills affect strategy logic
    |
    +-- Market Regime Change --------------------------- -10%
    |     - Historical patterns no longer valid
    |     - New market participant structure
    |
    +-- Expected Live Return --------------------------- 35%
```

### Gradual Deployment

| Stage | Capital Scale | Goal |
|-------|---------------|------|
| **Paper Trading** | $0 | Verify system stability |
| **Micro live** | 1-5% of total | Verify execution quality |
| **Small live** | 10-20% | Accumulate live data |
| **Normal operation** | Planned scale | Continuous monitoring |

---

## 7.8 Multi-Agent Perspective

### Agent Division in Backtest Systems

![Backtest Agent Flow](assets/backtest-agent-flow.svg)

### Why Need Independent Validation Agent?

- **Strategy developers are biased**: Always want to prove their strategy works
- **Automated detection more reliable**: Won't miss check items
- **Standardized process**: Ensures every strategy goes through same review

---

## Code Implementation (Optional)

### Walk-Forward Validation Framework

```python
import pandas as pd
import numpy as np
from typing import Callable, List, Dict

def walk_forward_validation(
    data: pd.DataFrame,
    strategy_fn: Callable,
    train_window: int = 252,  # ~1 year
    test_window: int = 63,    # ~3 months
    step: int = 21            # Roll monthly
) -> List[Dict]:
    """
    Walk-Forward Validation

    Parameters:
    - data: DataFrame with price data
    - strategy_fn: Strategy function, receives training data, returns model/parameters
    - train_window: Training window size (days)
    - test_window: Test window size (days)
    - step: Roll step size (days)

    Returns:
    - List of results from each round
    """
    results = []
    total_len = len(data)

    for start in range(0, total_len - train_window - test_window, step):
        train_end = start + train_window
        test_end = train_end + test_window

        train_data = data.iloc[start:train_end]
        test_data = data.iloc[train_end:test_end]

        # Train/optimize on training set
        model = strategy_fn(train_data)

        # Evaluate on test set
        test_returns = evaluate_strategy(model, test_data)

        results.append({
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'test_return': test_returns.sum(),
            'test_sharpe': calculate_sharpe(test_returns)
        })

    return results


def detect_look_ahead_bias(backtest_fn: Callable, data: pd.DataFrame) -> bool:
    """
    Detect Look-Ahead Bias

    Principle: If using future data, trade time won't match signal time
    """
    trades = backtest_fn(data)

    for trade in trades:
        signal_time = trade['signal_time']
        execution_time = trade['execution_time']

        # Signal time should be < execution time
        if signal_time >= execution_time:
            print(f"Possible Look-Ahead Bias: signal {signal_time} >= execution {execution_time}")
            return True

    return False


def monte_carlo_backtest(
    base_results: pd.Series,
    n_simulations: int = 1000,
    return_perturbation: float = 0.1
) -> Dict:
    """
    Monte Carlo Simulation

    Parameters:
    - base_results: Original backtest daily returns
    - n_simulations: Number of simulations
    - return_perturbation: Return perturbation magnitude

    Returns:
    - Simulation result statistics
    """
    simulated_returns = []

    for _ in range(n_simulations):
        # Shuffle order + add noise
        shuffled = base_results.sample(frac=1, replace=False)
        noisy = shuffled * (1 + np.random.uniform(-return_perturbation, return_perturbation, len(shuffled)))
        total_return = (1 + noisy).prod() - 1
        simulated_returns.append(total_return)

    simulated_returns = np.array(simulated_returns)

    return {
        'mean': simulated_returns.mean(),
        'std': simulated_returns.std(),
        'percentile_5': np.percentile(simulated_returns, 5),
        'percentile_50': np.percentile(simulated_returns, 50),
        'percentile_95': np.percentile(simulated_returns, 95),
        'prob_positive': (simulated_returns > 0).mean()
    }
```

---

## Lesson Deliverables

After completing this lesson, you will have:

1. **Deep understanding of backtest pitfalls** - Know why backtests profit but live trading loses
2. **Bias detection ability** - Can identify Look-Ahead Bias and data leakage
3. **Correct backtesting methodology** - Walk-Forward, OOS, Monte Carlo
4. **Live expectation management** - Understand return decay from backtest to live
5. **Backtest quality gate checklist** - Reusable 20-item standard

### Verification Checklist

| Checkpoint | Standard | Self-Test Method |
|------------|----------|------------------|
| **Look-Ahead detection** | Can identify look-ahead bias in code | Given code snippet, point out where future data is used |
| **Data splitting** | Can correctly split train/validation/test | Given 5 years data, draw temporal split diagram |
| **Overfitting identification** | Can state 3 overfitting signals | List detection metrics without notes |
| **Quality Gate** | Can recite core standards of 20 checks | Fill in checklist from scratch |

**Diagnostic Exercise**:

A strategy has these backtest results - diagnose possible issues:
- Training annual return: 85%
- Test annual return: 12%
- +/-10% parameter change causes 50-200% return change
- Tested 150 strategy variants
- Only has 2018-2022 data

<details>
<summary>Click to reveal diagnosis</summary>

**Problem Diagnosis**:

1. **Severe overfitting** - Training 85% vs Test 12%, 7x gap, far exceeds 50% threshold
2. **Parameter sensitive** - +/-10% change causes 50-200% return change, fails stability requirement
3. **Multiple testing problem** - 150 variants, p-value threshold should be 0.05/150 = 0.00033
4. **Insufficient data** - 4 years doesn't cover 2008, 2020 crises, stability unknown

**Conclusion**: This strategy cannot go live. Needs:
- Simplify model to reduce parameters
- Get longer data (at least 10 years)
- Apply Bonferroni correction to filter strategies
- Conduct Walk-Forward and Monte Carlo validation

</details>

---

## Key Takeaways

- [x] Understand the essence of Look-Ahead Bias and detection methods
- [x] Master correct time series data splitting to avoid data leakage
- [x] Recognize dangers of overfitting and multiple testing problems
- [x] Learn to correctly model trading costs
- [x] Master Walk-Forward, OOS, Monte Carlo validation methods

---

## Extended Reading

- [Background: Famous Quant Disasters](../Part1-Quick-Start/Background/Famous-Quant-Disasters.md) - Real cases of backtest failures
- [Background: Statistical Traps of Sharpe Ratio](Background/Statistical-Traps-of-Sharpe-Ratio.md) - Multiple testing and Deflated Sharpe Ratio
- [Background: Tick-Level Backtest Framework](Background/Tick-Level-Backtest-Framework.md) - High-precision backtesting and queue position simulation
- *Advances in Financial Machine Learning* - Chapter 12: Backtesting

---

## Part 2 Summary (Up to Lesson 07)

At this point, you’ve dissected the two biggest “silent killers” in quant: data and backtesting.

Up to Lesson 07, you learned:

| Lesson | Core Takeaways |
|--------|----------------|
| Lesson 02 | Market structure, trading costs, strategy lifecycle |
| Lesson 03 | Time series, returns, risk measurement, fat tails |
| Lesson 04 | Technical indicators are feature engineering, not buy/sell signals |
| Lesson 05 | Trend following vs mean reversion, strategy choice depends on market state |
| Lesson 06 | Data engineering challenges: API, timezone, quality, bias |
| Lesson 07 | Backtest pitfalls and correct validation methods |

## Next Lesson Preview

**Lesson 08: Beta, Hedging, and Market Neutrality**

Your strategy made money, but where did it come from? Is it your Alpha (true excess return), or just riding the market up (Beta)? Next lesson we dive deep into the source of risk, learn how to decompose returns, how to hedge, and why retail investors can't do true market neutral.
