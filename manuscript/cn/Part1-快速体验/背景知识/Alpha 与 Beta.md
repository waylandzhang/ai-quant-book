# 背景知识：Alpha 与 Beta

> 在量化投资中，Alpha (α) 代表“超额收益”，Beta (β) 代表“市场基准收益”。理解两者的关系是构建对冲策略和风险管理的核心。

---

## 一、 核心概念定义

### 1.1 Beta (β): 系统性风险与收益
**Beta** 衡量投资组合相对于大盘（基准）的波动敏感度。它反映了你从“市场上涨”中分到的那部分利润，也反映了你承担的结构性风险。

- **计算公式**: $\beta_i = \frac{Cov(r_i, r_m)}{Var(r_m)}$
  - $r_i$: 资产收益率
  - $r_m$: 市场基准收益率
- **基准值**:
  - **β = 1**: 与市场同步（如沪深300指数基金）
  - **β > 1**: 激进型（多为成长股、科技股）
  - **β < 1**: 防御型（多为公用事业、大消费）
  - **β < 0**: 负相关（极少数，如反向杠杆ETF）
  - **β ≈ 0**: **市场中性 (Market Neutral)**，收益不随大盘波动。

### 1.2 Alpha (α): 超额收益
**Alpha** 是经过风险调整（排除 Beta 影响）后的纯粹盈利。它被视为投资经理的“技能”或模型的“独门秘籍”。

- **现代定义**: 无法被已知因子（如市值、价值、动量）解释的收益。
- **现状**: 随着市场效率提高，传统的 Alpha 正在逐渐衰减（Alpha Decay）并演变为基准收益。

---

## 二、 CAPM 模型与公式

经典资本资产定价模型 (CAPM) 揭示了回报的构成：

$$E(R_i) = R_f + \beta_i (E(R_m) - R_f) + \alpha_i$$

- $R_f$: 无风险利率 (Risk-free rate)
- $E(R_m) - R_f$: 市场风险溢价
- **Alpha 的本质**: 实际回报率与 CAPM 预测回报率之间的差额。

---

## 三、 演进：从 Alpha 到 Smart Beta

现代量化将收益进一步拆解：

1.  **Traditional Beta**: 纯大盘波动。
2.  **Smart Beta (因子收益)**: 介于 α 与 β 之间。通过系统性暴露于某些因子（如：市值 Size, 价值 Value, 动量 Momentum）获得的收益。
3.  **Pure Alpha**: 真正的超额技能（如：高频套利、另类数据挖掘、AI 深度模式识别）。

| 维度 | Beta (β) | Smart Beta | Alpha (α) |
| :--- | :--- | :--- | :--- |
| **收益来源** | 市场整体表现 | 因子暴露 | 独特信息/算法 |
| **成本** | 极低 (指数基金) | 较低 | 高 (对冲基金) |
| **可规模化** | 极高 | 高 | 有限 |
| **透明度** | 完全透明 | 规则透明 | 黑盒/非公开 |

---

## 四、 中美市场差异

| 特点 | A股市场 | 美股市场 |
| :--- | :--- | :--- |
| **Alpha 环境** | **Alpha 丰富**。散户多，定价不效率。 | **Beta 驱动**。机构化程度高，Alpha 难求。 |
| **获取难度** | 较高但相对稳健（波动大）。 | 极高（高度有效市场）。 |
| **中性策略** | 虽有效但受限于对冲工具成本。 | 工具极其丰富，流动性极佳。 |

---

## 五、 Python 实战：计算 Alpha 和 Beta

使用 `statsmodels` 进行线性回归计算：

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

def calculate_alpha_beta(portfolio_returns, market_returns, rf_rate=0.02/252):
    """
    计算日频 Alpha 和 Beta
    """
    # 1. 计算超额收益
    y = portfolio_returns - rf_rate
    x = market_returns - rf_rate
    
    # 2. 线性回归: y = alpha + beta * x
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    
    alpha = model.params[0]
    beta = model.params[1]
    
    return alpha, beta

# 示例数据
# alpha_daily, beta = calculate_alpha_beta(returns_df['strategy'], returns_df['benchmark'])
# print(f"Daily Alpha: {alpha_daily:.4f}, Beta: {beta:.2f}")
```

---

## 六、 顶级量化基金的追求

- **Renaissance (文艺复兴)**: Medallion 基金以 **低 Beta, 极高 Alpha** 闻名。其逻辑是寻找不随大盘波动的“统计学套利”机会。
- **量化中性策略**: 通过做多 Alpha 组合，同时做空对应规模的期指（Beta 归零），从而实现在不论大盘涨跌的情况下都能稳定获利。

---

> **总结**: 对于散户，关注 Beta 带来的长期增长；对于量化研究者，核心任务是利用原始数据挖掘具有预测能力的 **Alpha 信号**。