# AI Quantitative Trading: From Zero to One

# 《AI量化交易从0到1》

> **用多智能体架构构建可落地的量化交易系统**

[![Status: WIP](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)](.)
[![Chinese: Complete](https://img.shields.io/badge/中文-已完成-green)](./manuscript/cn/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**作者**: [Wayland Zhang](https://waylandz.com)

---

## 这本书讲什么？

**不是策略圣杯，而是教你构建可落地的量化交易系统。**

市面上的量化教程大多停留在：
- 某个回测框架的 API 翻译
- 技术指标的堆砌和参数优化
- 过拟合的"神奇策略"展示

这些只能让你"玩"量化，不能让你"做"量化。

真正的量化系统需要回答：
- **数据从哪来？** API 限流、缺失值、复权、时区
- **回测怎么不骗自己？** 未来函数、过拟合、交易成本
- **为什么单一模型不够？** Regime 变化、信号冲突、风险分散
- **如何控制风险？** 止损、仓位、因子暴露、熔断机制
- **怎么上生产？** 执行滑点、系统监控、故障恢复

这本书用**多智能体架构**回答这些问题：不同的 Agent 负责不同的职责（信号、风控、执行），协作完成交易决策。

---

## 项目状态

| 语言 | 状态 | 进度 | 链接 |
|------|------|------|------|
| 🇨🇳 中文 | ✅ 完成 | 22 课 + 21 背景知识 + 4 附录 | [阅读](./manuscript/cn/) |
| 🇺🇸 English | 📝 计划中 | - | - |
| 🇯🇵 日本語 | 📝 计划中 | - | - |

---

## 目录结构

```
ai-quant-book/
├── manuscript/              # 源稿（Markdown）
│   ├── cn/                  # 中文版（已完成，陆续随视频同步开源中）
│   │   ├── Part1-快速体验/  # 30分钟建立直觉
│   │   ├── Part2-量化基础/  # 市场、数据、策略、回测
│   │   ├── Part3-机器学习/  # 从模型到 Agent
│   │   ├── Part4-多智能体/  # 架构、Regime、风控
│   │   ├── Part5-生产与实战/ # 执行、运维、实战
│   │   └── Resources & Links/
│   ├── en/                  # English (planned)
│   └── jp/                  # 日本語 (planned)
├── book/                    # 发布版本（未来）
└── README.md                # 本文件
```

---

## 内容概览

全书分为 **5 个部分、22 课**：

| 部分 | 主题 | 课程数 | 核心内容 |
|------|------|--------|----------|
| **Part 1** | 快速体验 | 1 | 30分钟部署市场分析 Agent |
| **Part 2** | 量化基础 | 7 | 市场结构、统计基础、策略范式、数据工程、回测陷阱 |
| **Part 3** | 机器学习 | 2 | 监督学习应用、从模型到 Agent 的转变 |
| **Part 4** | 多智能体 | 7 | 架构设计、Regime 检测、LLM 应用、风控、组合管理 |
| **Part 5** | 生产实战 | 5 | 成本建模、执行系统、运维监控、项目实战 |

### 背景知识（21篇）

深入讲解核心概念：Alpha与Beta、订单簿机制、市场微结构、特征工程陷阱、机器学习在金融中的限制、组合优化等。

### 附录（4篇）

- 实盘交易记录标准
- 量化系统的12种死亡方式
- 人类决策与自动化边界
- 常见问题 FAQ

---

## 目标读者

| 读者类型 | 你会获得什么 |
|----------|-------------|
| **程序员想转量化** | 从零构建交易系统的完整路径 |
| **量化研究员** | 多智能体架构、生产级风控设计 |
| **投资者/PM** | 理解量化系统的能力边界和风险 |

### 前置要求

- **必需**：基本编程概念（Python 为主）
- **有帮助**：统计学基础、金融市场常识
- **不需要**：不需要机器学习或深度学习背景

---

## 快速开始

→ [**中文版完整目录**](./manuscript/cn/README.md)

### 推荐阅读路径

| 读者类型 | 推荐路径 |
|---------|---------|
| **零基础入门** | Part 1 → Part 2 全部 → Part 3 → Part 4 → Part 5 |
| **有编程基础** | 第01课 → Part 2 快速浏览 → Part 3-5 全部 |
| **有量化基础** | 第01课 → 第08课 → Part 3-5 全部 |
| **只关心架构** | 第01课 → 第10-17课 → 附录B |

---

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Meta Agent                              │
│            (Regime Detection + Task Routing)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌───────────┐ ┌───────────┐ ┌───────────┐
   │  Trend    │ │   Mean    │ │  Crisis   │
   │  Agent    │ │  Revert   │ │  Agent    │
   └───────────┘ └───────────┘ └───────────┘
         │             │             │
         └─────────────┼─────────────┘
                       ▼
                ┌─────────────┐
                │ Risk Agent  │◄───── 一票否决权
                └─────────────┘
```

---

## 风险声明

> **量化交易有风险，投资需谨慎。**

本书是教育性质，不构成投资建议。

- 书中策略仅供学习，不保证盈利
- 实盘交易前请充分理解风险
- 永远不要用你输不起的钱交易
- 历史表现不代表未来收益

---

## License

本书内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 协议。

---

> *"Whatever happens in the stock market today has happened before and will happen again."*
> — Jesse Livermore

> *"We search through historical data looking for anomalous patterns that we believe are predictive of future price action."*
> — Jim Simons
