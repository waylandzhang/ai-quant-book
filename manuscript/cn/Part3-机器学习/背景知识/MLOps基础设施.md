# MLOps 基础设施

> **模型一旦部署，就开始衰退。MLOps 不是可选的高级配置，而是量化策略存活的基础设施。**

---

## 从"模型跑通了"到"模型能用"

2023 年，一位量化研究员完成了一个动量策略模型：

- 回测夏普 1.8
- IC 均值 0.04
- 代码整洁，测试通过

他兴奋地部署到生产环境。

**三个月后**：

- 第一个月：夏普 1.2（"市场不好"）
- 第二个月：夏普 0.4（"再观察一下"）
- 第三个月：夏普 -0.3（"模型失效了？"）

**发生了什么？**

调查发现：
1. 生产环境用的特征计算代码和回测不同，某个 bug 导致 RSI 计算偏移了一天
2. 模型版本混乱，不确定当前运行的是哪个版本
3. 无法回溯问题，因为没有保存特征快照和模型输入
4. 发现问题时，已经不知道什么时候开始出错的

**教训**：模型研发只是开始，**可复现性、版本管理、漂移监控**才是生产系统的核心。这就是 MLOps。

---

## 一、为什么量化需要 MLOps？

### 量化特有的挑战

| 传统 ML | 量化 ML |
|---------|---------|
| 模型部署后相对稳定 | 市场结构持续变化，模型必然衰退 |
| 数据分布相对固定 | 金融数据高度非平稳 |
| 模型错误影响用户体验 | 模型错误直接导致资金损失 |
| 可以离线批量预测 | 需要实时推理，延迟敏感 |
| 特征来自稳定数据源 | 特征来自多个供应商，可能延迟或缺失 |

### MLOps 的三大支柱

```
量化 MLOps = Feature Store + Model Registry + Drift Monitor
             （特征库）      （模型注册表）     （漂移监控）

作用：
1. Feature Store  → 保证回测和实盘特征一致（可复现性）
2. Model Registry → 追踪模型版本和性能（可审计性）
3. Drift Monitor  → 检测模型衰退（及时止损）
```

---

## 二、Feature Store（特征仓库）

### 核心问题：Point-in-Time 正确性

量化中最隐蔽的 bug 是**前瞻偏差（Look-ahead Bias）**。

```
错误示例（前瞻偏差）：

2024-01-15 的训练样本：
  特征：RSI = 65（用了 2024-01-15 当天的收盘价计算）
  标签：明天涨跌

问题：
  实际上 2024-01-15 收盘价要等到 16:00 才知道
  但 RSI 计算用了这个值
  → 模型学到了"未来信息"

正确做法：
  2024-01-15 的训练样本：
    特征：用 2024-01-14 收盘价计算的 RSI
    标签：2024-01-15 → 2024-01-16 的涨跌
```

Feature Store 的核心能力就是确保 **Point-in-Time 查询**：给定任意历史时间点，返回**当时已知**的特征值。

### 双时间戳设计

```
特征事件表 (feature_events)：
┌─────────────┬──────────────┬────────────────┬────────────────┬─────────┐
│ entity_key  │ feature_name │ event_time     │ ingest_time    │ value   │
├─────────────┼──────────────┼────────────────┼────────────────┼─────────┤
│ AAPL.NASDAQ │ momentum_5d  │ 2024-01-15     │ 2024-01-15 20:00 │ 0.035 │
│ AAPL.NASDAQ │ rsi_14       │ 2024-01-15     │ 2024-01-15 20:00 │ 62.5  │
└─────────────┴──────────────┴────────────────┴────────────────┴─────────┘

两个时间戳的含义：
- event_time：特征对应的业务时间（如"这是 2024-01-15 的 RSI"）
- ingest_time：特征写入系统的时间（如"20:00 才计算完成"）

Point-in-Time 查询规则：
  WHERE event_time <= as_of_time AND ingest_time <= as_of_time
```

**为什么需要两个时间戳？**

```
场景：回测 2024-01-16 09:30 的交易决策

如果只用 event_time：
  查询：event_time <= '2024-01-16 09:30'
  可能返回 event_time='2024-01-15' 但 ingest_time='2024-01-16 22:00' 的数据
  → 前瞻偏差！

正确的双时间戳查询：
  查询：event_time <= '2024-01-16 09:30' AND ingest_time <= '2024-01-16 09:30'
  只返回当时已经可用的特征
```

### 数据库设计（TimescaleDB）

```sql
-- TimescaleDB 是专为时序数据优化的 PostgreSQL 扩展
CREATE TABLE IF NOT EXISTS feature_events (
    entity_key       TEXT NOT NULL,               -- 如 'AAPL.NASDAQ'
    feature_name     TEXT NOT NULL,               -- 特征名
    feature_version  INT  NOT NULL DEFAULT 1,     -- 版本（计算逻辑变更时升级）

    event_time       TIMESTAMPTZ NOT NULL,        -- 业务时间
    value_double     DOUBLE PRECISION,            -- 数值型特征
    value_json       JSONB,                       -- 复杂特征（向量等）

    ingest_time      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 可追溯性
    producer         TEXT,                        -- 生产者（如 'momentum_job'）
    producer_version TEXT,                        -- 代码版本（git SHA）
    run_id           TEXT,                        -- 任务 ID

    PRIMARY KEY (entity_key, feature_name, feature_version, event_time)
);

-- 转为时序表，自动分区
SELECT create_hypertable('feature_events', 'event_time', if_not_exists => TRUE);

-- 最新特征查询优化
CREATE INDEX IF NOT EXISTS idx_feature_events_latest
    ON feature_events (entity_key, feature_name, feature_version, event_time DESC);

-- 压缩策略（7天后压缩，节省 90%+ 空间）
ALTER TABLE feature_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'entity_key, feature_name, feature_version',
    timescaledb.compress_orderby = 'event_time DESC'
);
SELECT add_compression_policy('feature_events', INTERVAL '7 days');
```

### Python 实现

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

@dataclass
class FeatureValue:
    """查询返回的特征值"""
    entity_key: str
    feature_name: str
    feature_version: int
    event_time: datetime
    value: float | dict[str, Any]


class FeatureStore:
    """
    TimescaleDB-backed Feature Store

    核心功能：
    1. write_features: 写入特征
    2. get_latest: 获取最新特征值
    3. get_point_in_time: Point-in-Time 批量查询（训练集构建）
    """

    def __init__(self, conninfo: str, producer: str | None = None):
        self._conninfo = conninfo
        self._producer = producer

    def write_features(
        self,
        entity_key: str,
        timestamp: datetime,
        features: dict[str, float],
        *,
        feature_version: int = 1,
        availability_lag: timedelta | None = None,
    ) -> int:
        """
        写入特征值

        Args:
            entity_key: 实体标识（如 'AAPL.NASDAQ'）
            timestamp: 特征的业务时间（event_time）
            features: 特征字典 {feature_name: value}
            feature_version: 特征版本（计算逻辑变更时升级）
            availability_lag: 数据可用性延迟（回填时使用）
                如果某特征需要 T+1 才能获取，设置 availability_lag=timedelta(days=1)
                这样 ingest_time = event_time + 1 day

        Returns:
            写入的特征数量
        """
        if not features:
            return 0

        # 计算 ingest_time
        ingest_time = datetime.now()
        if availability_lag is not None:
            ingest_time = timestamp + availability_lag

        # 构建批量插入（ON CONFLICT 保证幂等）
        sql = """
            INSERT INTO feature_events
                (entity_key, feature_name, feature_version, event_time, value_double, ingest_time, producer)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_key, feature_name, feature_version, event_time) DO NOTHING
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for name, value in features.items():
                    cur.execute(sql, [
                        entity_key, name, feature_version,
                        timestamp, float(value), ingest_time, self._producer
                    ])
            conn.commit()

        return len(features)

    def get_latest(
        self,
        entity_key: str,
        feature_names: list[str] | None = None,
        *,
        as_of: datetime | None = None,
    ) -> dict[str, FeatureValue]:
        """
        获取实体的最新特征值

        Args:
            entity_key: 实体标识
            feature_names: 要查询的特征列表（None 表示全部）
            as_of: Point-in-Time 时间点（None 表示当前）

        Returns:
            {feature_name: FeatureValue}
        """
        # 关键：双时间戳过滤
        sql = """
            SELECT DISTINCT ON (feature_name, feature_version)
                feature_name, feature_version, value_double, event_time
            FROM feature_events
            WHERE entity_key = %s
              AND feature_version = 1
        """
        params = [entity_key]

        # Point-in-Time 过滤
        if as_of is not None:
            sql += " AND event_time <= %s AND ingest_time <= %s"
            params.extend([as_of, as_of])

        # 特征名过滤
        if feature_names:
            sql += " AND feature_name = ANY(%s)"
            params.append(feature_names)

        sql += " ORDER BY feature_name, feature_version, event_time DESC"

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return {
            row[0]: FeatureValue(
                entity_key=entity_key,
                feature_name=row[0],
                feature_version=row[1],
                event_time=row[3],
                value=row[2],
            )
            for row in rows
        }

    def get_point_in_time(
        self,
        entity_times: list[tuple[str, datetime]],
        feature_names: list[str] | None = None,
    ) -> list[FeatureValue]:
        """
        批量 Point-in-Time 查询（构建训练集的核心方法）

        Args:
            entity_times: [(entity_key, as_of_time), ...]
            feature_names: 要查询的特征列表

        Returns:
            对于每个 (entity, time) 对，返回当时可用的最新特征
        """
        if not entity_times:
            return []

        # 使用 CTE 和 DISTINCT ON 实现高效 PIT 查询
        values_sql = ", ".join(["(%s, %s)"] * len(entity_times))
        params = []
        for entity_key, as_of_time in entity_times:
            params.extend([entity_key, as_of_time])

        sql = f"""
        WITH entity_times(entity_key, as_of_time) AS (
            VALUES {values_sql}
        )
        SELECT DISTINCT ON (et.entity_key, fe.feature_name)
            et.entity_key,
            et.as_of_time,
            fe.feature_name,
            fe.feature_version,
            fe.value_double,
            fe.event_time AS feature_time
        FROM entity_times et
        JOIN feature_events fe
            ON fe.entity_key = et.entity_key
           AND fe.event_time <= et.as_of_time
           AND fe.ingest_time <= et.as_of_time
        WHERE fe.feature_version = 1
        ORDER BY et.entity_key, fe.feature_name, fe.event_time DESC
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            FeatureValue(
                entity_key=row[0],
                feature_name=row[2],
                feature_version=row[3],
                event_time=row[5],
                value=row[4],
            )
            for row in rows
        ]
```

### 使用示例

```python
# 初始化
store = FeatureStore(
    conninfo="postgres://localhost:5432/trading",
    producer="momentum_job_v2"
)

# 写入特征
store.write_features(
    entity_key="AAPL.NASDAQ",
    timestamp=datetime(2024, 1, 15, 16, 0),  # 收盘时间
    features={
        "momentum_5d": 0.035,
        "rsi_14": 62.5,
        "volume_ratio": 1.15,
    }
)

# 实时推理：获取最新特征
latest = store.get_latest("AAPL.NASDAQ", ["momentum_5d", "rsi_14"])
print(f"最新 RSI: {latest['rsi_14'].value}")

# 构建训练集：Point-in-Time 查询
training_dates = [
    ("AAPL.NASDAQ", datetime(2024, 1, 10, 9, 30)),
    ("AAPL.NASDAQ", datetime(2024, 1, 11, 9, 30)),
    ("AAPL.NASDAQ", datetime(2024, 1, 12, 9, 30)),
    ("MSFT.NASDAQ", datetime(2024, 1, 10, 9, 30)),
    ("MSFT.NASDAQ", datetime(2024, 1, 11, 9, 30)),
]

features = store.get_point_in_time(training_dates, ["momentum_5d", "rsi_14"])
# 返回每个时间点当时可用的特征值，不会有前瞻偏差
```

---

## 三、Model Registry（模型注册中心）

### 为什么需要模型注册？

```
场景：模型表现下降，需要排查

没有注册中心：
  - "现在跑的是哪个版本？" → 不知道
  - "这个版本的参数是什么？" → 文件里找
  - "上个版本在哪？" → 可能被覆盖了
  - "这个版本的回测表现是多少？" → 重新跑

有注册中心：
  SELECT * FROM models WHERE name = 'momentum_v2';
  → 版本、参数、指标、训练时间、代码版本，一目了然
```

### 数据库设计

```sql
-- 模型元数据
CREATE TABLE IF NOT EXISTS models (
    model_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL,
    version       INT NOT NULL,
    strategy_type TEXT,                  -- 'momentum', 'mean_reversion', etc.
    description   TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, version)
);

-- 模型指标
CREATE TABLE IF NOT EXISTS model_metrics (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id      UUID REFERENCES models(model_id),
    metric_name   TEXT NOT NULL,         -- 'sharpe_ratio', 'ic', 'max_drawdown'
    value         DOUBLE PRECISION,
    dataset_type  TEXT,                  -- 'train', 'val', 'test', 'backtest', 'live'
    evaluated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 模型工件（权重文件等）
CREATE TABLE IF NOT EXISTS model_artifacts (
    artifact_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id      UUID REFERENCES models(model_id),
    artifact_path TEXT NOT NULL,         -- 's3://models/momentum_v2/weights.pkl'
    artifact_type TEXT,                  -- 'weights', 'config', 'scaler', 'onnx'
    checksum      TEXT,                  -- SHA256
    size_bytes    BIGINT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 训练运行记录
CREATE TABLE IF NOT EXISTS model_training_runs (
    run_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id      UUID REFERENCES models(model_id),
    params        JSONB,                 -- 训练超参数
    dataset_start TIMESTAMPTZ,
    dataset_end   TIMESTAMPTZ,
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    status        TEXT DEFAULT 'running' -- 'running', 'completed', 'failed'
);
```

### Python 实现

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID
import hashlib
import json


@dataclass
class ModelInfo:
    """模型元数据"""
    model_id: UUID
    name: str
    version: int
    strategy_type: str | None
    description: str | None
    created_at: datetime


@dataclass
class ModelWithMetrics:
    """模型及其指标"""
    model: ModelInfo
    metrics: dict[str, float]  # {metric_name_dataset: value}


class ModelRegistry:
    """
    模型注册中心

    功能：
    1. register_model: 注册新模型版本
    2. log_metrics: 记录评估指标
    3. log_artifact: 记录模型工件
    4. get_best_model: 获取最佳模型
    """

    def __init__(self, dsn: str):
        self.dsn = dsn

    def register_model(
        self,
        name: str,
        strategy_type: str | None = None,
        params: dict | None = None,
        description: str | None = None,
        version: int | None = None,
    ) -> UUID:
        """
        注册新模型版本

        Args:
            name: 模型名称（如 'momentum_v2'）
            strategy_type: 策略类型
            params: 训练参数
            description: 描述
            version: 版本号（None 则自动递增）

        Returns:
            模型 UUID
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 自动版本号
                if version is None:
                    cur.execute(
                        "SELECT COALESCE(MAX(version), 0) + 1 FROM models WHERE name = %s",
                        (name,)
                    )
                    version = cur.fetchone()[0]

                # 插入模型
                cur.execute(
                    """
                    INSERT INTO models (name, version, strategy_type, description)
                    VALUES (%s, %s, %s, %s)
                    RETURNING model_id
                    """,
                    (name, version, strategy_type, description)
                )
                model_id = cur.fetchone()[0]

                # 记录训练参数
                if params:
                    cur.execute(
                        """
                        INSERT INTO model_training_runs (model_id, params, started_at, status)
                        VALUES (%s, %s, %s, 'completed')
                        """,
                        (model_id, json.dumps(params), datetime.now())
                    )

            conn.commit()
            return model_id

    def log_metrics(
        self,
        model_id: UUID,
        metrics: dict[str, float],
        dataset_type: str | None = None,
    ) -> None:
        """
        记录模型指标

        Args:
            model_id: 模型 UUID
            metrics: {指标名: 值}，如 {'sharpe_ratio': 1.5, 'ic': 0.04}
            dataset_type: 数据集类型（'train', 'val', 'test', 'backtest', 'live'）
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for metric_name, value in metrics.items():
                    cur.execute(
                        """
                        INSERT INTO model_metrics (model_id, metric_name, value, dataset_type)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (model_id, metric_name, value, dataset_type)
                    )
            conn.commit()

    def log_artifact(
        self,
        model_id: UUID,
        path: str | Path,
        artifact_type: str | None = None,
    ) -> UUID:
        """
        记录模型工件

        Args:
            model_id: 模型 UUID
            path: 工件路径（本地或 S3）
            artifact_type: 类型（'weights', 'config', 'scaler'）

        Returns:
            工件 UUID
        """
        path = Path(path)
        checksum = None
        size_bytes = None

        if path.exists():
            size_bytes = path.stat().st_size
            # 计算 SHA256
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            checksum = sha256.hexdigest()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_artifacts
                        (model_id, artifact_path, artifact_type, checksum, size_bytes)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING artifact_id
                    """,
                    (model_id, str(path), artifact_type, checksum, size_bytes)
                )
                artifact_id = cur.fetchone()[0]
            conn.commit()
            return artifact_id

    def get_model(self, name: str, version: int | None = None) -> ModelInfo | None:
        """获取模型（默认最新版本）"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if version is None:
                    cur.execute(
                        """
                        SELECT model_id, name, version, strategy_type, description, created_at
                        FROM models WHERE name = %s
                        ORDER BY version DESC LIMIT 1
                        """,
                        (name,)
                    )
                else:
                    cur.execute(
                        """
                        SELECT model_id, name, version, strategy_type, description, created_at
                        FROM models WHERE name = %s AND version = %s
                        """,
                        (name, version)
                    )

                row = cur.fetchone()
                if row:
                    return ModelInfo(*row)
                return None

    def get_best_model(
        self,
        strategy_type: str,
        metric_name: str,
        dataset_type: str = "test",
        higher_is_better: bool = True,
    ) -> ModelWithMetrics | None:
        """
        获取指定策略类型下表现最好的模型

        Args:
            strategy_type: 策略类型
            metric_name: 排序指标（如 'sharpe_ratio'）
            dataset_type: 数据集类型
            higher_is_better: 是否越高越好

        Returns:
            最佳模型及其指标
        """
        order = "DESC" if higher_is_better else "ASC"

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT m.model_id, m.name, m.version, m.strategy_type,
                           m.description, m.created_at, mm.value
                    FROM models m
                    JOIN model_metrics mm ON m.model_id = mm.model_id
                    WHERE m.strategy_type = %s
                      AND mm.metric_name = %s
                      AND mm.dataset_type = %s
                    ORDER BY mm.value {order}
                    LIMIT 1
                    """,
                    (strategy_type, metric_name, dataset_type)
                )

                row = cur.fetchone()
                if not row:
                    return None

                model = ModelInfo(*row[:6])

                # 获取该模型的所有指标
                cur.execute(
                    """
                    SELECT metric_name, value, dataset_type
                    FROM model_metrics
                    WHERE model_id = %s
                    """,
                    (model.model_id,)
                )

                metrics = {
                    f"{r[0]}_{r[2]}": r[1]
                    for r in cur.fetchall()
                }

                return ModelWithMetrics(model=model, metrics=metrics)
```

### 使用示例

```python
registry = ModelRegistry(dsn="postgres://localhost:5432/trading")

# 注册新模型
model_id = registry.register_model(
    name="momentum_xgb",
    strategy_type="momentum",
    params={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "features": ["ret_5d", "ret_20d", "vol_20d", "rsi_14"],
    },
    description="XGBoost momentum model with RSI features"
)

# 记录回测指标
registry.log_metrics(model_id, {
    "sharpe_ratio": 1.65,
    "total_return": 0.28,
    "max_drawdown": 0.12,
    "ic": 0.042,
    "ir": 0.85,
}, dataset_type="backtest")

# 记录测试集指标
registry.log_metrics(model_id, {
    "sharpe_ratio": 1.35,
    "ic": 0.035,
}, dataset_type="test")

# 保存模型工件
registry.log_artifact(model_id, "models/momentum_xgb_v3.pkl", "weights")
registry.log_artifact(model_id, "models/momentum_xgb_v3_config.json", "config")

# 获取最佳动量模型
best = registry.get_best_model("momentum", "sharpe_ratio", "test")
if best:
    print(f"最佳模型: {best.model.name} v{best.model.version}")
    print(f"测试集夏普: {best.metrics.get('sharpe_ratio_test', 'N/A')}")
```

---

## 四、Drift Monitor（漂移监控）

### 漂移的三个维度

| 维度 | 检测指标 | 含义 | 阈值建议 |
|------|---------|------|---------|
| **数据漂移** | PSI | 特征分布变化 | < 0.10 正常，> 0.25 严重 |
| **预测漂移** | IC | 预测与实际收益相关性 | > 0.02 正常，< 0.01 严重 |
| **性能漂移** | 滚动夏普 | 策略收益风险比 | > 0.5 正常，< 0 严重 |

### 核心指标计算

```python
import numpy as np
from scipy.stats import spearmanr


def calculate_ic(signals: np.ndarray, returns: np.ndarray) -> float:
    """
    计算 Information Coefficient

    IC = Spearman相关系数(预测信号, 实际收益)

    解读：
    - IC > 0.05: 优秀
    - IC 0.02-0.05: 良好
    - IC < 0.02: 需要关注
    - IC < 0: 模型可能有问题
    """
    if len(signals) < 2:
        return 0.0

    # 移除 NaN
    mask = ~(np.isnan(signals) | np.isnan(returns))
    signals, returns = signals[mask], returns[mask]

    if len(signals) < 2:
        return 0.0

    ic, _ = spearmanr(signals, returns)
    return float(ic) if not np.isnan(ic) else 0.0


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    计算 Population Stability Index (PSI)

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    解读：
    - PSI < 0.10: 分布稳定
    - PSI 0.10-0.25: 轻度漂移，需观察
    - PSI > 0.25: 显著漂移，需要行动
    """
    eps = 1e-6

    # 基于基准分布创建分箱
    _, bin_edges = np.histogram(expected, bins=bins)

    # 计算各箱比例
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    # PSI 公式
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def calculate_sharpe(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    计算年化夏普比率

    Sharpe = mean(returns) / std(returns) * sqrt(252)
    """
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret < 1e-10:
        return 0.0

    return (mean_ret / std_ret) * np.sqrt(periods_per_year)
```

### Drift Monitor 实现

```python
from dataclasses import dataclass
from datetime import date


@dataclass
class DriftMetrics:
    """每日漂移指标"""
    date: date
    strategy_id: str
    ic: float | None = None
    ic_5d_avg: float | None = None
    psi: float | None = None
    sharpe_5d: float | None = None
    sharpe_20d: float | None = None
    ic_alert: bool = False
    psi_alert: bool = False
    sharpe_alert: bool = False


@dataclass
class AlertConfig:
    """告警阈值配置"""
    ic_warning: float = 0.02
    ic_critical: float = 0.01
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    sharpe_warning: float = 0.5
    sharpe_critical: float = 0.0


class DriftMonitor:
    """
    漂移监控服务

    每日运行，计算 IC、PSI、夏普等指标，存储到数据库，触发告警。
    """

    def __init__(self, dsn: str, strategy_id: str = "default"):
        self.dsn = dsn
        self.strategy_id = strategy_id
        self.config = AlertConfig()

    def calculate_metrics(self, target_date: date) -> DriftMetrics:
        """计算指定日期的漂移指标"""
        metrics = DriftMetrics(date=target_date, strategy_id=self.strategy_id)

        # 获取信号和收益
        signals, returns = self._get_signals_and_returns(target_date)
        if len(signals) > 0:
            metrics.ic = calculate_ic(signals, returns)

        # 获取历史收益计算夏普
        daily_returns = self._get_daily_returns(lookback_days=60)
        if len(daily_returns) >= 5:
            metrics.sharpe_5d = calculate_sharpe(daily_returns[-5:])
        if len(daily_returns) >= 20:
            metrics.sharpe_20d = calculate_sharpe(daily_returns[-20:])

        # 检查告警
        if metrics.ic is not None:
            metrics.ic_alert = metrics.ic < self.config.ic_critical
        if metrics.psi is not None:
            metrics.psi_alert = metrics.psi > self.config.psi_critical
        if metrics.sharpe_20d is not None:
            metrics.sharpe_alert = metrics.sharpe_20d < self.config.sharpe_critical

        return metrics

    def save_metrics(self, metrics: DriftMetrics) -> None:
        """保存指标到数据库"""
        sql = """
            INSERT INTO drift_metrics (
                date, strategy_id, ic, ic_5d_avg, psi, sharpe_5d, sharpe_20d,
                ic_alert, psi_alert, sharpe_alert
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, strategy_id) DO UPDATE SET
                ic = EXCLUDED.ic,
                psi = EXCLUDED.psi,
                sharpe_5d = EXCLUDED.sharpe_5d,
                sharpe_20d = EXCLUDED.sharpe_20d,
                ic_alert = EXCLUDED.ic_alert,
                psi_alert = EXCLUDED.psi_alert,
                sharpe_alert = EXCLUDED.sharpe_alert
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [
                    metrics.date, metrics.strategy_id,
                    metrics.ic, metrics.ic_5d_avg, metrics.psi,
                    metrics.sharpe_5d, metrics.sharpe_20d,
                    metrics.ic_alert, metrics.psi_alert, metrics.sharpe_alert,
                ])
            conn.commit()

    def run_daily(self, target_date: date | None = None) -> DriftMetrics:
        """每日漂移监控任务"""
        if target_date is None:
            target_date = date.today()

        print(f"Running drift monitoring for {target_date}")

        metrics = self.calculate_metrics(target_date)
        self.save_metrics(metrics)

        # 输出告警
        if metrics.ic_alert:
            print(f"[ALERT] IC = {metrics.ic:.4f} below threshold {self.config.ic_critical}")
        if metrics.psi_alert:
            print(f"[ALERT] PSI = {metrics.psi:.4f} above threshold {self.config.psi_critical}")
        if metrics.sharpe_alert:
            print(f"[ALERT] Sharpe = {metrics.sharpe_20d:.4f} below threshold {self.config.sharpe_critical}")

        return metrics
```

### 告警响应矩阵

| 告警类型 | 严重程度 | 建议行动 |
|---------|---------|---------|
| IC < 0.02 连续 5 天 | 警告 | 检查特征计算是否正常 |
| IC < 0.01 | 严重 | 降低仓位 50%，启动模型诊断 |
| IC < 0 连续 3 天 | 紧急 | 暂停策略，人工审核 |
| PSI > 0.10 | 警告 | 监控后续变化 |
| PSI > 0.25 | 严重 | 触发再训练流程 |
| Sharpe < 0.5 连续 10 天 | 警告 | 检查市场状态 |
| Sharpe < 0 连续 5 天 | 严重 | 降低仓位，准备再训练 |

---

## 五、集成：从研究到生产

### 完整工作流

```
┌─────────────────────────────────────────────────────────────────────┐
│                          研究阶段                                    │
├─────────────────────────────────────────────────────────────────────┤
│  1. 特征开发                                                         │
│     └─→ 写入 Feature Store（设置正确的 availability_lag）            │
│                                                                      │
│  2. 构建训练集                                                       │
│     └─→ Feature Store.get_point_in_time()                           │
│     └─→ 导出 Parquet（不可变快照）                                   │
│                                                                      │
│  3. 模型训练                                                         │
│     └─→ 记录参数、代码版本                                           │
│     └─→ 注册到 Model Registry                                        │
│                                                                      │
│  4. 回测评估                                                         │
│     └─→ log_metrics(dataset_type='backtest')                        │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          部署阶段                                    │
├─────────────────────────────────────────────────────────────────────┤
│  5. 模型选择                                                         │
│     └─→ get_best_model(strategy_type, metric, dataset_type='test') │
│                                                                      │
│  6. 加载模型                                                         │
│     └─→ 从 artifact_path 加载权重                                   │
│     └─→ 验证 checksum                                               │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          运行阶段                                    │
├─────────────────────────────────────────────────────────────────────┤
│  7. 实时推理                                                         │
│     └─→ Feature Store.get_latest() 获取特征                         │
│     └─→ 模型预测                                                    │
│     └─→ 输出信号                                                    │
│                                                                      │
│  8. 每日监控                                                         │
│     └─→ Drift Monitor 计算 IC/PSI/Sharpe                            │
│     └─→ 触发告警                                                    │
│                                                                      │
│  9. 再训练（如需要）                                                 │
│     └─→ 回到步骤 2                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 可复现性检查清单

| 检查项 | 如何实现 | 验证方法 |
|-------|---------|---------|
| **代码版本** | 记录 git SHA | `producer_version` 字段 |
| **特征版本** | `feature_version` 列 | 查询时指定版本 |
| **训练数据** | Parquet 快照 + fingerprint | 重新训练应得到相同结果 |
| **模型参数** | `model_training_runs.params` | JSON 存储 |
| **模型权重** | `model_artifacts.checksum` | SHA256 校验 |
| **评估指标** | `model_metrics` 表 | 按时间追溯 |

### 每日运维脚本示例

```python
from datetime import date, datetime

def daily_mlops_job(
    feature_store: FeatureStore,
    model_registry: ModelRegistry,
    drift_monitor: DriftMonitor,
    strategy_id: str,
):
    """每日 MLOps 任务"""
    today = date.today()
    print(f"=== MLOps Daily Job: {today} ===")

    # 1. 特征健康检查
    print("\n[1] Feature Health Check")
    latest = feature_store.get_latest("AAPL.NASDAQ")
    for name, fv in latest.items():
        age_hours = (datetime.now() - fv.event_time).total_seconds() / 3600
        if age_hours > 24:
            print(f"  WARNING: {name} is {age_hours:.1f} hours old")
        else:
            print(f"  OK: {name} updated {age_hours:.1f} hours ago")

    # 2. 模型状态检查
    print("\n[2] Model Status Check")
    current_model = model_registry.get_model("momentum_xgb")
    if current_model:
        print(f"  Current: {current_model.name} v{current_model.version}")
        print(f"  Created: {current_model.created_at}")

    # 3. 漂移监控
    print("\n[3] Drift Monitoring")
    drift_metrics = drift_monitor.run_daily(today)
    print(f"  IC: {drift_metrics.ic}")
    print(f"  Sharpe (20d): {drift_metrics.sharpe_20d}")

    # 4. 决策
    if drift_metrics.ic_alert or drift_metrics.sharpe_alert:
        print("\n[ACTION REQUIRED] Consider retraining or reducing position size")
    else:
        print("\n[OK] All metrics within normal range")


# 定时任务（如 cron）
# 0 6 * * * python -c "from mlops import daily_mlops_job; daily_mlops_job(...)"
```

---

## 延伸阅读

- *Machine Learning Design Patterns* by Lakshmanan, Robinson, and Munn
- [模型漂移与再训练策略](模型漂移与再训练.md)
