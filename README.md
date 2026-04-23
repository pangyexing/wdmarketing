# wdmarketing

银行营销建模 —— 固定预算下的 Top-K 排序（面向正样本预测与召回）。

## 设计要点

- **两阶段解耦**：Stage 1 只做特征分析与筛选（PSI/IV/WOE/Lift/相关性），全程不训模型；Stage 2 基于选定的特征列表跑 Hyperopt + XGBoost，产出可部署的预测程序。
- **多产品通用**：每个产品一个 YAML（`configs/products/<name>.yaml`），与 `configs/global.yaml` 默认值合并。
- **分块无偏性**：3000+ 维特征按**列分块**流式分析，所有 PSI/IV/相关性与全表计算数学上等价（见 `tests/test_chunked_correctness.py`）。
- **分析 vs 训练严格分离**：缺失值规则（0/负/空视为缺失，默认填 -999）只在 Stage 2 训练期落盘；Stage 1 分析一律用 NaN-aware 计算，`-999` 绝不污染 PSI/IV/相关性统计。
- **三层相关性去重**：
  - 时间窗家族（`_7d/_30d/_90d/_all`）自动识别，家族内阈值 0.90
  - 业务语义簇（"机构数/还款金额/申请数"等）手动声明，簇内阈值 0.85
  - 其他全局 0.95 兜底
- **部署友好**：`validation_samples.csv` 使用**原始特征值**，部署人员直接把查库得到的原始行喂给 `predict.py`，缺失值处理在内部完成。

## 环境

```
Python 3.6.13
xgboost==1.5.2
shap, hyperopt, numpy<1.20, pandas<1.2, scikit-learn<0.25, matplotlib<3.4
pyyaml, jsonschema<4, dataclasses (backport)
```

安装：`pip install -r requirements.txt`

## 使用

### Stage 1 —— 特征分析与筛选

```bash
PYTHONPATH=src python3 scripts/run_analysis.py --product bank_marketing
```

产出：
- `artifacts/bank_marketing/analysis/report/summary.csv` — 综合排名主表（英文+中文列名）
- `artifacts/bank_marketing/analysis/report/index.html` — 浏览器本地打开可排序的单页汇总
- `artifacts/bank_marketing/analysis/per_feature/<feat>/{dist,woe,psi,missing,lift}.png` — 每特征五张图
- `artifacts/bank_marketing/selected_features/v1_auto.txt` — 自动推荐的特征列表

用户手工步骤：
```bash
cp artifacts/bank_marketing/selected_features/v1_auto.txt \
   artifacts/bank_marketing/selected_features/v2_manual_noleak.txt
# 编辑 v2 删除泄漏特征（如 UCI 的 duration）、添加备注
# 更新 configs/products/bank_marketing.yaml 里 selected_features.active_version
```

### Stage 2 —— 训练 + 部署产物

```bash
PYTHONPATH=src python3 scripts/run_training.py \
  --product bank_marketing --run-id prod01 \
  --features-version v2_manual_noleak
```

产出 `artifacts/bank_marketing/models/prod01/`：
- `booster.json` — XGBoost 原生模型（xgb 1.5.2）
- `feature_list.txt` — 特征顺序（含 `__isnan` 指示列）
- `missing_spec.json` — 训练期 fit 的缺失值规则与填充值（`predict.py` 只重放不重算）
- `predict.py` — 单文件自包含预测程序，只依赖 xgboost/numpy/pandas/argparse
- `validation_samples.csv` — 100 行**原始特征** + `y_true` + `y_pred_expected`（1e-6 容差自测）
- `importance.csv` — 特征重要性（gain/weight/cover + 中文列名）
- `metrics.json` / `metrics.md` — 全量指标：PR-AUC / ROC-AUC / KS / Precision@K / Lift@K / Top-K CVR
- `binned_lift_{train,valid,oot}.csv` — 10 分位 lift 表
- `plots/*.png` — ROC / PR / KS / gain / calibration / SHAP / 重要性柱状图
- `run_manifest.json` — 实验快照

### 部署

把 `artifacts/<product>/models/<run_id>/` 整个目录拷贝到目标机器，运行：

```bash
# 打 1e-6 级别一致性自测（强烈建议每次上线跑一次）
python predict.py --validate

# 打分
python predict.py predict --input raw_samples.csv --output scores.csv
```

部署人员只需理解：输入 CSV 是原始业务数据（允许 `0 / 负值 / 空 / 哨兵值 / NaN`），输出是 `row_index, score`。缺失值处理由 `predict.py` 内部完成。

## 关键配置说明

### 缺失值规则

全局默认（`configs/global.yaml`）：`0 / 负值 / 空` 视为缺失，填充 `-999`。

为什么是 -999：对"通常为正数"的业务特征，-999 < 任何真实值，XGBoost 可清晰地把它当作"缺失"桶来学。由于分析期走 NaN-aware 计算（`analysis_use_mask: true`），-999 **不会污染** PSI/IV/相关性统计。

针对真实业务数据，`configs/products/<name>.yaml` 里用 `missing.per_column` 覆盖特例（参考 `bank_marketing.yaml` 里对 `balance`/`previous`/`pdays` 的处理）。

可选策略：
- `fill_strategy: median / mean` —— 数据驱动，自动跳过 sanity check
- `fill_strategy: special` —— 保留哨兵值（如 `pdays=-1` 保留为独立 WOE 箱）
- `fill_strategy: keep_nan` —— 完全交给 XGBoost 原生 NaN 处理，不填充
- `generate_missing_indicator: true` —— 高缺失列自动生成 `<feat>__isnan` 二值特征

### 切分策略

`configs/global.yaml` 下 `training.split`：
- `strategy: stratified` —— 按 label 分层随机切（默认）
- `strategy: time` —— 要求 `data.time_column`（`yyyymmdd` 整数/字符串），按时间升序切，模拟真实上线

### 时间窗家族与语义簇

- **自动家族**：`txn_amt_7d / _30d / _90d / _all` 等由 `feature_groups.window_pattern` 自动识别，家族内最多保留 `max_per_family`（默认 2）个。
- **语义簇**：在 product YAML 的 `feature_groups.semantic_groups` 里手动声明（如"机构数/还款金额/申请数"），簇内最多保留 `max_keep` 个。

两种机制互相独立，均通过 `selector.py` 合并入综合打分。

## 测试

```bash
python3 -m pytest tests/ -v
```

关键测试：
- `test_chunked_correctness.py` —— 列分块 vs 全表 PSI/IV/相关性差 < 1e-10，**如失败则 Stage 1 结果全部不可信**
- `test_missing.py` —— 缺失值规则、sanity check、special 策略 roundtrip
- `test_ranking_metrics.py` —— Precision@K / Lift@K / KS 的极端情况

## 在 UCI 数据上的 sanity 结果

| 模型 | duration 在？ | Valid PR-AUC | Valid Lift@10% | OOT Lift@10% |
|---|---|---|---|---|
| smoke01 (v1_auto) | 是 | 0.60 | 5.08 | 4.96 |
| noleak (v2_manual_noleak) | 否 | 0.45 | 4.12 | 3.84 |

剔除泄漏特征 `duration` 后模型依旧保持 ≥ 3× Lift，说明 XGBoost 仍能从 `poutcome / pdays / contact` 等特征学出有效排序。

## 未来扩展

- 真实因果 Uplift 建模（`treatment_column` 已在 config 里预留）
- 更多产品的 `configs/products/*.yaml` 与 `*_columns.csv`
- MLflow / 实验管理（当前靠 `run_manifest.json` 落文件）
