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
Python 3.7
xgboost==1.5.0
shap, hyperopt, numpy<1.20, pandas<1.2, scikit-learn<0.25, matplotlib<3.4
pyyaml, jsonschema<4    # dataclasses is stdlib on 3.7
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
- `booster.json` / `booster.bin` — XGBoost 原生模型（xgb 1.5.0）。导出格式由 `export.model_format` 控制：`["json"]`（默认，文本可读）、`["bin"]`（原生二进制）或 `["json","bin"]`（两者皆出）。`predict.py` 会自动探测 `.bin`/`.json` 加载。
- `feature_list.txt` — 特征顺序（含 `__isnan` 指示列）
- `missing_spec.json` — 训练期 fit 的缺失值规则与填充值（`predict.py` 只重放不重算）
- `calibration.json` — valid 上拟合的 isotonic 校准查表（`export.calibration` 控制，默认开启；单模型排序不变，跨模型融合必须用校准分）。`predict.py` 的 `score` 列保持原始分，校准分输出在新增的 `score_calibrated` 列
- `predict.py` — 单文件自包含预测程序，只依赖 xgboost/numpy/pandas/argparse
- `validation_samples.csv` — 100 行**原始特征** + `y_true` + `y_pred_expected`（1e-6 容差自测；有校准时另含 `y_pred_calibrated_expected`，`--validate` 双路检查）
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

### xc 漏斗工作流（响应+资质双模型 → 校准融合漏斗分析 + 端到端基线）

**建模漏斗**只取两段式：响应模型用 `is_finish_task`（全量人群），资质模型在 `is_finish_task==1` 人群上分两个版本（V1 用 `is_credit_succ`：1 正 / 0 负；V2 用 `credit_1v1`：1/2/3 正、0 和 -1 负，合表时衍生为二值列 `is_credit_1v1`）。**分析漏斗**基于响应 × 资质的融合分，按全流程 `is_reg → is_finish_task → 授信` 评估各阶段与综合提升（`is_reg` 仅用于分析，不建模）。另有两个**端到端基线**（全量人群直推授信结果的单模型），用于对照"两段式融合是否真的更优"。

| 模型 | 人群 | 标签 |
|---|---|---|
| 响应 `xc_resp_finish` | 全量 | `is_finish_task` |
| 资质 V1 `xc_qual_finish` | is_finish_task==1 | `is_credit_succ`（1 正 / 0 负） |
| 资质 V2 `xc_qual_finish_1v1` | is_finish_task==1 | `is_credit_1v1`（credit_1v1 ∈ {1,2,3} 正；0、-1 负）+ 档位价值加权（790/290/120） |
| 端到端 V1 `xc_e2e_credit` | 全量 | `is_credit_succ`（基线对照） |
| 端到端 V2 `xc_e2e_credit_1v1` | 全量 | `is_credit_1v1`（基线对照，档位价值加权） |

xc 各产品统一配置 `tuner_objective: precision_at_k`（调参直接最大化 CV P@10% 而非全曲线 PR-AUC）与 `cv_strategy: time_forward`（前向扩窗时间折，与按时间切分的最终评估口径一致）。

```bash
# 1. 合表：产出 data/xc_full.csv（全量）+ xc_qual_finish.csv（is_finish_task==1，
#    含衍生标签 is_credit_1v1），并打印全流程分析漏斗（两个授信口径各一份）。
#    特征/标签文件时间列均为 dt（特征 yyyy-mm-dd 或 yyyymmdd，标签 yyyymmdd），归一成 int yyyymmdd 后按 (id, dt) join
PYTHONPATH=src python3 scripts/build_xc_dataset.py \
  --features data/xc_features.csv --labels data/xc_labels.csv

# 2. 每个模型各跑 Stage-1 + Stage-2（资质 V2 换 xc_qual_finish_1v1；
#    端到端基线换 xc_e2e_credit / xc_e2e_credit_1v1）
PYTHONPATH=src python3 scripts/run_analysis.py --product xc_resp_finish
PYTHONPATH=src python3 scripts/run_training.py --product xc_resp_finish --run-id r01
PYTHONPATH=src python3 scripts/run_analysis.py --product xc_qual_finish
PYTHONPATH=src python3 scripts/run_training.py --product xc_qual_finish --run-id q01
PYTHONPATH=src python3 scripts/run_analysis.py --product xc_e2e_credit
PYTHONPATH=src python3 scripts/run_training.py --product xc_e2e_credit --run-id e01

# 3. 融合漏斗分析：fused = 校准响应分 × 校准资质分（每个 bundle 的
#    calibration.json 在 valid 上拟合 isotonic，消除 scale_pos_weight 带来的
#    概率畸变）；fused_alpha = resp^α × qual^(1-α)，α 在 OOT 之前的拟合窗
#    （两模型 valid 期重叠段）网格搜索、最大化授信 lift@K。
#    默认仅在所有模型都未见过的 OOT 时段评估（可用 --start-dt/--end-dt 覆盖）
PYTHONPATH=src python3 scripts/run_funnel_eval.py \
  --resp-product xc_resp_finish --resp-run-id r01 \
  --qual-product xc_qual_finish --qual-run-id q01 \
  --e2e-product xc_e2e_credit --e2e-run-id e01
# 资质 V2：--qual-product xc_qual_finish_1v1 --qual-stage is_credit_1v1
#   --e2e-product xc_e2e_credit_1v1 --tier-values "1:120,2:290,3:790"
#   （--tier-values 额外输出 value_capture 行 = top-K 人均业务价值提升）
# 其他开关：--alpha 0.6 手动指定 / --no-alpha 跳过 α 融合 / --alpha-k 0.10 目标 K
```

产出 `artifacts/funnel_eval/<resp>__<qual>/funnel_eval.{csv,md}` + `fusion_spec.json`：
- **absolute 口径**：top-K 内 `is_reg / is_finish_task / 授信标签` 率 vs 全人群基线，授信行（`is_credit_succ` 或 `is_credit_1v1`，随 `--qual-stage`）即端到端**综合提升**；
- **conditional 口径**：top-K 内 reg、finish|reg、credit|finish 的逐段转化 vs 人群基线，定位提升来自哪一段；
- 同时给出 `fused / fused_alpha / resp_only / qual_only / e2e_only` 排序对照，量化融合与 α 加权增益、检验两段式是否优于端到端单模型；
- `fusion_spec.json` 是部署侧的执行契约：α、拟合/评估窗口、各 α 的拟合曲线、serving_formula（用两个 bundle 的 `score_calibrated` 列按公式融合）。

注意：中间/下游/平行结果列绝不能入特征（各 config 已 park 在 `id_columns`，如 `xc_qual_finish` 里的 `is_credit_1v1` 与原始 `credit_1v1`）。`xc_qual_finish_1v1` 的 `credit_1v1` 同时被 `training.sample_weight` 引用 —— 只作训练损失权重，调参 P@K、早停与上报指标均按人头不加权。

生产环境完整执行步骤（含检查点、验收标准、上线打分与监控）见 `docs/PRODUCTION_RUNBOOK.md`；
交互式漏斗探索（连续 top-K 提升曲线、资质 V1/V2 对比）见 `notebooks/10_funnel_eval_xc.ipynb`。

### Notebooks

`notebooks/` 按工作流编号。01–05 通过开头的 `PRODUCT` / `RUN_ID` 参数适配任意产品（默认 xc，非 xc 产品运行时 xc 专属格自动跳过）；00 与 06–09 面向 hzz / home_credit 工作流；10/11 为 xc 专属。04 及之后的内核需带 xgboost（conda `env_ml`）。

| notebook | 用途 | 对应 runbook |
|---|---|---|
| `00_hzz_raw_preprocess_check` | `preprocess_hzz_raw.py` 预处理产物 sanity check（12 day + 4 mon 原始表 → per-table / merged） | hzz 数据落地 |
| `01_data_overview` | 数据/合表总览：标签与档位分布、分析漏斗、标签成熟度、join 质检、缺失概览 | 第 1 步检查点 |
| `02_feature_analysis` | Stage-1 报告解读：rank_score 口径、lift 软门槛、PSI 稳定性、相关性簇、per-feature 图 | 第 2 步 |
| `03_feature_selection_review` | 筛选链路 v1_auto→v2_model→v3_manual diff、null importance 体检、跨 xc 产品清单重叠 | 第 2.5/3 步检查点 |
| `04_model_training` | Stage-2 交互训练：切分体检、hyperopt 轨迹、训练后退化/lift 快速诊断 | 第 4 步（探索） |
| `05_model_evaluation` | 训练产物验收：manifest/校准体检、退化检查、分位 lift、5 个 xc 模型横向汇总 | 第 4/5 步检查点 |
| `06_window_family_analysis` | 时间窗家族排序偏差诊断："越久覆盖率越高 → 排名越前"的定量分析与缓解开关评估 | Stage-1 复盘 |
| `07_stage2_funnel_audit` | Stage-2 候选→最终漏斗审计：探索 ranker 打分、剪枝前后对比 | Stage-2 复盘 |
| `08_seed_distribution_compare` | 种子人群 (ABCD) 特征分布对比：双基准 PSI，看输入端漂移 | 上线人群体检 |
| `09_seed_score_compare` | ABCD vs OOT 模型分数分布对比，看输出端漂移 | 上线人群体检 |
| `10_funnel_eval_xc` | 融合漏斗探索：连续 top-K 曲线、alpha 敏感性、conditional 分段、资质 V1/V2 对比 | 第 6 步（探索） |
| `11_xc_monitoring` | 上线后监控与复盘：分数/特征 PSI 漂移、上线窗口转化复盘（固定线上 α） | 第 8 步 |

## 关键配置说明

### 特征筛选漏斗（三个可选的模型信号，按需启用其一）

Stage-1 统计筛选（PSI/IV/Lift/相关性 → `v1_auto.txt`）之外，共有三个**模型驱动**的筛选机制。它们解决同一个问题（用模型信号收窄特征清单），但介入的阶段不同——同一产品通常只启用其中一条，避免叠加后难以归因：

| 机制 | 介入点 | 输入/输出 | 何时用 |
|---|---|---|---|
| **probing**（`analysis.probing.enabled` + `scripts/build_sparse_cache.py`） | Stage-1 打分内 | CSR 稀疏缓存 → `gain_rank_pct` 加权进 `rank_score` | 高维稀疏宽表（千维+），想让交互级信号影响 v1_auto 排名本身 |
| **null importance**（`analysis.null_importance.enabled` 或 `run_analysis.py --model-screen`） | Stage-1 之后（Stage-1.5） | `v1_auto.txt` → 目标置换筛选 → `v2_model.txt` | 想要一个显式的"模型体检"清单版本，人工 review 后再进 Stage-2（xc 线的默认路径） |
| **Stage-2 候选漏斗**（`training.stage2_candidate_count` + `stage2_pruning`） | Stage-2 训练内 | 宽候选池 → 探索 XGB（gain/stability/shap/permutation）→ `final_feature_count` | 不想维护多份清单文件，让训练流程自己收窄（hzz 线的默认路径） |

`analysis.stage1_top_n` 显式控制 `v1_auto.txt` 大小，且优先于 `stage2_candidate_count` 推导的候选池大小。

### hzz 原始数据预处理

hzz 线的原始多表数据先经 `scripts/preprocess_hzz_raw.py`（配置在 `configs/preprocess/hzz_raw.yaml`）合成建模宽表，再走标准 Stage-1/2；产物质检见 `notebooks/00_hzz_raw_preprocess_check.ipynb`。合成冒烟数据可用 `scripts/build_hzz_raw_synthetic.py` / `build_hzz_day_synthetic.py` 生成。

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

- **自动家族**：`txn_amt_7d / _30d / _90d / _all` 等由 `feature_groups.window_pattern` 自动识别，家族内最多保留 `max_per_family`（默认 2）个。可用 `feature_groups.enable_window_family: false` 关闭（特征是纯编号、无时间窗后缀时，如 `xc`）。
- **语义簇**：在 product YAML 的 `feature_groups.semantic_groups` 里手动声明（如"机构数/还款金额/申请数"），簇内最多保留 `max_keep` 个。

两种机制互相独立，均通过 `selector.py` 合并入综合打分。相关性（数值 `|r|`）去冗余聚类与名称无关，始终生效。

### 综合打分权重与正例偏向筛选

`selector.py` 的 `rank_score` 权重可在 `analysis.rank_weights` 配置（默认值等价旧公式 `z(iv)+z(lift)+z(gini)-z(psi)-0.5·1[missing>0.5]`）：

- `iv / lift / gini / concentration / psi` —— 各信号权重；调高 `lift`、`concentration` 即偏向"把正例排到前面"的特征。
- `missing_penalty` + `missing_penalty_threshold` —— 缺失惩罚的权重与触发阈值（按需压低，避免过度惩罚高缺失特征）。
- `analysis.lift_keep_min` —— 正例软门槛：`iv < iv_min` 但 `lift_at_k >= lift_keep_min` 的特征仍保留（默认 `null` 关闭）。

`xc` 即采用此组合：抬高 `lift`/`concentration`、压低 `missing_penalty`、开启 `lift_keep_min`，使筛选主要偏向正例且不过度惩罚缺失。

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
