# xc 数据生产环境执行步骤

> 本文档**只针对 xc 数据**(`data/xc_features.csv` + `data/xc_labels.csv` 及其衍生的
> 5 个 xc 模型),不适用于 bank_marketing / home_credit 等其他产品。

建模漏斗:两段式(响应 + 资质,资质分 V1/V2 两个口径);分析漏斗:**校准**融合分
(校准响应 × 校准资质,可选 α 加权)按全流程 `is_reg → is_finish_task → 授信` 评估
(`is_reg` 仅用于分析,不建模);另有端到端单模型基线作对照。

涉及的 5 个模型:

| product | 人群 | 标签 | 角色 |
|---|---|---|---|
| `xc_resp_finish` | 全量 | is_finish_task | 响应 |
| `xc_qual_finish` | is_finish_task==1 | is_credit_succ(1 正 / 0 负) | 资质 V1 |
| `xc_qual_finish_1v1` | is_finish_task==1 | is_credit_1v1(credit_1v1 ∈ {1,2,3} 正;0、-1 负);档位价值加权 790/290/120 | 资质 V2 |
| `xc_e2e_credit` | 全量 | is_credit_succ | 端到端基线 V1(只做对照,默认不上线) |
| `xc_e2e_credit_1v1` | 全量 | is_credit_1v1;档位价值加权 | 端到端基线 V2(只做对照,默认不上线) |

xc 各产品调参目标为 `precision_at_k`(CV P@10%)、CV 用前向扩窗时间折
(`cv_strategy: time_forward`)。**注意**:旧 run 的 `trials.pkl` 是按 PR-AUC 目标
搜的,与新目标不可比 —— 换目标后必须用新 run-id(管线会自动拒绝混用)。

## 0. 前置条件

**环境**(训练机):
- 训练/打分必须用带 xgboost 的 Python 环境(conda env `env36`,
  即 `conda run -n env36 python`,下文记作 `$PY`);
  依赖清单见 `requirements.txt`,一步建环境见 `environment.yaml`。
- 所有命令在仓库根目录执行,带 `PYTHONPATH=src`。

**输入数据**(两份 CSV,键为 `id` + 日期,合表时统一成 `dt`(yyyymmdd)):
- `data/xc_features.csv`:`id, dt(yyyy-mm-dd 或 yyyymmdd), feat1..featN`(原始业务特征,允许缺失/哨兵值;
  历史上时间列名为 `apply_time` 的旧文件用 `--feat-time-col apply_time` 兼容)
- `data/xc_labels.csv`:`id, dt(yyyymmdd), is_reg, is_finish_task, is_credit_succ, credit_1v1`
- 标签观察期必须已成熟(授信结果已回流),否则 `is_credit_succ` / `credit_1v1` 会系统性偏低。

## 1. 合表(建模表 + 分析漏斗基线)

```bash
PYTHONPATH=src $PY scripts/build_xc_dataset.py \
  --features data/xc_features.csv --labels data/xc_labels.csv --out-dir data
```

产出:`data/xc_full.csv`(全量,服务响应模型)、`data/xc_qual_finish.csv`
(is_finish_task==1,同时服务资质 V1 和 V2)。合表时从 `credit_1v1` 衍生二值标签
`is_credit_1v1`(1/2/3 → 1;0、-1 → 0;其余值告警并按 0 处理),原始 `credit_1v1`
列保留仅供审计。

**检查点**:终端打印的 "Analysis funnels" 两个口径(endpoint = is_credit_succ /
is_credit_1v1)的逐段转化率(reg、finish|reg、credit|finish)须与业务大盘一致
(量级偏差 >20% 即停,排查 join 匹配率与标签口径);join 行数损失、重复键告警、
credit_1v1 异常值告警须确认无异常。

## 2. Stage-1 特征分析(5 个 xc 模型各一次)

```bash
for p in xc_resp_finish xc_qual_finish xc_qual_finish_1v1 xc_e2e_credit xc_e2e_credit_1v1; do
  PYTHONPATH=src $PY scripts/run_analysis.py --product $p
done
```

**检查点**:逐个打开 `artifacts/<product>/analysis/report/index.html`,
确认入选特征无泄漏(任何任务/授信行为**之后**才产生的字段必须剔除)。

> 磁盘提示:Stage-1 单遍扫描会在 `artifacts/<product>/analysis/scan_cache/`
> 下写临时 .npy 块(约 行数×特征数×8 字节),跑完自动清理。磁盘紧张时在
> product yaml 的 `io_overrides.scan_cache` 里把 `dir` 指向 scratch 盘,
> 或 `enabled: false` 回退到慢速路径(不占磁盘)。

## 2.5 模型法特征筛选(Stage-1.5,xc 已启用)

对 Stage-1 的 `v1_auto` 列表再做一层 null importance(标签置换)筛选:多次
shuffle 标签训练小型 XGBoost 得到各特征 gain 的"空分布",真实 gain 未显著
超过空分布的特征剔除,产出 `selected_features/v2_model.txt`。xc 各产品已在
配置中启用(`analysis_overrides.null_importance.enabled: true`),第 2 步
`run_analysis.py` 跑完 Stage-1 后**自动执行**(因此第 2 步必须用带 xgboost
的 `$PY`),无需单独操作。筛选漏斗:原始 ~1000 维 → Stage-1 统计粗筛 ≤350
(`analysis.stage1_top_n`,v1_auto)→ 本步模型精筛 ≤200
(`training.final_feature_count`,v2_model)。

单独补跑/重跑本步(不重跑 Stage-1)时:

```bash
for p in xc_resp_finish xc_qual_finish xc_qual_finish_1v1 xc_e2e_credit xc_e2e_credit_1v1; do
  PYTHONPATH=src $PY scripts/run_model_screen.py --product $p
done
```

**检查点**:`report/null_importance.csv` 中被剔除特征的 `gain_actual` 应明显
低于其 `null_keep_ref`;`analysis/null_importance_meta.json` 的 `n_written`
应接近 200(明显偏少说明大量候选未过显著性,复查 Stage-1 列表质量)。
其余参数见 global.yaml `analysis.null_importance`。

## 3. 人工特征评审(强烈建议)

以 Stage-1.5 的筛后列表为起点:

```bash
for p in xc_resp_finish xc_qual_finish xc_qual_finish_1v1 xc_e2e_credit xc_e2e_credit_1v1; do
  cp artifacts/$p/selected_features/v2_model.txt \
     artifacts/$p/selected_features/v3_manual_noleak.txt
done
# 逐个编辑 v3_manual_noleak.txt:删除泄漏/不可上线特征,加备注
```

## 4. Stage-2 训练(5 个 xc 模型各一次)

run-id 建议 `prod_<yyyymmdd>`。生产训练不要传 `--max-evals`(用配置默认 30)。

```bash
for p in xc_resp_finish xc_qual_finish xc_qual_finish_1v1 xc_e2e_credit xc_e2e_credit_1v1; do
  PYTHONPATH=src $PY scripts/run_training.py \
    --product $p --run-id prod_20260610 --features-version v3_manual_noleak
done
```

**检查点**:各 `artifacts/<xc 产品>/models/prod_20260610/metrics.md` 中 OOT 与 Valid
指标不应大幅劣于 Train(KS/PR-AUC 腰斩即怀疑过拟合或时间漂移);
`importance.csv` 头部特征业务可解释;`calibration.json` 存在且 `x` 单调递增、
`y ∈ [0,1]`(valid 样本/正例过少会跳过校准并告警 —— 此时该 bundle 不能参与
校准融合,需排查 valid 窗口);`run_manifest.json` 含 `split_boundaries`、
`tuner_objective`、`cv_strategy`(1v1 产品另含 `sample_weight`)。

## 5. 部署一致性自测(5 个 bundle 必跑)

```bash
for p in xc_resp_finish xc_qual_finish xc_qual_finish_1v1 xc_e2e_credit xc_e2e_credit_1v1; do
  $PY artifacts/$p/models/prod_20260610/predict.py --validate
done
```

全部 `OK (raw)` + `OK (calibrated)`(1e-6 容差,双路检查)方可继续。

## 6. 融合漏斗分析(上线前验收,资质 V1/V2 各一次)

```bash
# 资质 V1:finish → is_credit_succ(带端到端基线对照)
PYTHONPATH=src $PY scripts/run_funnel_eval.py \
  --resp-product xc_resp_finish --resp-run-id prod_20260610 \
  --qual-product xc_qual_finish --qual-run-id prod_20260610 \
  --e2e-product xc_e2e_credit --e2e-run-id prod_20260610

# 资质 V2:finish → is_credit_1v1(--qual-stage 必须同步切换;
# --tier-values 输出 top-K 人均业务价值提升行 value_capture)
PYTHONPATH=src $PY scripts/run_funnel_eval.py \
  --resp-product xc_resp_finish --resp-run-id prod_20260610 \
  --qual-product xc_qual_finish_1v1 --qual-run-id prod_20260610 \
  --qual-stage is_credit_1v1 \
  --e2e-product xc_e2e_credit_1v1 --e2e-run-id prod_20260610 \
  --tier-values "1:120,2:290,3:790"
```

融合用各 bundle 的**校准分**(calibration.json 缺失会回退原始分并大声告警 ——
生产验收不允许回退);`fused_alpha` 的 α 在 OOT 之前的拟合窗(各模型 valid 期
重叠段)自动网格搜索,最大化授信 lift@K(K 默认 0.10,`--alpha-k` 可调,应与
实际投放比例一致)。默认仅在所有模型都未见过的 OOT 窗口评估
(边界读各 bundle 的 `run_manifest.json`,可 `--start-dt/--end-dt` 覆盖)。
产出(每份目录里 `funnel_eval.{csv,md}` + `fusion_spec.json`):
- `artifacts/funnel_eval/xc_resp_finish-prod_20260610__xc_qual_finish-prod_20260610/`
- `artifacts/funnel_eval/xc_resp_finish-prod_20260610__xc_qual_finish_1v1-prod_20260610/`

**验收标准**(看 absolute 口径、目标投放比例那一档,如 top 10%;
授信行为 `is_credit_succ`(V1)或 `is_credit_1v1`(V2)):
- `fused_alpha` 的授信行(端到端综合提升)≥ `fused`(plain 乘积)≥
  `resp_only` 和 `qual_only` —— 校准 + α 融合有增益;
- `fused_alpha` ≥ `e2e_only` —— 两段式确实优于端到端单模型;若 `e2e_only`
  反超,优先排查两段式(资质外推、α 拟合窗样本量),并评估改用单模型上线;
- 综合提升达到业务起投门槛(如 ≥2×,按预算与单客成本定);
- conditional 口径确认提升来源符合预期(响应段、资质段都应 >1);
- `fusion_spec.json`:`alpha_source` 应为 `grid_fit`(`default_fallback` 说明
  拟合窗样本不足,α 退回 0.5,需人工确认);`fit_window` 与 `eval_window`
  不重叠;V2 另看 `value_capture` 行的人均价值提升。

**资质口径选型**:对比 V1/V2 两份报告的综合提升与业务目标
(V1 = 授信成功口径,V2 = credit_1v1 1/2/3 档口径 + 价值加权),
取贴合投放目标且提升更高者上线。

需要更细的探索(连续 top-K 提升曲线、漏斗对比图、资质 V1/V2 汇总)时,
用 `notebooks/10_funnel_eval_xc.ipynb`(内核选带 xgboost 的 env36 环境)。

## 7. 上线部署与打分(以选定资质 V1 为例)

把两个 bundle 目录 + 验收产出的 `fusion_spec.json` 拷贝到生产机
(只需 xgboost/numpy/pandas):

```bash
scp -r artifacts/xc_resp_finish/models/prod_20260610  prod:/srv/xc/resp_bundle
scp -r artifacts/xc_qual_finish/models/prod_20260610  prod:/srv/xc/qual_bundle
scp artifacts/funnel_eval/xc_resp_finish-prod_20260610__xc_qual_finish-prod_20260610/fusion_spec.json \
    prod:/srv/xc/fusion_spec.json
# 若选资质 V2,qual_bundle 换成 artifacts/xc_qual_finish_1v1/models/prod_20260610,
# fusion_spec.json 换成对应 V2 验收目录里的那份
```

生产机上:

```bash
# 上线前每台机器先自测(应输出 OK (raw) + OK (calibrated) 两行)
python /srv/xc/resp_bundle/predict.py --validate
python /srv/xc/qual_bundle/predict.py --validate

# 对当日待投放名单打分(raw.csv = 原始 xc 特征,列同训练特征,允许缺失)
python /srv/xc/resp_bundle/predict.py predict --input raw.csv --output resp_scores.csv
python /srv/xc/qual_bundle/predict.py predict --input raw.csv --output qual_scores.csv

# 融合分:按 fusion_spec.json 的 serving_formula 执行 ——
# fused = score_calibrated_resp^alpha * score_calibrated_qual^(1-alpha)
# (必须用 score_calibrated 列;score 列是未校准原始分,只供单模型场景)
python - <<'EOF'
import json
import pandas as pd

spec = json.load(open("/srv/xc/fusion_spec.json"))
alpha = float(spec["alpha"])
r = pd.read_csv("resp_scores.csv")[["row_index", "score_calibrated"]] \
      .rename(columns={"score_calibrated": "resp"})
q = pd.read_csv("qual_scores.csv")[["row_index", "score_calibrated"]] \
      .rename(columns={"score_calibrated": "qual"})
m = r.merge(q, on="row_index")
m["fused"] = m["resp"].clip(lower=1e-12) ** alpha * m["qual"].clip(lower=1e-12) ** (1 - alpha)
m.sort_values("fused", ascending=False).to_csv("fused_scores.csv", index=False)
print("alpha =", alpha, "| top row:", m["fused"].max())
EOF
```

## 8. 上线后监控与复盘

- 投放后标签成熟(授信结果回流)即用新的 xc_features/xc_labels 重跑第 1 步合表 +
  第 6 步漏斗分析,加 `--start-dt <上线日>` 限定窗口,对比线上实际综合提升 vs 验收值
  (注意资质 V2 要带 `--qual-stage is_credit_1v1`;**复盘必须带
  `--alpha <线上 fusion_spec.json 里的 alpha>`**,评估的才是实际投放的那套排序,
  而不是在新窗口重拟合的 α);
- 监控融合分分布漂移(对照两个 bundle 内 `validation_samples.csv` 的分数区间)与
  reg/finish/credit 各段转化率;漂移明显或提升衰减 → 回到第 1 步用新数据重训
  (新 run-id,旧 bundle 保留可回滚);
- 交互式监控工具:`notebooks/11_xc_monitoring.ipynb`(内核 env36)——
  新批次 vs 训练参照窗的分数 PSI、头部特征 PSI(缺失率跳变同样计入)、
  标签成熟后按线上 α 固定排序的上线窗口转化复盘(对照验收报告);
- xc 特征上游口径变更,必须重新走第 2~6 步全流程。
