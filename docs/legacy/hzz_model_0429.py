# -*- coding: utf-8 -*-
import sys
import numpy as np
import xgboost as xgb
import pickle
import importlib
import copy, json
import os

sys.path.append('.')
importlib.reload(sys)
from pyspark.sql import SparkSession
import pyspark.sql.functions as fuc
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType, StructField, StructType, DoubleType
from pyspark.sql import functions as F

# -----------------------------------------------------------------------------
# 【配置修改区】
# -----------------------------------------------------------------------------
KEEP_COLS = ['nine_mobl_md5', 'nine_mobl_sha2', 'id_cardae_md5','id_cardae_sha2','nine_mobl_aks']

# 全局变量占位
flist1_global = []
flist2_global = []


def dump_feats(row):
    res = []
    # 1. 处理 PIN
    try:
        pin = row['pin'] if 'pin' in row else row['user_pin']
        res.append(str(pin) if pin is not None else '')
    except:
        res.append('')

    # 2. 提取特征向量
    def get_feature_vector(target_flist):
        features = []
        for feat_name in target_flist:
            try:
                val = row[feat_name]
                features.append(float(val) if val is not None else np.nan)
            except:
                features.append(np.nan)
        return json.dumps(features)

    res.append(get_feature_vector(flist1_global))
    res.append(get_feature_vector(flist2_global))
 
    
    # 3. 提取保留字段
    for col_name in KEEP_COLS:
        try:
            val = row[col_name]
            res.append(str(val) if val is not None else "") # 统一转空字符串防止NoneType报错
        except:
            res.append("")
            
    return res

# --- 1. 显式定义 Input Schema (修复报错的核心) ---
base_input_cols = ['pin', 'feat1', 'feat2']
input_fields = [StructField(name, StringType(), True) for name in base_input_cols]
for col in KEEP_COLS:
    input_fields.append(StructField(col, StringType(), True))
udf_input_schema = StructType(input_fields)

# --- 2. Output Schema ---
# 【修改点】meta_output_names 缩减为 pred_score1, pred_score2
meta_output_names = ['pin', 'pred_score1', 'pred_score2'] + KEEP_COLS
feat_schema = StructType([StructField(e, StringType(), True) for e in meta_output_names])


# --- 在 UDF 外部（模块级别）定义一个字典，用于在 Executor 内存中缓存模型 ---
_GLOBAL_MODEL_CACHE = {}

@pandas_udf(feat_schema, PandasUDFType.GROUPED_MAP)
def xgb_predict(pdf):
    base_cols = ['pin'] + KEEP_COLS
    df_res = pdf[base_cols].copy(deep=True)
    
    # 两个模型的文件名
    model_files = [binary_xgb_file1_name, binary_xgb_file2_name]
    feat_cols = ['feat1', 'feat2']
    
    # 【新增修改】：为两个模型分别定义截断树的数量（与离线测试保持一致）
    # 索引 0 对应第一个模型 (254)，索引 1 对应第二个模型 (202)
    best_iters = [254, 202] 
    
    for i in range(2):
        col_name = 'pred_score{}'.format(i+1)
        feat_col = feat_cols[i]
        model_file = model_files[i]
        limit_iter = best_iters[i] # 动态获取当前模型的迭代次数
        
        try:
            # 1. 解析 JSON 特征
            raw_feats = pdf[feat_col]
            feas_list = [json.loads(e) for e in raw_feats]
            feas_array = np.array(feas_list)
            
            # 2. 【性能优化】：使用全局字典缓存模型，避免在 Executor 循环中被重复加载几千次
            if i not in _GLOBAL_MODEL_CACHE:
                bst = xgb.Booster()
                bst.load_model(model_file)
                _GLOBAL_MODEL_CACHE[i] = bst
            
            bst = _GLOBAL_MODEL_CACHE[i]
            
            # 3. 构建 DMatrix
            dtest = xgb.DMatrix(feas_array, missing=np.nan)
            
            # 4. 【关键修改】：传入对应的 iteration_range 截断预测
            preds = bst.predict(dtest, iteration_range=(0, limit_iter))
            
            df_res[col_name] = preds.astype(str)
            
        except Exception as e:
            print(f"Error predicting {col_name} with XGBoost JSON: {str(e)}")
            df_res[col_name] = "-1.0"

    return df_res[meta_output_names]




def load_flist(filepath):
    flist = []
    with open(filepath, 'r') as f:
        for l in f:
            l = l.strip('\n').strip()
            if l == '': continue
            flist.append(l)
    return flist

if __name__ == '__main__':
    print("Arguments:", sys.argv)
    input_date = sys.argv[1]
    
    # 【修改点】调整参数获取逻辑，只需 2 对路径（模型路径, 特征路径）
    m_path1, f_path1 = sys.argv[2], sys.argv[3]
    m_path2, f_path2 = sys.argv[4], sys.argv[5]
    num_partitions = int(sys.argv[6]) # 【修改点】由于删除了第3个模型参数，分区参数索引变更为 6

    app_name = os.path.basename(sys.argv[0])
    spark = SparkSession.builder.appName(app_name) \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    # 【修改点】只需分发 2 个文件
    sc.addPyFile(m_path1)
    sc.addPyFile(m_path2)

    global binary_xgb_file1_name, binary_xgb_file2_name
    binary_xgb_file1_name = os.path.basename(m_path1)
    binary_xgb_file2_name = os.path.basename(m_path2)

    # 【修改点】只需加载 2 个特征列表
    flist1_global = load_flist(f_path1)
    flist2_global = load_flist(f_path2)
    
    all_needed_features = list(set(flist1_global + flist2_global))
    
    table_date = ''.join(input_date.split('-'))
    spark.sql("""use dmb_tmp""")
    tmp_table_name = "dmb_tmp.jm_hzz_{table_date}_model_0429".format(table_date=table_date)
    
    # 优化 SQL：只 Select 需要的列，避免 Select * 带来的潜在列名冲突
    # 如果特征很多，且都在 a 表，用 a.* 也可以，但要确保 b 表没有重名列覆盖 a 表的特征
    create_tmp_sql = """
        create table if not exists {tmp_table_name} as
        select 
            a.fx_360_phone_nbank_else_orgcnt,
            a.fx_360_cust_bank_tra_count,
            a.jq_last_fail_apply_days_from_now,
            a.12m_min_nowgapdays,
            a.bt_biz_type_cnt,
            a.6m_nbank_topnet_allnum_ratio,
            a.bt_crdt_lmt,
            a.fx_360_cust_nbank_nsloan_count,
            a.fx_360_cust_nbank_nsloan_orgcnt,
            a.bt_credit_cnt,
            a.jq_first_fail_apply_days_from_now,
            a.3m_nbank_topnet_allnum_ratio,
            a.fx_360_phone_nbank_max_moncnt,
            a.12m_mean_prevdays,
            a.fx_360_cust_nbank_selfcnt,
            a.fx_360_cust_nbank_sloan_orgcnt,
            a.3m_mid_allnum_ratio,
            a.12m_allnum,
            a.3m_high_allnum_ratio,
            a.fx_360_phone_coon_orgcnt,
            a.bt_set_autopay_pay_mob,
            a.jq_last_cash_apply_days_from_now,
            a.fx_360_cust_nbank_ca_orgcnt,
            a.bt_credit_chnl_code_cnt,
            a.12m_nbank_topnet_allnum_ratio,
            a.12m_nbank_allnum_ratio,
            a.fx_360_cust_nbank_finlea_count,
            a.jt_curr_crdt_amt,
            a.12m_nbank_itfin_allnum_ratio,
            a.12m_nbank_other_allnum_ratio,
            a.fx_360_phone_coon_count,
            a.fx_360_cust_af_count,
            a.bt_aval_crdt_lmt,
            a.jt_his_max_ovrd_days,
            a.jt_credit_pnsh_rate,
            a.nbank_carfin_1m_3m_allnum_ratio,
            a.jq_last_suc_apply_days_from_now,
            a.nbank_topnet_1m_3m_allnum_ratio,
            a.jq_eduction_tag,
            a.nbank_topnet_7d_1m_allnum_ratio,
            a.fx_360_cust_nbank_finlea_orgcnt,
            a.12m_bank_country_allnum_ratio,
            a.fx_180_phone_nbank_week_orgcnt,
            a.fx_180_cust_nbank_sloan_orgcnt,
            a.fx_180_phone_coon_orgcnt,
            a.fx_360_cust_af_orgcnt,
            a.fx_180_cust_nbank_selfcnt,
            a.12m_low_orgnum,
            a.fx_90_phone_coon_orgcnt,
            a.bt_max_over_days_his,
            a.jq_loan_apply_amt_sum,
            a.12m_cv_prevdays,
            a.jq_cash_succ_rate_his,
            a.jq_eduction_tag_2_cnt_rate,
            a.fx_180_cust_bank_tra_count,
            a.6m_nbank_topnet_orgnum,
            a.jt_first_crdt_months,
            a.6m_nbank_topnet_daynum,
            a.3m_nbank_topnet_daynum,
            a.nbank_topnet_6m_12m_allnum_ratio,
            a.jt_accm_shd_day_fee_amt,
            a.bt_user_lvl_code,
            a.12m_nbank_carfin_allnum_ratio,
            a.3m_6m_allnum_ratio,
            a.fx_90_cust_nbank_nsloan_orgcnt,
            a.bt_repay_xjk_cnt_his,
            a.fx_360_phone_nbank_nsloan_orgcnt,
            a.12m_max_nowgapdays,
            a.3m_orgnum,
            a.jt_reg_paid_order_all_radio_his,
            a.12m_mid_allnum,
            a.bt_repay_jdapp_cnt_180,
            a.fx_90_cust_nbank_selfcnt,
            a.bt_last_paid_nor_days_his,
            a.6m_nbank_other_allnum_ratio,
            a.nbank_cons_3m_6m_allnum_ratio,
            a.12m_high_allnum_ratio,
            a.6m_nbank_cons_allnum_ratio,
            a.jt_actv_time_months,
            a.12m_nbank_cons_allnum_ratio,
            a.jq_loan_apply_fail_cnt_sum,
            a.jq_loan_aval_amt_sum,
            a.12m_bank_daynum,
            a.mid_1m_3m_allnum_ratio,
            a.jt_failed_ratio_24m,
            a.bt_actv_time_months,
            a.12m_orgnum,
            a.jt_last_ord_days_from_now,
            a.12m_nbank_topnet_allnum,
            a.3m_nbank_other_allnum_ratio,
            a.jt_first_apply_curr_month_diff,
            a.1m_nbank_topnet_orgnum,
            a.12m_nbank_orgnum,
            a.bt_loan_plan_cnt_sum_his,
            a.6m_mid_allnum_ratio,
            a.bank_6m_12m_allnum_ratio,
            a.3m_nbank_carfin_allnum_ratio,
            a.6m_12m_allnum_ratio,
            a.6m_cv_prevdays,
            a.jt_all_paid_prin_avg_his,
            a.nbank_topnet_3m_6m_allnum_ratio,
            a.12m_mid_allnum_ratio,
            a.jt_recvbl_stag_fee_std_his,
            a.6m_nbank_topnet_night_allnum,
            a.6m_nbank_carfin_allnum_ratio,
            a.fx_360_phone_caoff_orgcnt,
            a.jt_loan_prin_amt_sum_his,
            a.6m_max_prevdays,
            a.bt_yhk_pay_cnt_all,
            a.12m_nbank_carfin_orgnum,
            a.6m_high_allnum_ratio,
            a.nbank_6m_12m_allnum_ratio,
            a.jt_curr_first_crdt_amt_per,
            a.3m_max_prevdays,
            a.12m_mid_orgnum,
            a.bt_last_order_plan1_datediff,
            a.jt_unpayoff_prin_std_his,
            a.3m_mid_orgnum,
            a.3m_bank_country_allnum_ratio,
            a.3m_cv_prevdays,
            a.bt_tot_sc_order_cnt,
            a.6m_bank_country_orgnum,
            a.3m_mid_daynum,
            a.nbank_carfin_3m_6m_allnum_ratio,
            a.cf_quota_quality_180d,
            a.jt_loan_prin_ord_cnt_his,
            a.nbank_other_1m_3m_allnum_ratio,
            a.12m_stddev_prevdays,
            a.3m_bank_city_allnum_ratio,
            a.jq_apply_cnt_radio_180_his,
            a.3m_nbank_topnet_orgnum,
            a.jq_apply_prod_cnt_180,
            a.3m_stddev_prevdays,
            a.jq_last_apply_fromnow_months,
            a.nbank_other_3m_6m_allnum_ratio,
            a.12m_bank_city_allnum_ratio,
            a.jq_cash_succ_rate_12m,
            a.mid_6m_12m_allnum_ratio,
            a.jt_recvbl_stag_fee_avg_his,
            a.12m_bank_city_night_allnum,
            a.jt_apply_reject_radio_60_his,
            a.bq_credit_amt_last,
            a.jq_first_suc_apply_days_from_now,
            a.3m_bank_allnum_ratio,
            a.bt_pre_repay_cnt_radio_365,
            a.12m_mid_daynum,
            a.6m_nbank_carfin_orgnum,
            a.jt_last_apply_first_month_diff,
            a.fx_180_phone_nbank_ca_orgcnt,
            a.bt_repay_pre_paid_amt_radio_365,
            a.jq_loan_aval_amt_cashfail_avg,
            a.6m_nbank_orgnum,
            a.jq_loan_apply_amt_max,
            a.bt_net_loan_plan_amt_sum_365,
            a.fx_180_phone_nbank_nsloan_orgcnt,
            a.bt_aval_crdt_rate,
            a.1m_nbank_cons_orgnum,
            a.12m_bank_allnum_ratio,
            a.3m_orgtypenum,
            a.6m_stddev_prevdays,
            a.bt_first_sc_order_datediff,
            a.6m_bank_city_allnum_ratio,
            a.jq_cash_interval_max_his,
            a.nbank_cons_6m_12m_allnum_ratio,
            a.bt_net_loan_plan_amt_radio_180,
            a.jt_curr_crdt_amt_radio,
            a.jq_apply_cnt_radio_365_his,
            a.bt_repay_card_amt_his,
            a.jt_loan_prin_amt_max_his,
            a.bt_last_paid_days_his,
            a.6m_orgnum,
            a.bank_3m_6m_allnum_ratio,
            a.3m_nbank_allnum_ratio,
            a.6m_mid_daynum,
            a.fx_180_cust_nbank_finlea_orgcnt,
            a.fail_apply_amt_his,
            a.12m_nbank_carfin_daynum,
            a.fx_90_phone_nbank_sloan_orgcnt,
            a.bank_country_6m_12m_allnum_ratio,
            a.bt_repay_plan_cnt_180,
            a.3m_nbank_topnet_weekend_allnum,
            a.bt_net_loan_plan1_amt_avg_365,
            a.6m_nbank_topnet_allnum,
            a.nbank_topnet_3d_7d_allnum_ratio,
            a.12m_nbank_topnet_daynum,
            a.bt_plan_cnt_avg_365,
            a.jt_shd_repay_amt_min_his,
            a.jt_reg_paid_order_radio_1m_his,
            a.6m_nbank_allnum_ratio,
            a.12m_nbank_carfin_allnum,
            a.fx_90_phone_nbank_finlea_orgcnt,
            a.bt_repay_month_his,
            a.bt_pre_paid_amt_sum_270,
            a.bt_loan_amt_max_270,
            a.bt_fee_bill_dtl_amt_his,
            a.bt_pre_repay_cnt_radio_his,
            a.jq_loan_prd_amt_mth_avg,
            a.6m_bank_city_weekend_allnum,
            a.jt_nonsetl_loan_radio_his,
            a.6m_nbank_carfin_daynum,
            a.fx_180_phone_nbank_else_orgcnt,
            a.fx_180_cust_nbank_nsloan_count,
            a.fx_180_phone_nbank_max_moncnt,
            a.6m_mean_prevdays,
            a.fx_180_cust_nbank_ca_orgcnt,
            a.fx_180_cust_nbank_finlea_count,
            a.fx_180_cust_af_count,
            a.fx_15_phone_nbank_oth_count,
            a.fx_15_cust_coon_count,
            a.6m_low_orgnum,
            a.6m_allnum,
            a.jq_apply_tot_cnt_180,
            a.3m_nbank_topnet_allnum,
            a.fx_180_cust_af_orgcnt,
            a.fx_180_phone_coon_count,
            a.fx_15_phone_nbank_cons_orgcnt,
            a.jq_loan_apply_amt_avg,
            a.jq_cash_interval_std_his,
            a.fx_180_phone_caoff_orgcnt,
            a.fx_15_cust_nbank_count,
            a.jq_cash_interval_avg_his,
            a.bt_loan_plan_amt_sum_his,
            a.bt_repay_pre_plan_cnt_radio_270,
            a.6m_bank_country_allnum_ratio,
            a.6m_mid_allnum,
            a.fx_15_cust_nbank_nsloan_count,
            a.jt_curr_crdt_total_amt_radio,
            a.bt_repy_time_days_max_270,
            a.6m_mid_orgnum,
            a.jt_loan_prin_sum_radio_60_365,
            a.bt_loan_plan3_amt_sum_his,
            a.nbank_carfin_6m_12m_allnum_ratio,
            a.6m_bank_city_night_allnum,
            a.jt_last_apply_succ_last_month_diff,
            a.3m_nbank_other_orgnum,
            a.bt_net_loan_cnt_radio_30_270,
            a.6m_orgtypenum,
            a.bt_bill_dtl_prin_sum_180,
            a.6m_nbank_carfin_allnum,
            a.bt_fee_bill_dtl_amt_730,
            a.bt_loan_plan1_radio_his,
            a.nbank_cons_1m_3m_allnum_ratio,
            a.3m_nbank_carfin_allnum,
            a.jt_loan_prin_sum_radio_180_365,
            a.6m_bank_daynum,
            a.jq_apply_org_cnt_his_12m_rate,
            a.jt_loan_prin_sum_radio_365_wkd,
            a.jq_loan_reate_cv,
            a.bt_nor_paid_cnt_radio_his,
            a.bt_bill_dt,
            a.nbank_other_6m_12m_allnum_ratio,
            a.mid_3m_6m_allnum_ratio,
            a.bt_repy_time_days_max_180,
            a.fx_15_cust_caon_orgcnt,
            a.bt_shd_pay_prin_sc_avg_90,
            a.bt_net_loan_plan1_amt_avg_180,
            a.jq_loan_apply_amt_cv,
            a.bt_repay_pre_paid_amt_radio_180,
            a.bt_loan_amt_radio_90_270,
            a.6m_bank_city_daynum,
            a.jt_all_paid_prin_over_sum_his,
            a.acct_no as pin,
            b.nine_mobl_md5,
            b.nine_mobl_sha2,
            b.id_cardae_md5,
            b.id_cardae_sha2,
            b.nine_mobl_aks
        from
            (select * from dmb_rpt.dmb_jdt_dmbrpt_hzz_model_feats_0430_s_det_d where dt='{observe_date}') a
        LEFT JOIN 
            (select * from idm.idm_c01_per_auth_prm_s_d where dt='{observe_date}') b
        on a.acct_no = b.user_pin
    """.format(tmp_table_name=tmp_table_name, observe_date=input_date)
    
    # 先 Drop 再 Create，确保数据是最新的
    spark.sql("drop table if exists {}".format(tmp_table_name))
    spark.sql(create_tmp_sql)
    
    df_all = spark.sql("select * from {}".format(tmp_table_name)).repartition(num_partitions)
    
    # 2. 筛选列
    cols_to_select = ['pin'] + all_needed_features + KEEP_COLS
    # 去重
    cols_to_select = list(set(cols_to_select))
    
    # 确保列存在，防止 select 报错 (可选优化)
    # available_cols = df_all.columns
    # cols_to_select = [c for c in cols_to_select if c in available_cols]

    df1 = df_all.select(cols_to_select)
    
    # 3. RDD Map
    encode_rdd = df1.rdd.map(lambda x: dump_feats(x))
    
    # 4. 创建 DataFrame (修复点：使用 StructType)
    dfres = spark.createDataFrame(encode_rdd, schema=udf_input_schema)
    
    # 增加随机分组列
    dfres = dfres.select(*list(dfres.columns), (fuc.ceil(fuc.rand() * num_partitions)).alias('grouper'))
    
    # 5. 预测
    df_prob = dfres.groupby('grouper').apply(xgb_predict)
    
    # 6. 注册临时表
    df_prob.createOrReplaceTempView('res_tmp_hzz_output_{table_date}'.format(table_date=table_date))

    # 7. 动态构建最终 SQL
    extra_cols_sql_part = ""
    for col in KEEP_COLS:
        extra_cols_sql_part += ", max({}) as {}".format(col, col)

    final_sql = """
        insert overwrite table dmb_rpt.dmb_jdt_dmbrpt_hzz_model_result_s_det_d partition (dt='{observe_date}')
        select 
            pin,
            max(cast(pred_score1 as double)) as pred_score1,
            max(cast(pred_score2 as double)) as pred_score2
            {extra_cols_sql}
        from res_tmp_hzz_output_{table_date}
        group by pin
    """.format(observe_date=input_date, extra_cols_sql=extra_cols_sql_part,table_date=table_date)

    print("Executing SQL:\n", final_sql)
    spark.sql(final_sql)
    print("Job Finished.")