import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import joblib
import logging
import datetime
import sys
import warnings
import re

# 环境变量设置
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# =================== 1. 配置 =================== #

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

CONFIG = {
    'input_path': r'D:\8PVwithf\tongguanxian\统计结果\hexin\alldata.csv',
    'output_dir': 'pseudo_labeling_results',

    'coarse_candidates': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    'fine_search_step': 0.01,
    'fine_search_window': 2,

    'self_training_conf_threshold': 0.90,
    'max_iter': 5,
    'max_new_pos_labels': 100,
    'max_new_neg_labels': 200,

    'n_splits': 5,
    'val_a_size': 0.15,
    'val_b_size': 0.15,
    'random_state': 42,

    'lgb_params': {
        'objective': 'binary', 'class_weight': 'balanced', 'n_estimators': 1000,
        'learning_rate': 0.03, 'num_leaves': 20, 'max_depth': 7, 'min_child_samples': 20,
        'n_jobs': -1, 'verbosity': -1, 'max_bin': 63, 'subsample': 0.8, 'colsample_bytree': 0.8
    },
    'xgb_params': {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 1000,
        'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'n_jobs': -1, 'verbosity': 0, 'scale_pos_weight': 1, 'early_stopping_rounds': 50
    },
    'early_stopping_rounds': 50,

    'id_cols': ['ID', 'type', 'label', 'cluster'],
    # [关键] pure_types 依然用于计算相似度，但后面我们会过滤
    'pure_types': ['gd', 'btsm', 'ld', 'cgf'],
    'mixed_types': ['hhgf'],
    'label_map': {'gd': 'AGRI', 'btsm': 'IMP', 'ld': 'BARE', 'cgf': 'PURE_PV'}
}

MODELS_DIR = os.path.join(CONFIG['output_dir'], 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
FINAL_OUTPUT_PREFIX = os.path.join(CONFIG['output_dir'], f'final_result_{TIMESTAMP}_')


# =================== 2. 基础函数 (V13版: 统计特征) =================== #

def load_and_preprocess_data(config):
    path = config['input_path']
    if not os.path.exists(path):
        logger.error(f"文件不存在: {path}")
        sys.exit(1)
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='gbk')

    feature_candidates = [c for c in df.columns if c not in config['id_cols']]
    embed_pattern = re.compile(r'^A\d+')
    embed_cols = [c for c in feature_candidates if embed_pattern.match(c)]
    phys_cols_raw = [c for c in feature_candidates if c not in embed_cols]

    imputer = SimpleImputer(strategy='mean')
    df.loc[:, embed_cols + phys_cols_raw] = imputer.fit_transform(df[embed_cols + phys_cols_raw])

    # 统计特征增强 (V13 logic)
    df['phys_mean'] = df[phys_cols_raw].mean(axis=1)
    df['phys_std'] = df[phys_cols_raw].std(axis=1)
    df['phys_range'] = df[phys_cols_raw].max(axis=1) - df[phys_cols_raw].min(axis=1)

    phys_cols = phys_cols_raw + ['phys_mean', 'phys_std', 'phys_range']
    return df, embed_cols, phys_cols


def compute_similarity_matrix(df, config, phys_cols):
    # 保持 V12 的 IsolationForest + Median 清洗逻辑
    pure_mask = df['type'].isin(config['pure_types'])
    mixed_mask = df['type'].isin(config['mixed_types'])
    pure_df = df.loc[pure_mask].copy()
    mixed_df = df.loc[mixed_mask].copy()

    pure_df.loc[:, 'semantic_label'] = pure_df['type'].map(config['label_map'])
    pure_df = pure_df.dropna(subset=['semantic_label'])

    clean_pure_samples = []
    for label in pure_df['semantic_label'].unique():
        subset = pure_df[pure_df['semantic_label'] == label]
        if len(subset) > 20:
            clf = IsolationForest(contamination=0.1, random_state=config['random_state'], n_jobs=-1)
            preds = clf.fit_predict(subset[phys_cols])
            clean_subset = subset[preds == 1]
        else:
            clean_subset = subset
        clean_pure_samples.append(clean_subset)

    pure_df_clean = pd.concat(clean_pure_samples)

    # [关键修正] 在这里把 PURE_PV 从 pure_df_clean 中剔除，防止它进入负样本池
    # 但是！计算相似度时我们依然需要 PURE_PV 的原型吗？
    # 答：V14证明显式距离特征不好。V13不使用显式特征。
    # 为了保险，我们只用 AGRI 原型来算相似度？不，还是保留所有原型算 sim_df，
    # 但在 main 函数里构造 pure_negatives 时剔除 PURE_PV。

    prototypes = pure_df_clean.groupby('semantic_label')[phys_cols].median()

    scaler = StandardScaler()
    scaler.fit(pure_df_clean[phys_cols])

    mixed_scaled = scaler.transform(mixed_df[phys_cols])
    proto_scaled = scaler.transform(prototypes)

    sim_matrix = cosine_similarity(mixed_scaled, proto_scaled)
    sim_df = pd.DataFrame(sim_matrix, index=mixed_df.index, columns=prototypes.index)

    return mixed_df, pure_df_clean, sim_df


def get_initial_labels(mixed_df, sim_df, threshold):
    df_out = mixed_df.copy()

    def _get_label(row):
        cls = row.idxmax()
        score = row.max()
        if score < threshold: return 'UNCERTAIN_MIXED'
        return 'AGRI_PV' if cls == 'AGRI' else 'NOT_AGRI_PV'

    df_out.loc[:, 'label'] = sim_df.apply(_get_label, axis=1)
    df_out.loc[:, 'is_initial'] = True
    return df_out


# =================== 3. 核心训练逻辑 (回归 V13 架构) =================== #

def train_stacking_cv_return_models(X_train, y_train, embed_cols, phys_cols, config):
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['random_state'])
    fold_models = []

    sc_e = StandardScaler().fit(X_train[embed_cols])
    sc_s = StandardScaler().fit(X_train[phys_cols])

    X_e_sc = pd.DataFrame(sc_e.transform(X_train[embed_cols]), columns=embed_cols, index=X_train.index)
    X_s_sc = pd.DataFrame(sc_s.transform(X_train[phys_cols]), columns=phys_cols, index=X_train.index)

    oof_preds = np.zeros((len(y_train), 2))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_e_tr, X_e_val = X_e_sc.iloc[tr_idx], X_e_sc.iloc[val_idx]
        X_s_tr, X_s_val = X_s_sc.iloc[tr_idx], X_s_sc.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        # Base 1: LightGBM
        clf_e = lgb.LGBMClassifier(**config['lgb_params'])
        cbs = [lgb.early_stopping(stopping_rounds=config['early_stopping_rounds'], verbose=False)]
        clf_e.fit(X_e_tr, y_tr, eval_set=[(X_e_val, y_val)], eval_metric='auc', callbacks=cbs)

        # Base 2: XGBoost (兼容性修正版)
        if HAS_XGB:
            pos_ratio = (len(y_tr) - sum(y_tr)) / sum(y_tr)
            xgb_params = config['xgb_params'].copy()
            xgb_params['scale_pos_weight'] = pos_ratio
            xgb_params['early_stopping_rounds'] = config['early_stopping_rounds']
            clf_s = xgb.XGBClassifier(**xgb_params)
            clf_s.fit(X_s_tr, y_tr, eval_set=[(X_s_val, y_val)], verbose=False)
        else:
            clf_s = lgb.LGBMClassifier(**config['lgb_params'])
            clf_s.fit(X_s_tr, y_tr, eval_set=[(X_s_val, y_val)], eval_metric='auc', callbacks=cbs)

        oof_preds[val_idx, 0] = clf_e.predict_proba(X_e_val)[:, 1]
        oof_preds[val_idx, 1] = clf_s.predict_proba(X_s_val)[:, 1]
        fold_models.append({'clf_e': clf_e, 'clf_s': clf_s})

    X_meta = pd.DataFrame(oof_preds, columns=['e', 's'], index=X_train.index)
    meta = LogisticRegression(class_weight='balanced', random_state=config['random_state'], max_iter=2000)
    meta.fit(X_meta, y_train)

    return {'meta': meta, 'folds': fold_models, 'sc_e': sc_e, 'sc_s': sc_s}


def predict_ensemble(models_dict, X, embed_cols, phys_cols):
    sc_e = models_dict['sc_e']
    sc_s = models_dict['sc_s']
    X_e_sc = pd.DataFrame(sc_e.transform(X[embed_cols]), columns=embed_cols, index=X.index)
    X_s_sc = pd.DataFrame(sc_s.transform(X[phys_cols]), columns=phys_cols, index=X.index)

    n_folds = len(models_dict['folds'])
    pred_e_sum = np.zeros(len(X))
    pred_s_sum = np.zeros(len(X))

    for fold_m in models_dict['folds']:
        pred_e_sum += fold_m['clf_e'].predict_proba(X_e_sc)[:, 1]
        pred_s_sum += fold_m['clf_s'].predict_proba(X_s_sc)[:, 1]

    avg_e = pred_e_sum / n_folds
    avg_s = pred_s_sum / n_folds
    X_meta = pd.DataFrame({'e': avg_e, 's': avg_s}, index=X.index)
    return models_dict['meta'].predict_proba(X_meta)[:, 1]


# =================== 4. 实验主循环 =================== #

def run_experiment(threshold, mixed_df_raw, pure_negatives, sim_df, embed_cols, phys_cols, config, mode='coarse'):
    max_iter_actual = 2 if mode == 'coarse' else config['max_iter']
    if mode == 'coarse': logger.info(f" > [Coarse] Thresh {threshold} ...")

    current_df = get_initial_labels(mixed_df_raw, sim_df, threshold)
    labeled_df = current_df[current_df['label'] != 'UNCERTAIN_MIXED']

    if len(labeled_df[labeled_df['label'] == 'AGRI_PV']) < 10: return None

    try:
        X_train_init, X_val_combined, y_train_init, y_val_combined = train_test_split(
            labeled_df,
            (labeled_df['label'] == 'AGRI_PV').astype(int),
            test_size=(config['val_a_size'] + config['val_b_size']),
            stratify=(labeled_df['label'] == 'AGRI_PV').astype(int),
            random_state=config['random_state']
        )
        val_a_ratio = config['val_a_size'] / (config['val_a_size'] + config['val_b_size'])
        X_val_a, X_val_b, y_val_a, y_val_b = train_test_split(
            X_val_combined, y_val_combined, test_size=(1 - val_a_ratio), stratify=y_val_combined,
            random_state=config['random_state']
        )
    except ValueError:
        return None

    val_b_pos_count = sum(y_val_b)

    X_pure = pure_negatives
    y_pure = pd.Series([0] * len(X_pure), index=X_pure.index)
    X_train_curr = pd.concat([X_train_init, X_pure])
    y_train_curr = pd.concat([y_train_init, y_pure])
    unlabeled_pool = current_df[~current_df.index.isin(labeled_df.index)].copy()

    best_iter_f1 = -1
    best_models_data = None
    final_score = 0.0

    for iter_i in range(max_iter_actual):
        if sum(y_train_curr) < 10: break

        try:
            models_dict = train_stacking_cv_return_models(X_train_curr, y_train_curr, embed_cols, phys_cols, config)
        except Exception as e:
            logger.error(f"Error: {e}")
            break

        val_a_probs = predict_ensemble(models_dict, X_val_a, embed_cols, phys_cols)
        val_a_preds = (val_a_probs > 0.5).astype(int)
        f1_a = f1_score(y_val_a, val_a_preds, zero_division=0)

        if f1_a > best_iter_f1:
            best_iter_f1 = f1_a
            val_b_probs = predict_ensemble(models_dict, X_val_b, embed_cols, phys_cols)
            val_b_preds = (val_b_probs > 0.5).astype(int)
            f1_b = f1_score(y_val_b, val_b_preds, zero_division=0)
            final_score = f1_b
            best_models_data = {'models': models_dict, 'final_score': f1_b, 'val_b_pos_count': val_b_pos_count,
                                'init_pos_count': len(labeled_df[labeled_df['label'] == 'AGRI_PV'])}

        if len(unlabeled_pool) == 0: break
        probs = predict_ensemble(models_dict, unlabeled_pool, embed_cols, phys_cols)
        conf_mask = (probs >= config['self_training_conf_threshold']) | (
                    probs <= (1 - config['self_training_conf_threshold']))
        candidates = unlabeled_pool.loc[conf_mask].copy()
        if len(candidates) == 0: break
        candidates.loc[:, 'prob'] = probs[conf_mask]
        candidates.loc[:, 'conf'] = np.abs(candidates['prob'] - 0.5)
        pos_cands = candidates.loc[candidates['prob'] > 0.5].sort_values('conf', ascending=False).head(
            config['max_new_pos_labels'])
        neg_cands = candidates.loc[candidates['prob'] < 0.5].sort_values('conf', ascending=False).head(
            config['max_new_neg_labels'])
        final_cands = pd.concat([pos_cands, neg_cands])
        if len(final_cands) == 0: break
        new_X = final_cands[embed_cols + phys_cols]
        new_y = (final_cands['prob'] > 0.5).astype(int)
        X_train_curr = pd.concat([X_train_curr, new_X])
        y_train_curr = pd.concat([y_train_curr, new_y])
        unlabeled_pool = unlabeled_pool.drop(final_cands.index)

    return {'final_score': final_score, 'data': best_models_data}


# =================== 5. 全局入口 =================== #

def main():
    df, embed_cols, phys_cols = load_and_preprocess_data(CONFIG)
    logger.info(f"特征工程完成。物理特征数: {len(phys_cols)}")

    mixed_df_raw, pure_df_clean, sim_df = compute_similarity_matrix(df, CONFIG, phys_cols)

    # [核心修改] 构建纯负样本时，显式剔除 PURE_PV
    # 我们希望模型学习: AGRI_PV(1) vs [AGRI, IMP, BARE](0)
    # 而不是让 PURE_PV 混在 0 里面干扰 AGRI_PV 的识别
    pure_negatives = pure_df_clean[pure_df_clean['semantic_label'] != 'PURE_PV'].copy()
    logger.info(f"纯负样本构建完成 (剔除PURE_PV): {len(pure_negatives)} 条")

    # 阶段一：粗搜
    logger.info(f"=== 阶段一：粗粒度阈值搜索 {CONFIG['coarse_candidates']} ===")
    coarse_results = []
    for th in CONFIG['coarse_candidates']:
        res = run_experiment(th, mixed_df_raw, pure_negatives, sim_df, embed_cols, phys_cols, CONFIG, mode='coarse')
        if res and res['data']:
            raw_score = res['final_score']
            count = res['data']['val_b_pos_count']
            weighted_score = raw_score * 0.8 if count < 5 else raw_score
            logger.info(f"  Thresh {th}: F1={raw_score:.4f} (Pos={count}) -> Score={weighted_score:.4f}")
            coarse_results.append({'th': th, 'score': weighted_score})

    if not coarse_results: return

    sorted_coarse = sorted(coarse_results, key=lambda x: x['score'], reverse=True)
    top_candidates = sorted_coarse[:2]

    # 阶段二：细搜
    logger.info(f"\n=== 阶段二：多点细搜 (Top-2: {[c['th'] for c in top_candidates]}) ===")
    fine_search_points = []
    step = CONFIG['fine_search_step']
    window = CONFIG['fine_search_window']
    for cand in top_candidates:
        base_th = cand['th']
        local_points = [round(base_th + i * step, 2) for i in range(-window, window + 1)]
        fine_search_points.extend(local_points)
    fine_search_points = sorted(list(set([p for p in fine_search_points if 0.05 <= p <= 0.95])))

    fine_results = []
    for th in fine_search_points:
        res = run_experiment(th, mixed_df_raw, pure_negatives, sim_df, embed_cols, phys_cols, CONFIG, mode='fine')
        if res and res['data']:
            raw_score = res['final_score']
            count = res['data']['val_b_pos_count']
            penalty = 1.0
            if count < 5:
                penalty = 0.85
            elif count < 10:
                penalty = 0.95
            final_metric = raw_score * penalty
            logger.info(f"  [Fine] Thresh {th}: F1={raw_score:.4f} (Pos={count}) -> Metric={final_metric:.4f}")
            fine_results.append({'th': th, 'metric': final_metric, 'models': res['data']['models']})

    if not fine_results: return

    best_final = max(fine_results, key=lambda x: x['metric'])
    best_th_final = best_final['th']
    best_models_dict = best_final['models']

    logger.info(f"\n====== 最终最佳: {best_th_final} (Metric: {best_final['metric']:.4f}) ======")

    logger.info(f"\n====== 最终最佳: {best_th_final} (Metric: {best_final['metric']:.4f}) ======")

    # 保存与推理
    prefix = f"best_th_{int(best_th_final * 100)}"
    joblib.dump(best_models_dict['meta'], os.path.join(MODELS_DIR, f'{prefix}_meta.joblib'))

    target_df = mixed_df_raw.copy()
    final_probs = predict_ensemble(best_models_dict, target_df, embed_cols, phys_cols)
    target_df['final_prob'] = final_probs

    # ================= [修复] 纯度过滤逻辑 ================= #
    logger.info("正在进行纯度过滤 (剔除隐性纯光伏)...")

    # 1. 提取模型中的辅助数据 (之前返回的字典里没有这些，需要重新计算或者从pure_negatives拿)
    # 注意：train_stacking_cv_return_models 返回的字典里并没有 pv_centroid
    # 我们需要重新计算一下 pure_pv 的原型，用来做过滤

    # 重新获取 pure_pv 数据 (利用 compute_similarity_matrix 返回的 pure_df_clean)
    # 注意：main 函数开头已经有了 mixed_df_raw, pure_df_clean, sim_df
    pure_pv_samples = pure_df_clean[pure_df_clean['semantic_label'] == 'PURE_PV']

    if len(pure_pv_samples) > 0:
        # 计算原型
        pv_centroid = pure_pv_samples[phys_cols].median().values.reshape(1, -1)

        # 训练一个临时的 Scaler (仅用于计算这里的相似度)
        filter_scaler = StandardScaler()
        filter_scaler.fit(pure_df_clean[phys_cols])

        # 计算所有样本到纯光伏的相似度
        target_phys_sc = filter_scaler.transform(target_df[phys_cols])
        pv_centroid_sc = filter_scaler.transform(pv_centroid)

        # 赋值给 target_df
        target_df['sim_to_pure_pv'] = cosine_similarity(target_phys_sc, pv_centroid_sc).flatten()
    else:
        logger.warning("未找到 PURE_PV 样本，跳过纯度过滤。")
        target_df['sim_to_pure_pv'] = 0.0  # 默认不相似

    # 设定过滤阈值
    PURE_PV_FILTER_THRESH = 0.985

    # 导出
    for th_conf in [0.5, 0.6, 0.7, 0.8, 0.9]:
        # 1. 先筛选置信度
        base_res = target_df[target_df['final_prob'] >= th_conf].copy()
        total_count = len(base_res)

        if total_count == 0:
            logger.info(f"导出 > {th_conf}: 无满足置信度的样本。")
            continue

        # 2. 再根据相似度过滤 (保留相似度 < 阈值的)
        # 此时 base_res 已经继承了 sim_to_pure_pv 列
        filtered_res = base_res[base_res['sim_to_pure_pv'] < PURE_PV_FILTER_THRESH].copy()
        filtered_count = len(filtered_res)

        removed_count = total_count - filtered_count

        save_path = f"{FINAL_OUTPUT_PREFIX}th{int(best_th_final * 100)}_conf_{int(th_conf * 100)}_filtered.csv"
        filtered_res.to_csv(save_path, index=False)

        logger.info(f"导出 > {th_conf}: 原始 {total_count} -> 剔除纯光伏 {removed_count} -> 最终 {filtered_count} 条")


if __name__ == "__main__":
    main()