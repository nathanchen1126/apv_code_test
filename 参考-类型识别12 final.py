import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, silhouette_score, roc_auc_score, precision_score, recall_score, confusion_matrix, \
    balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging
import datetime
import sys
import warnings
import re

# --- 0. 环境与依赖设置 ---
N_JOBS = 4
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

CONFIG = {
    # 请修改为您的实际文件路径
    'input_path': r'C:\Users\zgy\Desktop\papers\nygf\alldata.csv',
    'output_dir': 'pseudo_labeling_results',

    # --- 聚类与相似度核心配置 ---
    'n_agri_prototypes': 2,
    'feature_cumulative_threshold': 0.50,

    # 核心物理特征前缀 (用于引导特征选择，匹配月度/年度)
        'core_clustering_features': [
        # 1. 关键的年度物候特征
        'NDVI_mean_annual_amp',      # 植被年振幅 (最重要)
        'NDVI_mean_annual_max',      # 植被年最大值
        'NDVI_mean_peak_month',      # 植被峰值月
        'NDVI_mean_grow_len',        # 生长季长度

        # 2. 关键波段的年度均值
        'B4_mean_annual_mean',       # 红光 (植被吸收)
        'B8_mean_annual_mean',       # 近红外 (植被反射)
        'B11_mean_annual_mean',      # 短波红外1 (水分/土壤/光伏板敏感)
        'B12_mean_annual_mean',      # 短波红外2

        # 3. 关键的年度纹理和高程
        'B8_contrast_mean_annual_mean',   # 近红外纹理 (光伏板和农田纹理不同)
        'NDVI_contrast_mean_annual_mean', # NDVI 纹理
        'Elevation_mean_annual_mean',     # 高程
    ],

    # --- 阈值搜索 ---
    'coarse_candidates': [0.30, 0.35, 0.40, 0.45],
    'fine_search_step': 0.01,
    'fine_search_window': 2,

    # --- 自训练参数 ---
    'self_training_conf_threshold': 0.90,  # 提高门槛以求更纯净的伪标签
    'max_iter': 5,
    'max_new_pos_labels': 100,  # 限制每次新增伪标签的正样本数
    'max_new_neg_labels': 100,  # 限制每次新增伪标签的负样本数，与正样本平衡
    'initial_neg_sample_ratios': {  # 初始负样本各类别抽样比例 (相对初始正样本数)
        'AGRI': 1.0,  # 纯耕地 (最困难负样本，初期多学习)
        'PURE_PV': 0.8,  # 纯光伏 (关键参考，初期多学习)
        'IMP': 0.5,  # 不透水面 (中等难度负样本)
        'BARE': 0.5  # 裸土 (中等难度负样本)
    },
    'hold_out_test_size': 0.2,  # 【新增】Hold-out 评估集比例

    # --- 模型参数 ---
    'n_splits': 5,  # K-Fold 交叉验证折数
    'random_state': 42,

    'lgb_params': {
        'objective': 'binary', 'class_weight': 'balanced', 'n_estimators': 1000,
        'learning_rate': 0.03, 'num_leaves': 20, 'max_depth': 7,
        'n_jobs': N_JOBS, 'verbosity': -1, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': 42
    },
    'xgb_params': {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 1000,
        'learning_rate': 0.03, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'n_jobs': N_JOBS, 'verbosity': 0, 'early_stopping_rounds': 50,
        'random_state': 42
    },

    # --- 列定义 ---
    'id_cols': ['ID', 'ID1', 'type', 'label', 'cluster', 'Lat', 'Lon', 'X', 'Y', 'lat', 'lon'],
    'pure_types': ['gd', 'btsm', 'ld', 'cgf'],  # gd:耕地, btsm:不透水面, ld:裸地, cgf:纯光伏
    'mixed_types': ['hhgf'],  # hhgf: 混合光伏
    'label_map': {'gd': 'AGRI', 'btsm': 'IMP', 'ld': 'BARE', 'cgf': 'PURE_PV'},
    'isolation_forest_contamination': 0.05,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
MODELS_DIR = os.path.join(CONFIG['output_dir'], 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# =================== 2. 数据加载与预处理 (含时序插值) =================== #

def preprocess_missing_values_time_series(df, feature_cols):
    logger.info(">>> 使用高级缺失值处理 (时序插值 + 全局均值)...")
    df_imputed = df.copy()
    ts_groups = {}
    pattern = re.compile(r'^(.*)_(0[1-9]|1[0-2])$')  # 严格匹配 _01 到 _12
    for col in feature_cols:
        match = pattern.match(col)
        if match:
            base_name, _ = match.groups()
            if base_name not in ts_groups: ts_groups[base_name] = []
            ts_groups[base_name].append(col)

    for base_name, cols in ts_groups.items():
        cols.sort(key=lambda x: int(x.split('_')[-1]))
        if len(cols) < 3: continue

        subset = df_imputed.loc[:, cols]
        if subset.isnull().any().any():
            subset_interp = subset.interpolate(axis=1, limit_direction='both', method='linear')
            df_imputed.loc[:, cols] = subset_interp

    numeric_cols = df_imputed[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if df_imputed[numeric_cols].isnull().any().any():
        imputer = SimpleImputer(strategy='mean')
        df_imputed.loc[:, numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])

    df_imputed.loc[:, numeric_cols] = df_imputed[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
    return df_imputed


def load_and_preprocess_data(config):
    path = config['input_path']
    if not os.path.exists(path):
        logger.error(f"文件不存在: {path}")
        sys.exit(1)

    try:
        df = pd.read_csv(path, encoding='utf-8')
        if len(df.columns) < 5: df = pd.read_csv(path, encoding='utf-8', sep='\t')
    except:
        try:
            df = pd.read_csv(path, encoding='gbk')
        except:
            df = pd.read_csv(path, encoding='gbk', sep='\t')

    all_cols = list(df.columns)
    id_cols_present = [c for c in config['id_cols'] if c in all_cols]
    embed_pattern = re.compile(r'^A\d+')
    embed_cols = [c for c in all_cols if embed_pattern.match(c)]

    phys_cols_candidates = [c for c in all_cols if c not in id_cols_present and c not in embed_cols]
    phys_cols_raw = df[phys_cols_candidates].select_dtypes(include=[np.number]).columns.tolist()

    df_feats = preprocess_missing_values_time_series(df, phys_cols_raw)

    df_feats['phys_mean'] = df_feats[phys_cols_raw].mean(axis=1)
    df_feats['phys_std'] = df_feats[phys_cols_raw].std(axis=1)
    df_feats['phys_range'] = df_feats[phys_cols_raw].max(axis=1) - df_feats[phys_cols_raw].min(axis=1)

    phys_cols = phys_cols_raw + ['phys_mean', 'phys_std', 'phys_range']
    df_final = pd.concat([df[id_cols_present], df_feats[embed_cols + phys_cols]], axis=1)

    return df_final, embed_cols, phys_cols


# =================== 3. 特征选择 =================== #
def get_hybrid_clustering_features(X_data, y_data, all_phys_cols, config):
    logger.info(">>> 特征选择...")
    X_clean = X_data.replace([np.inf, -np.inf], 0).fillna(0)
    manual_feats = []
    if config.get('core_clustering_features'):
        for core_pattern in config['core_clustering_features']:
            matches = [c for c in all_phys_cols if core_pattern in c]
            manual_feats.extend(matches)
    manual_feats = list(set(manual_feats))

    lgb_params = {'objective': 'binary', 'n_estimators': 200, 'class_weight': 'balanced', 'verbose': -1,
                  'n_jobs': N_JOBS, 'random_state': config['random_state']}
    auto_feats = []
    try:
        if len(X_clean) > 5000:
            X_sub, _, y_sub, _ = train_test_split(X_clean, y_data, train_size=5000, stratify=y_data,
                                                  random_state=config['random_state'])
        else:
            X_sub, y_sub = X_clean, y_data

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_sub, y_sub)

        imp_df = pd.DataFrame({'feature': all_phys_cols, 'gain': model.feature_importances_}).sort_values('gain',
                                                                                                          ascending=False)

        if imp_df['gain'].sum() > 0:
            imp_df['cum'] = imp_df['gain'].cumsum() / imp_df['gain'].sum()
            auto_feats = imp_df[imp_df['cum'] <= config['feature_cumulative_threshold']]['feature'].tolist()
            if len(auto_feats) < 5: auto_feats = imp_df.head(5)['feature'].tolist()
        else:
            auto_feats = all_phys_cols
    except Exception as e:
        logger.warning(f"特征选择异常: {e}")
        auto_feats = all_phys_cols

    final = list(set(auto_feats + manual_feats))
    final = [c for c in final if c in X_data.columns]
    logger.info(f"   最终相似度特征数: {len(final)}")
    return final


# =================== 4. 相似度计算与初始标注 =================== #
def compute_similarity_matrix(df, config, all_phys_cols):
    pure_mask = df['type'].isin(config['pure_types'])
    mixed_mask = df['type'].isin(config['mixed_types'])
    pure_df = df.loc[pure_mask].copy()
    mixed_df = df.loc[mixed_mask].copy()
    pure_df['semantic_label'] = pure_df['type'].map(config['label_map'])
    pure_df = pure_df.dropna(subset=['semantic_label'])

    # 异常值清洗
    clean_samples = []
    for label, sub in pure_df.groupby('semantic_label'):
        if len(sub) > 20:
            clf = IsolationForest(contamination=config['isolation_forest_contamination'],
                                  random_state=config['random_state'], n_jobs=N_JOBS)
            preds = clf.fit_predict(sub[all_phys_cols])
            clean_samples.append(sub[preds == 1])
        else:
            clean_samples.append(sub)
    if not clean_samples:
        logger.error("无有效纯样本！")
        sys.exit(1)

    pure_clean = pd.concat(clean_samples)

    temp_X = pure_clean[all_phys_cols]
    temp_y = (pure_clean['semantic_label'] == 'AGRI').astype(int)  # 用于特征选择的 y
    sim_cols = get_hybrid_clustering_features(temp_X, temp_y, all_phys_cols, config)
    if not sim_cols: sim_cols = all_phys_cols  # 保底

    scaler_sim = StandardScaler().fit(pure_clean[sim_cols])
    mixed_scaled = scaler_sim.transform(mixed_df[sim_cols])
    protos, labels = [], []
    for label in ['AGRI', 'IMP', 'BARE', 'PURE_PV']:  # 确保包含所有需要的原型
        sub = pure_clean[pure_clean['semantic_label'] == label]
        if sub.empty: continue

        # AGRI 多原型逻辑 (如果需要)
        if label == 'AGRI' and config.get('n_agri_prototypes', 1) > 1 and len(sub) > config['n_agri_prototypes'] * 2:
            km = KMeans(n_clusters=config['n_agri_prototypes'], random_state=config['random_state'], n_init=10).fit(
                sub[sim_cols])
            for i, ctr in enumerate(km.cluster_centers_):
                protos.append(ctr)
                labels.append(f"AGRI_sub{i}")
        else:
            protos.append(sub[sim_cols].median().values)
            labels.append(label)

    if not protos:
        logger.error("原型构建失败！")
        sys.exit(1)

    proto_matrix = np.vstack(protos)
    sim_raw = cosine_similarity(mixed_scaled, scaler_sim.transform(proto_matrix))
    sim_df_raw = pd.DataFrame(sim_raw, index=mixed_df.index, columns=labels)

    # --- 核心修改：聚合相似度，并返回一个包含所有原型相似度的 DataFrame ---
    final_sim = pd.DataFrame(index=mixed_df.index)

    # 聚合 AGRI 相似度
    agri_cols = [c for c in labels if c.startswith('AGRI')]
    if agri_cols:
        final_sim['AGRI'] = sim_df_raw[agri_cols].max(axis=1)

    # 保留其他原型的相似度
    for label in ['IMP', 'BARE', 'PURE_PV']:
        if label in sim_df_raw.columns:
            final_sim[label] = sim_df_raw[label]

    return mixed_df, pure_clean, final_sim, sim_cols


def get_initial_labels(mixed_df, sim_df, threshold):
    df_out = mixed_df.copy()

    # 确保所有需要的相似度列都存在
    required_sim_cols = ['AGRI', 'IMP', 'BARE']
    for col in required_sim_cols:
        if col not in sim_df.columns:
            sim_df[col] = 0.0  # 如果某个原型不存在，其相似度为0

    def _label(row):
        # 1. 竞争性打标：找出相似度最高的原型
        winner_class = row[required_sim_cols].idxmax()

        # 2. 双重条件判断
        # 条件1: 获胜者必须是 AGRI
        # 条件2: 与 AGRI 的相似度必须大于阈值
        if winner_class == 'AGRI' and row['AGRI'] >= threshold:
            return 'AGRI_PV'
        else:
            # 所有其他情况，包括与 IMP/BARE 更相似，或者与 AGRI 相似但未达到阈值
            # 暂时都归为不确定，等待自训练处理
            return 'UNCERTAIN_MIXED'

    df_out['label'] = sim_df.apply(_label, axis=1)
    return df_out


# =================== 5. 可视化 =================== #
def visualize_tsne(mixed_df, pure_df, sim_df, sim_cols, thresholds, output_dir, config):  # 增加 sim_cols 参数
    logger.info("生成 t-SNE...")
    n_sample = 3000
    df_all = pd.concat([
        pure_df.assign(grp='Pure'),
        mixed_df.assign(grp='Mixed')
    ])

    if len(df_all) > n_sample:
        df_all = df_all.sample(n_sample, random_state=config['random_state'])

    if len(df_all) < 50:
        logger.warning("样本过少 (<50)，跳过 t-SNE")
        return

    X = df_all[sim_cols].values  # 使用 sim_cols
    X = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=min(30, len(df_all) - 1), random_state=config['random_state'])
    emb = tsne.fit_transform(X)
    df_all['x'], df_all['y'] = emb[:, 0], emb[:, 1]

    for th in thresholds:
        labels = get_initial_labels(mixed_df, sim_df, th)['label']
        label_map = labels.to_dict()

        def get_color(row):
            if row['grp'] == 'Pure': return 'Pure ' + row.get('semantic_label', 'Other')
            l = label_map.get(row.name, 'UNCERTAIN')
            return 'Agri-PV' if l == 'AGRI_PV' else 'Other Mixed'

        df_all['display'] = df_all.apply(get_color, axis=1)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_all, x='x', y='y', hue='display', style='grp',
                        s=30, alpha=0.7, edgecolor='none')
        plt.title(f'Thresh={th}')
        plt.savefig(os.path.join(output_dir, f'tsne_{th}.png'))
        plt.close()


# =================== 6. 模型训练 (全特征学习 + 无泄漏) =================== #

def train_stacking_cv(X, y, embed_cols, phys_cols, config, sim_cols=None):
    """
    K-Fold 训练并返回 Meta-Learner 和 Fold Models (用于后续预测)
    """
    oof_preds = np.zeros((len(y), 2))
    folds = []
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['random_state'])

    for tr_ix, val_ix in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_ix], X.iloc[val_ix]
        y_tr, y_val = y.iloc[tr_ix], y.iloc[val_ix]

        # 1. 独立 Scaling
        sc_e = StandardScaler().fit(X_tr[embed_cols])
        # 【策略】这里物理模型 M2 默认使用全部物理特征 phys_cols
        # 如果您希望 M2 只用 sim_cols，可以将下面的 phys_cols 改为 sim_cols
        cols_for_m2 = phys_cols
        sc_s = StandardScaler().fit(X_tr[cols_for_m2])

        X_tr_e = sc_e.transform(X_tr[embed_cols])
        X_tr_s = sc_s.transform(X_tr[cols_for_m2])
        X_val_e = sc_e.transform(X_val[embed_cols])
        X_val_s = sc_s.transform(X_val[cols_for_m2])

        m1 = lgb.LGBMClassifier(**config['lgb_params'])
        m1.fit(X_tr_e, y_tr, eval_set=[(X_val_e, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        if HAS_XGB:
            m2 = xgb.XGBClassifier(**config['xgb_params'])
            m2.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        else:
            m2 = lgb.LGBMClassifier(**config['lgb_params'])
            m2.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

        p1 = m1.predict_proba(X_val_e)[:, 1]
        p2 = m2.predict_proba(X_val_s)[:, 1]
        oof_preds[val_ix] = np.column_stack([p1, p2])

        # 保存模型和该折的 Scaler
        folds.append({'e': m1, 's': m2, 'sc_e': sc_e, 'sc_s': sc_s})

    meta_base = LogisticRegression(class_weight='balanced', random_state=config['random_state'])
    # 移除 cv='prefit'，使用标准的 cv=3 流程
    meta = CalibratedClassifierCV(meta_base, method='isotonic', cv=3)
    # meta 会在内部对 OOF 数据做校准
    meta.fit(pd.DataFrame(oof_preds, columns=['e', 's']), y)

    return {'meta': meta, 'folds': folds}


def predict_ensemble(models, X, embed_cols, phys_cols, sim_cols=None):
    pe_list, ps_list = [], []
    cols_for_m2 = phys_cols  # 保持与训练时一致

    for fold in models['folds']:
        X_e = fold['sc_e'].transform(X[embed_cols])
        X_s = fold['sc_s'].transform(X[cols_for_m2])
        pe_list.append(fold['e'].predict_proba(X_e)[:, 1])
        ps_list.append(fold['s'].predict_proba(X_s)[:, 1])
    pe = np.mean(pe_list, axis=0)
    ps = np.mean(ps_list, axis=0)
    return models['meta'].predict_proba(pd.DataFrame({'e': pe, 's': ps}))[:, 1]


# =================== 7. 实验主循环 (K-Fold 评估 + 负样本策略) =================== #

def run_experiment(threshold, mixed_df, pure_dev_source, sim_df, embed_cols, phys_cols, sim_cols, config, prefix=""):
    """
    V8 改进版：
    pure_dev_source: 仅包含用于开发（训练+自训练）的纯样本，严禁包含 Hold-out 集！
    """
    curr_df = get_initial_labels(mixed_df, sim_df, threshold)
    labeled = curr_df[curr_df['label'] != 'UNCERTAIN_MIXED']

    # 1. 初始正样本 (来自混合样本)
    pos_samples = labeled[labeled['label'] == 'AGRI_PV']
    if len(pos_samples) < 5:
        logger.warning(f"{prefix} Th={threshold}: 初始正样本不足 (<5)，跳过。")
        return None

    feature_cols = embed_cols + phys_cols

    X_pos = pos_samples[feature_cols]
    y_pos = pd.Series(1, index=X_pos.index)

    # 2. 初始负样本构造 (仅从 pure_dev_source 中采样)
    initial_neg_samples = pd.DataFrame()

    # 记录哪些纯样本被用作了初始负样本，剩下的 AGRI 要进 Pool
    used_pure_indices = []

    for neg_label, ratio in config['initial_neg_sample_ratios'].items():
        # 从开发集中筛选对应类别的样本
        subset = pure_dev_source[pure_dev_source['semantic_label'] == neg_label]

        sample_size = min(len(subset), max(1, int(len(pos_samples) * ratio)))
        if sample_size > 0:
            sampled_neg = subset.sample(sample_size, random_state=config['random_state'])
            initial_neg_samples = pd.concat([initial_neg_samples, sampled_neg])
            used_pure_indices.extend(sampled_neg.index.tolist())
            logger.info(f"   [Init Train] 加入 {len(sampled_neg)} 条纯 {neg_label}。")

    if initial_neg_samples.empty:
        return None

    X_neg = initial_neg_samples[feature_cols]
    y_neg = pd.Series(0, index=X_neg.index)

    X_cur = pd.concat([X_pos, X_neg])
    y_cur = pd.concat([y_pos, y_neg])

    # 3. Pool 构造 (混合样本 + 剩余的困难纯耕地)
    # A. 未标记的混合样本
    pool_mixed = mixed_df[~mixed_df.index.isin(pos_samples.index)][feature_cols].copy()
    pool_mixed['prob'] = 0.5  # 初始中立

    # B. 【关键】将 pure_dev_source 中剩余的 AGRI (纯耕地) 放入 Pool
    # 这些是模型必须学会区分的 "Hard Negatives"
    # 注意：只放 AGRI，不放 IMP/BARE (太简单)，PURE_PV 也可以考虑放
    remaining_agri = pure_dev_source[
        (pure_dev_source['semantic_label'] == 'AGRI') &
        (~pure_dev_source.index.isin(used_pure_indices))
        ][feature_cols].copy()

    remaining_agri['prob'] = 0.0  # 暗示它是负的，但允许模型改判(虽然不太可能)

    pool = pd.concat([pool_mixed, remaining_agri])

    logger.info(
        f"{prefix} Th={threshold} | Train: {len(X_cur)} (Pos:{len(X_pos)} Neg:{len(X_neg)}) | Pool: {len(pool)} (含 {len(remaining_agri)} AGRI)")

    best_res, best_score = None, 0

    # 4. 迭代循环 (使用 K-Fold 评估稳定性)
    skf_eval = StratifiedKFold(n_splits=3, shuffle=True, random_state=config['random_state'])

    for i in range(config['max_iter']):
        if sum(y_cur) < 5: break

        # --- K-Fold 训练与评估 ---
        try:
            oof_preds = np.zeros(len(y_cur))
            model_folds = []

            # K-Fold 训练
            for tr_idx, val_idx in skf_eval.split(X_cur, y_cur):
                X_t, X_v = X_cur.iloc[tr_idx], X_cur.iloc[val_idx]
                y_t, y_v = y_cur.iloc[tr_idx], y_cur.iloc[val_idx]

                # 训练 Fold 模型 (M2 使用全部 phys_cols)
                curr_model = train_stacking_cv(X_t, y_t, embed_cols, phys_cols, config, sim_cols=sim_cols)

                # 预测验证集
                p_v = predict_ensemble(curr_model, X_v, embed_cols, phys_cols, sim_cols)
                oof_preds[val_idx] = p_v
                model_folds.append(curr_model)  # 这里存的是含meta的完整模型

            # 计算当前训练集上的 CV 分数
            avg_f1 = f1_score(y_cur, (oof_preds > 0.5).astype(int), zero_division=0)
            avg_auc = roc_auc_score(y_cur, oof_preds)

            # 重新在全量当前数据上训练一个模型用于 Pool 预测
            model_full = train_stacking_cv(X_cur, y_cur, embed_cols, phys_cols, config, sim_cols)

        except Exception as e:
            logger.error(f"   Iter {i + 1} 错误: {e}");
            break

        logger.info(f"   -> Iter {i + 1}: CV F1={avg_f1:.4f} AUC={avg_auc:.4f} | Pool={len(pool)}")

        if avg_f1 >= best_score:
            best_score = avg_f1
            # 注意：这里保存的是全量模型
            best_res = {'m': model_full, 's': avg_f1, 'p': sum(y_cur)}

        if pool.empty: break

        # --- 自训练筛选 ---
        probs = predict_ensemble(model_full, pool[feature_cols], embed_cols, phys_cols, sim_cols)

        # 构造候选集
        cands = pool.copy()
        cands['prob'] = probs
        cands['conf'] = abs(cands['prob'] - 0.5)

        # 筛选高置信度
        high_conf_mask = (probs >= config['self_training_conf_threshold']) | (
                    probs <= 1 - config['self_training_conf_threshold'])
        cands = cands[high_conf_mask]

        if cands.empty:
            logger.info("   -> 无高置信度样本，停止。")
            break

        pos_new = cands[cands['prob'] > 0.5].nlargest(config['max_new_pos_labels'], 'conf')
        neg_new = cands[cands['prob'] < 0.5].nlargest(config['max_new_neg_labels'], 'conf')

        # 平衡控制：负样本不少于正样本的 1/3
        if len(pos_new) > 0:
            min_neg = int(len(pos_new) / 3)
            if len(neg_new) < min_neg:
                # 尝试放宽一点点负样本要求，或者直接截断正样本
                pos_new = pos_new.head(max(1, len(neg_new) * 3))

        if pos_new.empty and neg_new.empty: break

        logger.info(f"      + 新增: {len(pos_new)} Pos, {len(neg_new)} Neg")

        # 更新训练集
        X_cur = pd.concat([X_cur, pos_new[feature_cols], neg_new[feature_cols]])
        y_cur = pd.concat([y_cur, pd.Series(1, index=pos_new.index), pd.Series(0, index=neg_new.index)])

        # 从 Pool 中移除
        pool = pool.drop(pos_new.index).drop(neg_new.index)

    if best_res is None: return {'score': 0, 'data': None}
    return {'score': best_res['s'], 'data': best_res}


# =================== 8. 主流程 =================== #

def main():
    # 1. 加载
    df, embed_cols, all_phys_cols = load_and_preprocess_data(CONFIG)
    if 'ID' not in df.columns: df['ID'] = df.index
    logger.info(f"Loaded: {len(df)}, Features: {len(embed_cols) + len(all_phys_cols)}")

    # 2. 相似度计算 (对全量数据计算特征，不涉及标签泄露)
    mixed_raw, pure_clean, sim_df, sim_cols = compute_similarity_matrix(df, CONFIG, all_phys_cols)

    # === V8 核心：构建 Hold-out 独立评估集 ===
    # 目标：从纯样本中切分出 20% 永远不参与训练，只用于最后算指标

    # 确保每个类别都有样本进入 hold-out
    pure_dev_list = []
    pure_holdout_list = []

    for label in ['AGRI', 'IMP', 'BARE', 'PURE_PV']:
        sub = pure_clean[pure_clean['semantic_label'] == label]
        if len(sub) > 5:
            dev, holdout = train_test_split(sub, test_size=0.2, random_state=CONFIG['random_state'])
            pure_dev_list.append(dev)
            pure_holdout_list.append(holdout)
        else:
            # 样本太少全部用于开发
            pure_dev_list.append(sub)

    pure_dev_source = pd.concat(pure_dev_list)  # 用于训练和 Pool 的源头
    pure_holdout_set = pd.concat(pure_holdout_list)  # 锁进保险箱，只看最后一次

    logger.info(f"数据集划分: 开发集纯样本 {len(pure_dev_source)} | Hold-out 独立评估集 {len(pure_holdout_set)}")

    # 3. 粗搜
    logger.info("\n>>> 阶段一：粗粒度搜索 (Coarse Search)...")
    results = []
    for th in CONFIG['coarse_candidates']:
        # 注意：这里传入的是 pure_dev_source
        res = run_experiment(th, mixed_raw, pure_dev_source, sim_df, embed_cols, all_phys_cols, sim_cols, CONFIG,
                             prefix="[Coarse]")
        if res and res['data']:
            # 加权分数
            sc = res['score'] * (0.85 if res['data']['p'] < 5 else 1.0)
            logger.info(f"[Coarse] Th={th} CV-F1={res['score']:.4f}")
            results.append({'th': th, 'score': sc, 'raw_res': res})

    if not results:
        logger.error("粗搜未找到有效结果。")
        sys.exit(1)

    # 4. 细搜
    top_cands = sorted(results, key=lambda x: x['score'], reverse=True)[:1]
    fine_pts = set()
    for c in top_cands:
        for i in range(-CONFIG['fine_search_window'], CONFIG['fine_search_window'] + 1):
            if i == 0: continue
            p = round(c['th'] + i * CONFIG['fine_search_step'], 2)
            if 0.01 <= p <= 0.95: fine_pts.add(p)

    logger.info(f"\n>>> 阶段二：细粒度搜索... Points: {sorted(list(fine_pts))}")

    final_results = results.copy()
    for th in sorted(list(fine_pts)):
        if any(abs(r['th'] - th) < 1e-5 for r in final_results): continue
        res = run_experiment(th, mixed_raw, pure_dev_source, sim_df, embed_cols, all_phys_cols, sim_cols, CONFIG,
                             prefix="[Fine]")
        if res and res['data']:
            sc = res['score'] * (0.85 if res['data']['p'] < 5 else 1.0)
            logger.info(f"[Fine] Th={th} CV-F1={res['score']:.4f}")
            final_results.append({'th': th, 'score': sc, 'raw_res': res})

    # 5. 最佳模型应用
    best = max(final_results, key=lambda x: x['score'])
    logger.info(f"\n====== 最佳阈值: {best['th']} (CV-F1={best['score']:.4f}) ======")

    model = best['raw_res']['data']['m']
    joblib.dump(model, os.path.join(MODELS_DIR, f'best_model_{TIMESTAMP}.joblib'))

    # 6. 最终预测
    target = mixed_raw.copy()
    target['prob'] = predict_ensemble(model, target, embed_cols, all_phys_cols, sim_cols)

    # === 7. 终极可信度评估 (使用 Hold-out) ===
    logger.info("\n>>> 【关键】Hold-out 独立集评估报告:")
    if not pure_holdout_set.empty:
        # 预测 Hold-out 集
        ho_probs = predict_ensemble(model, pure_holdout_set, embed_cols, all_phys_cols, sim_cols)
        # 因为 Hold-out 全是负样本 (纯耕地、纯光伏、IMP、BARE 都是负)
        # 所以真实标签 y_true 全为 0
        y_ho_true = np.zeros(len(pure_holdout_set))
        y_ho_pred = (ho_probs > 0.5).astype(int)

        # 计算整体误报率 (False Positive Rate)
        total_fpr = y_ho_pred.mean()
        logger.info(f"   Hold-out 整体误报率: {total_fpr:.2%} ({sum(y_ho_pred)}/{len(y_ho_pred)})")

        # 分类别评估误报率
        pure_holdout_set['ho_prob'] = ho_probs
        pure_holdout_set['ho_pred'] = y_ho_pred

        for label in ['AGRI', 'PURE_PV', 'IMP', 'BARE']:
            sub = pure_holdout_set[pure_holdout_set['semantic_label'] == label]
            if len(sub) > 0:
                fp_rate = sub['ho_pred'].mean()
                logger.info(f"   [{label}] 误报率: {fp_rate:.2%} (样本数: {len(sub)})")
                if label == 'AGRI' and fp_rate > 0.3:
                    logger.warning("   ⚠️ 警告: 纯耕地误报率较高，模型可能难以区分耕地与农光。")
                if label == 'PURE_PV' and fp_rate > 0.2:
                    logger.warning("   ⚠️ 警告: 纯光伏误报率较高，模型可能混淆了光伏特征。")
                else:
                    logger.warning("   Hold-out 集为空，无法进行独立评估。")
    # 8. 导出结果 (加入纯度过滤)
    sc_pure = StandardScaler().fit(pure_clean[sim_cols])  # 用全量纯样本 fit scaler

    final = pd.DataFrame()
    if 'PURE_PV' in pure_clean['semantic_label'].values:
        pv_samples = pure_clean[pure_clean['semantic_label'] == 'PURE_PV']
        if not pv_samples.empty:
            pv_proto = pv_samples[sim_cols].median().values.reshape(1, -1)
            sim_pv = cosine_similarity(sc_pure.transform(target[sim_cols]), sc_pure.transform(pv_proto)).flatten()

            # 输出逻辑：概率 > 0.5 且 不像纯光伏
            final = target[(target['prob'] > 0.8) & (sim_pv < 0.985)]
        else:
            final = target[target['prob'] > 0.8]
    else:
        final = target[target['prob'] > 0.8]

    out_path = os.path.join(CONFIG['output_dir'], f'final_{TIMESTAMP}.csv')
    final.to_csv(out_path, index=False)
    logger.info(f"最终结果已保存: {out_path} ({len(final)} 条)")


if __name__ == "__main__":
    main()
