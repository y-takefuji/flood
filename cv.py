import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import lightgbm as lgb
import shap

# ============================================================
# 1. Load Data
# ============================================================
df = pd.read_csv('flood.csv')

target_col   = 'FloodProbability'
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols]          # keep as DataFrame for FA
y = df[target_col].values

TOP_N  = 6
TOP_N2 = 5

# ============================================================
# Helpers
# ============================================================
def cv_score(X_sub, y, model_type='LGB', cv=5):
    if model_type == 'RF':
        model = RandomForestRegressor(random_state=42)
    else:
        model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    scores = cross_val_score(model, X_sub, y, cv=cv, scoring='r2')
    return round(float(np.mean(scores)), 4)

def top_features(scores, names, n):
    return [names[i] for i in np.argsort(scores)[::-1][:n]]

def feature_agglomeration_selection(X, y, n):
    """
    X  : pandas DataFrame
    y  : ignored (unsupervised)
    n  : number of features to select

    Steps:
      1. Fit FeatureAgglomeration with n_clusters = min(5, n_features)
      2. Rank all features by their own variance (descending) — globally
      3. Walk through ranked list: pick a feature if its cluster has
         not been seen yet  →  one representative per cluster (the
         highest-variance one)
      4. If fewer than n features selected after the cluster pass,
         fill remaining slots from the leftover features in
         variance-descending order (no cluster constraint)
    Returns list of n selected feature names.
    """
    n_clusters = min(5, X.shape[1])
    fa = FeatureAgglomeration(n_clusters=n_clusters).fit(X)

    feat_var = [
        (X.columns[i], X.iloc[:, i].var(), fa.labels_[i])
        for i in range(X.shape[1])
    ]
    feat_var.sort(key=lambda t: t[1], reverse=True)

    selected, seen_clusters = [], set()
    for feat, _, cluster in feat_var:
        if len(selected) >= n:
            break
        if cluster not in seen_clusters:
            selected.append(feat)
            seen_clusters.add(cluster)

    remaining = [f for f, _, _ in feat_var if f not in selected]
    selected += remaining[:n - len(selected)]
    return selected[:n], None

# ============================================================
# 2. STEP 1 — Top-6 from Full Dataset (each algorithm independent)
# ============================================================
X_arr      = X.values
rng        = np.random.RandomState(42)
sample_idx = rng.choice(len(X_arr), size=min(100, len(X_arr)), replace=False)
X_sample   = X_arr[sample_idx]

# LGB
lgb_model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
lgb_model.fit(X_arr, y)
lgb_imp   = lgb_model.feature_importances_
lgb_top6  = top_features(lgb_imp, feature_cols, TOP_N)

# RF
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_arr, y)
rf_imp   = rf_model.feature_importances_
rf_top6  = top_features(rf_imp, feature_cols, TOP_N)

# LGB-SHAP
exp_lgb       = shap.TreeExplainer(lgb_model)
shap_lgb_vals = np.abs(exp_lgb.shap_values(X_sample)).mean(axis=0)
lgb_shap_top6 = top_features(shap_lgb_vals, feature_cols, TOP_N)

# RF-SHAP
exp_rf        = shap.TreeExplainer(rf_model)
shap_rf_vals  = np.abs(exp_rf.shap_values(X_sample)).mean(axis=0)
rf_shap_top6  = top_features(shap_rf_vals, feature_cols, TOP_N)

# FA
fa_top6, _ = feature_agglomeration_selection(X, y, TOP_N)

# HVGS
hvgs_scores = np.var(X_arr, axis=0)
hvgs_top6   = top_features(hvgs_scores, feature_cols, TOP_N)

# Spearman
sp_scores = np.array([abs(spearmanr(X_arr[:, j], y)[0])
                      for j in range(X_arr.shape[1])])
sp_top6   = top_features(sp_scores, feature_cols, TOP_N)

# ============================================================
# 3. CV6 — Cross-Validation with Top-6 Features
# ============================================================
cv6 = {
    'LGB'      : cv_score(df[lgb_top6].values,      y, 'LGB'),
    'RF'       : cv_score(df[rf_top6].values,        y, 'RF'),
    'LGB-SHAP' : cv_score(df[lgb_shap_top6].values,  y, 'LGB'),
    'RF-SHAP'  : cv_score(df[rf_shap_top6].values,   y, 'RF'),
    'FA'       : cv_score(df[fa_top6].values,         y, 'LGB'),
    'HVGS'     : cv_score(df[hvgs_top6].values,       y, 'LGB'),
    'Spearman' : cv_score(df[sp_top6].values,         y, 'LGB'),
}

# ============================================================
# 4. STEP 2 — Each algorithm removes its OWN #1 feature,
#             creates its OWN reduced dataset,
#             reselects Top-5 using its OWN criterion
# ============================================================
def get_top5(top6, feature_cols, X_full_arr, X_full_df, y, method, rng):
    highest      = top6[0]
    reduced_cols = [c for c in feature_cols if c != highest]
    X_red_arr    = X_full_arr[:, [feature_cols.index(c) for c in reduced_cols]]
    X_red_df     = X_full_df[reduced_cols]

    sample_idx_r = rng.choice(len(X_red_arr), size=min(100, len(X_red_arr)), replace=False)
    X_red_sample = X_red_arr[sample_idx_r]

    if method == 'LGB':
        m = lgb.LGBMRegressor(random_state=42, verbosity=-1)
        m.fit(X_red_arr, y)
        scores_r = m.feature_importances_
        return top_features(scores_r, reduced_cols, TOP_N2)

    elif method == 'RF':
        m = RandomForestRegressor(random_state=42)
        m.fit(X_red_arr, y)
        scores_r = m.feature_importances_
        return top_features(scores_r, reduced_cols, TOP_N2)

    elif method == 'LGB-SHAP':
        m = lgb.LGBMRegressor(random_state=42, verbosity=-1)
        m.fit(X_red_arr, y)
        exp      = shap.TreeExplainer(m)
        scores_r = np.abs(exp.shap_values(X_red_sample)).mean(axis=0)
        return top_features(scores_r, reduced_cols, TOP_N2)

    elif method == 'RF-SHAP':
        m = RandomForestRegressor(random_state=42)
        m.fit(X_red_arr, y)
        exp      = shap.TreeExplainer(m)
        scores_r = np.abs(exp.shap_values(X_red_sample)).mean(axis=0)
        return top_features(scores_r, reduced_cols, TOP_N2)

    elif method == 'FA':
        selected, _ = feature_agglomeration_selection(X_red_df, y, TOP_N2)
        return selected

    elif method == 'HVGS':
        scores_r = np.var(X_red_arr, axis=0)
        return top_features(scores_r, reduced_cols, TOP_N2)

    elif method == 'Spearman':
        scores_r = np.array([abs(spearmanr(X_red_arr[:, j], y)[0])
                              for j in range(X_red_arr.shape[1])])
        return top_features(scores_r, reduced_cols, TOP_N2)


top5 = {
    'LGB'      : get_top5(lgb_top6,      feature_cols, X_arr, X, y, 'LGB',      rng),
    'RF'       : get_top5(rf_top6,        feature_cols, X_arr, X, y, 'RF',       rng),
    'LGB-SHAP' : get_top5(lgb_shap_top6,  feature_cols, X_arr, X, y, 'LGB-SHAP', rng),
    'RF-SHAP'  : get_top5(rf_shap_top6,   feature_cols, X_arr, X, y, 'RF-SHAP',  rng),
    'FA'       : get_top5(fa_top6,         feature_cols, X_arr, X, y, 'FA',       rng),
    'HVGS'     : get_top5(hvgs_top6,       feature_cols, X_arr, X, y, 'HVGS',     rng),
    'Spearman' : get_top5(sp_top6,         feature_cols, X_arr, X, y, 'Spearman', rng),
}

top6_map = {
    'LGB'      : lgb_top6,
    'RF'       : rf_top6,
    'LGB-SHAP' : lgb_shap_top6,
    'RF-SHAP'  : rf_shap_top6,
    'FA'       : fa_top6,
    'HVGS'     : hvgs_top6,
    'Spearman' : sp_top6,
}

# ============================================================
# 5. Final Table: Method | CV6 R² | Top-6 | Top-5
# ============================================================
rows = []
for method in top6_map:
    rows.append({
        'Method'        : method,
        'CV6_R2'        : cv6[method],
        'Top6_Features' : ', '.join(top6_map[method]),
        'Top5_Features' : ', '.join(top5[method]),
    })

summary_df = pd.DataFrame(rows, columns=['Method', 'CV6_R2', 'Top6_Features', 'Top5_Features'])

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(summary_df.to_string(index=False))

summary_df.to_csv('result.csv', index=False)
print("\nSaved as result.csv")
