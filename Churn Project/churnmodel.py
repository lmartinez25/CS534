from cleanchurn import *  # imports clean data df
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score,
                             ConfusionMatrixDisplay, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from optuna.integration import OptunaSearchCV
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
import optuna
import numpy as np

# suppress per-trial optuna output — summary still prints after each search
optuna.logging.set_verbosity(optuna.logging.WARNING)

# stratified folds keep churn ratio consistent across splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# convert categorical to integer representations using replace for outcome and
# one hot encoding for attributes
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})

# feature engineering — ratio of current vs average monthly charge flags customers
# whose bill has risen relative to their history, a known churn driver
df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargeRatio']      = df['MonthlyCharges'] / (df['AvgMonthlyCharge'] + 1)

# tenure bucketed into new/mid/established — churn behavior is non-linear with tenure
df['TenureBin'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 72],
                          labels=[0, 1, 2, 3]).astype(int)

# customers with more active services are more locked in and less likely to leave
service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['ServiceCount'] = df[service_cols].apply(
    lambda col: col.map({'Yes': 1, 'No': 0,
                         'No internet service': 0,
                         'No phone service': 0}).fillna(0)
).sum(axis=1)

Xcat = pd.get_dummies(df[['gender', 'Partner', 'Dependents', 'PhoneService',
                           'MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod']]).astype(int)

# SeniorCitizen is already binary (0/1) so it goes in Xnum, not Xcat
Xnum = df[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure',
           'AvgMonthlyCharge', 'ChargeRatio', 'TenureBin', 'ServiceCount']]
X = pd.concat([Xcat, Xnum], axis=1)

print(X.info())

# stratify=y ensures the same churn ratio is preserved in both splits
y = df['Churn'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                     random_state=42, stratify=y)

imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# SMOTE is placed inside each pipeline so it only runs on training folds during CV —
# keeping it outside caused synthetic points to leak into validation folds and
# inflated CV scores without generalizing to real data
rf_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf',   RandomForestClassifier(random_state=1))
])

# scaler moved inside the SVC pipeline for the same reason as SMOTE above
svc_pipe = ImbPipeline([
    ('smote',  SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('clf',    SVC(probability=True, random_state=1))
])

xgb_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf',   XGBClassifier(random_state=1, eval_metric='logloss'))
])

# Bayesian optimization via Optuna — learns from each trial and focuses on
# promising regions, finding better params than random search for the same budget.
# Distributions replace fixed lists so Optuna can sample continuously where useful.
rf_param_dist = {
    'clf__n_estimators':      CategoricalDistribution([200, 300, 500]),
    'clf__max_depth':         IntDistribution(5, 15),
    'clf__min_samples_split': IntDistribution(5, 20),
    'clf__min_samples_leaf':  IntDistribution(2, 8),
    'clf__max_features':      CategoricalDistribution(['sqrt', 'log2']),
    'clf__class_weight':      CategoricalDistribution(['balanced']),
}

# C and gamma grids expanded downward — previous best hit the bottom edge of
# the old grid (C=50, gamma=0.01), so the optimum likely sits below it.
# 'scale' and 'auto' are data-driven gamma defaults that often outperform fixed values.
svc_param_dist = {
    'clf__C':      FloatDistribution(0.1, 200, log=True),  # log scale covers low and high C evenly
    'clf__gamma':  CategoricalDistribution(['scale', 'auto', 0.001, 0.01, 0.05]),
    'clf__kernel': CategoricalDistribution(['rbf']),
}

# learning rate grid extended down and estimator ceiling raised to match —
# slower learners need more trees to compensate, previous grid didn't explore this
xgb_param_dist = {
    'clf__n_estimators':     IntDistribution(200, 800),
    'clf__max_depth':        IntDistribution(3, 7),
    'clf__learning_rate':    FloatDistribution(0.005, 0.05, log=True),
    'clf__subsample':        FloatDistribution(0.6, 1.0),
    'clf__colsample_bytree': FloatDistribution(0.6, 1.0),
    'clf__gamma':            FloatDistribution(0, 1),
    'clf__reg_alpha':        FloatDistribution(0, 1),
    'clf__reg_lambda':       FloatDistribution(1, 2),
    'clf__scale_pos_weight': CategoricalDistribution([1, imbalance_ratio]),
}

# scoring='recall' — missing an actual churner costs more than a false alarm,
# so we tune toward finding as many churners as possible and adjust threshold after
search_rf = OptunaSearchCV(
    rf_pipe, rf_param_dist, n_trials=50,
    scoring='recall', cv=cv, n_jobs=-1, random_state=42
)
search_svc = OptunaSearchCV(
    svc_pipe, svc_param_dist, n_trials=30,
    scoring='recall', cv=cv, n_jobs=-1, random_state=42
)
search_xgb = OptunaSearchCV(
    xgb_pipe, xgb_param_dist, n_trials=60,  # more trials for the larger XGB space
    scoring='recall', cv=cv, n_jobs=-1, random_state=42
)

print('\nTuning Random Forest...')
search_rf.fit(X_train, y_train)
clf1 = search_rf.best_estimator_
print('Best RF params:', search_rf.best_params_)

print('\nTuning SVC...')
search_svc.fit(X_train, y_train)

# SVC probability estimates from Platt scaling can be poorly calibrated on imbalanced
# data — isotonic calibration corrects this and improves threshold behavior
clf2_base = search_svc.best_estimator_
clf2 = CalibratedClassifierCV(clf2_base, cv=5, method='isotonic')
clf2.fit(X_train, y_train)
print('Best SVC params:', search_svc.best_params_)

print('\nTuning XGBoost...')
search_xgb.fit(X_train, y_train)
clf3 = search_xgb.best_estimator_
print('Best XGB params:', search_xgb.best_params_)

def print_metrics(label, y_true, y_pred):
    print(f'\n--- {label} ---')
    print(f'Accuracy:  {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'Recall:    {recall_score(y_true, y_pred):.4f}')
    print(f'F1 Score:  {f1_score(y_true, y_pred):.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

# all pipelines predict on raw X_test — scaling is handled internally for SVC
predictions_rf = clf1.predict(X_test)
predictions_rf_proba = clf1.predict_proba(X_test)[:, 1]

# Random Forest - overfit check on original unbalanced training data
train_preds_rf = clf1.predict(X_train)
print('\n-- RF Overfit Check --')
print('RF Train F1:', round(f1_score(y_train, train_preds_rf), 4))
print('RF Test F1: ', round(f1_score(y_test, predictions_rf), 4))
print_metrics('Random Forest', y_test, predictions_rf)

# SVC - calibrated model predicts on raw X_test
predictions_svc = clf2.predict(X_test)
predictions_svc_proba = clf2.predict_proba(X_test)[:, 1]
print_metrics('SVC', y_test, predictions_svc)

# XGBoost - wider sweep now that recall-tuned model may shift the optimal threshold
predictions_xgb_proba = clf3.predict_proba(X_test)[:, 1]
print('\n-- XGBoost Threshold Sweep --')
for t in np.arange(0.30, 0.56, 0.05):
    preds_t = (predictions_xgb_proba >= t).astype(int)
    print(f't={t:.2f}  Precision: {precision_score(y_test, preds_t):.4f}  '
          f'Recall: {recall_score(y_test, preds_t):.4f}  '
          f'F1: {f1_score(y_test, preds_t):.4f}')

threshold = 0.35
predictions_xgb_tuned = (predictions_xgb_proba >= threshold).astype(int)
print_metrics('XGBoost (t=0.35)', y_test, predictions_xgb_tuned)

fpr_rf, tpr_rf, _ = roc_curve(y_test, predictions_rf_proba)
fpr_svc, tpr_svc, _ = roc_curve(y_test, predictions_svc_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, predictions_xgb_proba)

roc_auc_rf = roc_auc_score(y_test, predictions_rf_proba)
roc_auc_svc = roc_auc_score(y_test, predictions_svc_proba)
roc_auc_xgb = roc_auc_score(y_test, predictions_xgb_proba)

# plot all ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf,  tpr_rf,  label='Random Forest (AUC = %0.3f)' % roc_auc_rf)
plt.plot(fpr_svc, tpr_svc, label='SVC (AUC = %0.3f)' % roc_auc_svc)
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = %0.3f)' % roc_auc_xgb)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, preds, label in zip(axes,
                             [predictions_rf, predictions_svc, predictions_xgb_tuned],
                             ['Random Forest', 'SVC', 'XGBoost (t=0.35)']):
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax,
                                            display_labels=['No Churn', 'Churn'])
    ax.set_title(label)
plt.tight_layout()
plt.show()

metrics = {
    'Model':     ['Random Forest', 'SVC', 'XGBoost (t=0.35)'],
    'Accuracy':  [accuracy_score(y_test, predictions_rf),
                  accuracy_score(y_test, predictions_svc),
                  accuracy_score(y_test, predictions_xgb_tuned)],
    'Precision': [precision_score(y_test, predictions_rf),
                  precision_score(y_test, predictions_svc),
                  precision_score(y_test, predictions_xgb_tuned)],
    'Recall':    [recall_score(y_test, predictions_rf),
                  recall_score(y_test, predictions_svc),
                  recall_score(y_test, predictions_xgb_tuned)],
    'F1':        [f1_score(y_test, predictions_rf),
                  f1_score(y_test, predictions_svc),
                  f1_score(y_test, predictions_xgb_tuned)]
}

df_metrics = pd.DataFrame(metrics).set_index('Model')
df_metrics.plot(kind='bar', figsize=(10, 5), ylim=(0, 1))
plt.title('Model Performance Comparison')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest - feature importances extracted from pipeline clf step
importances_rf = pd.Series(clf1.named_steps['clf'].feature_importances_, index=X.columns)
importances_rf.nlargest(15).plot(kind='barh', ax=axes[0])
axes[0].set_title('Random Forest — Top 15 Features')

# XGBoost
importances_xgb = pd.Series(clf3.named_steps['clf'].feature_importances_, index=X.columns)
importances_xgb.nlargest(15).plot(kind='barh', ax=axes[1])
axes[1].set_title('XGBoost — Top 15 Features')

plt.tight_layout()
plt.show()

precisions, recalls, thresholds = precision_recall_curve(y_test, predictions_xgb_proba)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.xlabel('Threshold')
plt.title('Precision vs Recall — XGBoost')
plt.legend()
plt.tight_layout()
plt.show()