from cleanchurn import *  # imports clean data df
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler


# PARAMETER SEARCH FUNCTION
def find_parameters(clf, model_num, score):
    # uses grid search to find best parameters when running models
    # score is the scoring function used. we used f1, accuracy, precision and recall
    # model_num can be index 1 to 3
    # clf can be clf1, clf2, clf3
    names = ['Random Forest', 'SVM', 'XGBoost']
    print('\n\n', names[model_num - 1])

    # Random Forest grid
    parameters1 = {'n_estimators': [10, 30, 50, 100, 200],
                   'max_depth': [5, 10, 30, 50, 100, None]}

    # SVM grid
    parameters2 = {'kernel': ['rbf', 'poly', 'sigmoid'],
                   'C': [0.1, 1, 10, 20],
                   'gamma': ['scale', 'auto']}

    # XGBoost grid
    parameters3 = {'n_estimators': [50, 100, 200],
                   'max_depth': [3, 5, 10],
                   'learning_rate': [0.01, 0.1, 0.2],
                   'subsample': [0.8, 1.0],
                   'colsample_bytree': [0.8, 1.0]}

    param = [parameters1, parameters2, parameters3]

    # Choose correct dataset
    if model_num == 2:
        X_data = X_train_bal_scaled
        y_data = y_train_bal_scaled
    else:
        X_data = X_train_bal
        y_data = y_train_bal

    grid = GridSearchCV(clf, param[model_num - 1], cv=5, scoring=score, n_jobs=-1)
    grid.fit(X_data, y_data)

    print('Best parameters:', grid.best_params_)
    print(f'Best {score} score:', grid.best_score_)

    return grid.best_estimator_


# DATA PREPROCESSING
# convert categorical to integer representations using replace for outcome and
# one hot encoding for attributes
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})

Xcat = pd.get_dummies(df[['gender', 'Partner', 'Dependents', 'PhoneService',
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']]).astype(int)
Xnum = df[['MonthlyCharges', 'TotalCharges', 'tenure']]
X = pd.concat([Xcat, Xnum], axis=1)

print(X.info())

# Split data
y = df['Churn'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# Scale for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE balancing
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
X_train_bal_scaled, y_train_bal_scaled = smote.fit_resample(X_train_scaled, y_train)
print(y_train_bal.value_counts())


# MODELS
clf1 = RandomForestClassifier(random_state=1)
clf2 = SVC(probability=True, random_state=1)
clf3 = XGBClassifier(random_state=1, eval_metric='logloss')


# METRICS FUNCTION
def print_metrics(label, y_true, y_pred):
    print(f'\n--- {label} ---')
    print(f'Accuracy:  {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'Recall:    {recall_score(y_true, y_pred):.4f}')
    print(f'F1 Score:  {f1_score(y_true, y_pred):.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))


# HYPERPARAMETER TUNING
scores = ['accuracy', 'precision', 'recall', 'f1']

best_rf = find_parameters(clf1, 1, 'f1')
best_svc = find_parameters(clf2, 2, 'f1')
best_xgb = find_parameters(clf3, 3, 'f1')


# TRAIN BEST MODELS
# Random Forest
best_rf.fit(X_train_bal, y_train_bal)
pred_rf = best_rf.predict(X_test)
proba_rf = best_rf.predict_proba(X_test)[:, 1]
print_metrics('Random Forest', y_test, pred_rf)

# SVC
best_svc.fit(X_train_bal_scaled, y_train_bal_scaled)
pred_svc = best_svc.predict(X_test_scaled)
proba_svc = best_svc.predict_proba(X_test_scaled)[:, 1]
print_metrics('SVC', y_test, pred_svc)

# XGBoost
best_xgb.fit(X_train_bal, y_train_bal)
pred_xgb = best_xgb.predict(X_test)
proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
print_metrics('XGBoost', y_test, pred_xgb)


# ROC CURVES
fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
fpr_svc, tpr_svc, _ = roc_curve(y_test, proba_svc)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, proba_xgb)

roc_auc_rf = roc_auc_score(y_test, proba_rf)
roc_auc_svc = roc_auc_score(y_test, proba_svc)
roc_auc_xgb = roc_auc_score(y_test, proba_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
plt.plot(fpr_svc, tpr_svc, label=f'SVC (AUC = {roc_auc_svc:.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')

plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()