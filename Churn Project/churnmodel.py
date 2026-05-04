from cleanchurn import *  #imports clean data df
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

#convert categorical to integer representations using replace for outcome and
# one hot encoding for attributes
df['Churn'] = df['Churn'].replace({'Yes':1,'No':0})
Xcat = pd.get_dummies(df[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']]).astype(int)
Xnum = df[['MonthlyCharges', 'TotalCharges','tenure']]
X = pd.concat([Xcat, Xnum], axis=1)

print(X.info())

#split data
y = df['Churn'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# Scale features for SVC
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balance training sets
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)           # RF, XGB (unscaled)
X_train_bal_scaled, y_train_bal_scaled = smote.fit_resample(X_train_scaled, y_train)  # SVC (scaled)
print(y_train_bal.value_counts())

clf1 = RandomForestClassifier(random_state=1)
clf2 = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=1)
clf3 = XGBClassifier(random_state=1)

def print_metrics(label, y_true, y_pred):
    print(f'\n--- {label} ---')
    print(f'Accuracy:  {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'Recall:    {recall_score(y_true, y_pred):.4f}')
    print(f'F1 Score:  {f1_score(y_true, y_pred):.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

# Random Forest — fit and evaluate on unscaled data
clf1.fit(X_train_bal, y_train_bal)
predictions_rf = clf1.predict(X_test)
print_metrics('Random Forest', y_test, predictions_rf)

fpr_rf, tpr_rf, _ = roc_curve(y_test, predictions_rf)
roc_auc_rf = roc_auc_score(y_test, predictions_rf)

# SVC — fit and evaluate on scaled data
clf2.fit(X_train_bal_scaled, y_train_bal_scaled)
predictions_svc = clf2.predict(X_test_scaled)
predictions_svc_proba = clf2.predict_proba(X_test_scaled)[:, 1]
print_metrics('SVC', y_test, predictions_svc)

fpr_svc, tpr_svc, _ = roc_curve(y_test, predictions_svc_proba)
roc_auc_svc = roc_auc_score(y_test, predictions_svc_proba)

# XGBoost — fit and evaluate on unscaled data (XGB handles raw features well)
clf3.fit(X_train_bal, y_train_bal)
predictions_xgb = clf3.predict(X_test)
predictions_xgb_proba = clf3.predict_proba(X_test)[:, 1]
print_metrics('XGBoost', y_test, predictions_xgb)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, predictions_xgb_proba)
roc_auc_xgb = roc_auc_score(y_test, predictions_xgb_proba)

# Plot all ROC curves
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
                             [predictions_rf, predictions_svc, predictions_xgb],
                             ['Random Forest', 'SVC', 'XGBoost']):
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax,
                                            display_labels=['No Churn', 'Churn'])
    ax.set_title(label)
plt.tight_layout()
plt.show()

metrics = {
    'Model':     ['Random Forest', 'SVC', 'XGBoost'],
    'Accuracy':  [accuracy_score(y_test, predictions_rf),
                  accuracy_score(y_test, predictions_svc),
                  accuracy_score(y_test, predictions_xgb)],
    'Precision': [precision_score(y_test, predictions_rf),
                  precision_score(y_test, predictions_svc),
                  precision_score(y_test, predictions_xgb)],
    'Recall':    [recall_score(y_test, predictions_rf),
                  recall_score(y_test, predictions_svc),
                  recall_score(y_test, predictions_xgb)],
    'F1':        [f1_score(y_test, predictions_rf),
                  f1_score(y_test, predictions_svc),
                  f1_score(y_test, predictions_xgb)]
}

df_metrics = pd.DataFrame(metrics).set_index('Model')
df_metrics.plot(kind='bar', figsize=(10, 5), ylim=(0, 1))
plt.title('Model Performance Comparison')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest
importances_rf = pd.Series(clf1.feature_importances_, index=X.columns)
importances_rf.nlargest(15).plot(kind='barh', ax=axes[0])
axes[0].set_title('Random Forest — Top 15 Features')

# XGBoost
importances_xgb = pd.Series(clf3.feature_importances_, index=X.columns)
importances_xgb.nlargest(15).plot(kind='barh', ax=axes[1])
axes[1].set_title('XGBoost — Top 15 Features')

plt.tight_layout()
plt.show()